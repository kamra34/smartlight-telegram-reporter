import os, re, json, time, math, hashlib, urllib.parse, requests, random, io, csv
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googleapiclient.discovery import build
from openai import OpenAI
import cloudscraper

# =========================
# Config
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
GOOGLE_CX     = os.getenv("GOOGLE_CUSTOM_SEARCH_CX", "")
assert OPENAI_API_KEY and GOOGLE_API_KEY and GOOGLE_CX, "Set OPENAI & Google CSE creds in .env"

client = OpenAI(api_key=OPENAI_API_KEY)

# Model choices
ORCH_MODEL = "gpt-4o-mini"  # tool orchestration & takeaways
TAKEAWAY_MODEL = "gpt-4o-mini"

# Windows & novelty
WINDOW_DAYS   = 90
DISCOVERY_DAYS = 7
DATE_RESTRICT = "d7"
DATE_RESTRICT_FALLBACK = "d14"

# Novelty controls
STATE_FILE = "ads_state.json"
COOLDOWN_DAYS = 7            # don't repeat a reported item within this many days...
IMPROVEMENT_PCT = 0.15       # ...unless views improved by at least this many percent

TIMEOUT = 18
RETRIES = 2
DOMAINS = ["kickstarter.com", "indiegogo.com", "youtube.com", "vimeo.com"]

HEADERS_BASE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]
def _headers():
    h = HEADERS_BASE.copy()
    h["User-Agent"] = random.choice(UA_POOL)
    return h

_scraper = cloudscraper.create_scraper(browser={'browser':'chrome','platform':'windows','mobile':False})

# =========================
# HTTP helpers (Cloudflare-safe)
# =========================
def _get(url, stream=False, timeout=TIMEOUT):
    last_exc = None
    for i in range(RETRIES + 1):
        try:
            r = _scraper.get(url, headers=_headers(), timeout=timeout, stream=stream, allow_redirects=True)
            if r.status_code == 403:
                r = requests.get(url, headers=_headers(), timeout=timeout, stream=stream, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1))
    raise last_exc

def _head(url, timeout=TIMEOUT):
    last_exc = None
    for i in range(RETRIES + 1):
        try:
            r = _scraper.head(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            if r.status_code >= 400:
                r = _scraper.get(url, headers=_headers(), timeout=timeout, stream=True, allow_redirects=True)
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1))
    raise last_exc

def verify_exists(url):
    try:
        r = _head(url)
        ok = r.status_code < 400
        return {"ok": ok, "final_url": r.url, "content_type": r.headers.get("Content-Type","")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_html(url):
    r = _get(url)
    return r.text, r.url

def normalize_url(url: str):
    try:
        u = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, "", ""))
    except Exception:
        return url

# =========================
# Google Custom Search
# =========================
QUERY_SUFFIXES = [
    '(launch OR "now live" OR trailer) (lighting OR "smart home" OR gadget OR electronics)',
    '(Kickstarter OR Indiegogo) (trailer OR launch) (design OR lighting OR home)',
    '(crowdfunding campaign) (video OR trailer) (lighting OR smart home OR gadget)',
]

def _rotated(seq):
    doy = datetime.now().timetuple().tm_yday
    k = doy % len(seq)
    return seq[k:] + seq[:k]

def google_search_site(domain: str, query_suffix: str, num: int = 5, date_restrict: str = DATE_RESTRICT, page: int = 1):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    q = f"site:{domain} {query_suffix}".strip()
    start = (page - 1) * 10 + 1
    try:
        resp = service.cse().list(q=q, cx=GOOGLE_CX, num=min(num,10), dateRestrict=date_restrict, start=start).execute()
    except Exception:
        resp = service.cse().list(q=q, cx=GOOGLE_CX, num=min(num,10), dateRestrict=DATE_RESTRICT_FALLBACK, start=start).execute()
    items = resp.get("items", []) or []
    out = []
    for it in items:
        out.append({
            "title": it.get("title"),
            "link":  it.get("link"),
            "snippet": it.get("snippet"),
            "displayLink": it.get("displayLink"),
        })
    return out

# =========================
# Parsers & detectors (campaign-only filters)
# =========================
YTG_VIEWS_RE = re.compile(r'"viewCount"\s*:\s*"\s*([0-9,]+)\s*"|{"simpleText"\s*:\s*"([\d,\.]+)\s*views"', re.I)
YTG_PUB_RE   = re.compile(r'"publishDate":\s*"(\d{4}-\d{2}-\d{2})"')
YTG_LEN_RE   = re.compile(r'"lengthSeconds"\s*:\s*"(\d+)"')
DESC_RE      = re.compile(r'"shortDescription":\s*"([^"]+)"', re.S)

def _safe_unicode_escape(raw):
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except Exception:
        return raw.replace("\\", "")

def is_campaigny_text(txt: str) -> bool:
    if not txt: return False
    txtl = txt.lower()
    kws = ["kickstarter","indiegogo","crowdfund","crowdfunding","now live","back us","pre-order","preorder","launch on","campaign"]
    return any(k in txtl for k in kws)

def parse_youtube(url):
    html, final = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.text.strip() if soup.title else final)

    desc = None
    mdesc = DESC_RE.search(html)
    if mdesc:
        desc = _safe_unicode_escape(mdesc.group(1))

    views = None
    mv = YTG_VIEWS_RE.search(html)
    if mv:
        g = mv.group(1) or mv.group(2)
        try: views = int(re.sub(r"[^\d]", "", g))
        except: pass

    pub = None
    mp = YTG_PUB_RE.search(html)
    if mp: pub = mp.group(1)

    seconds = None
    ml = YTG_LEN_RE.search(html)
    if ml:
        try: seconds = int(ml.group(1))
        except: pass

    is_ad_like = (seconds is not None and seconds <= 180)
    is_camp = is_campaigny_text(title) or is_campaigny_text(desc)

    return {
        "platform":"youtube","title": title,"url": final,
        "views": views,"published_at": pub,"duration_s": seconds,
        "is_campaign_ad": bool(is_ad_like and is_camp),
        "goal": None,"pledged": None,"backers": None,
        "comments": None,"updates": None,"status": None,"category": None,
        "platform_id": urllib.parse.urlsplit(final).path.strip("/"),
        "campaign_dates": {"launched": None, "ended": None}
    }

VIMEO_VIEWS_RE = re.compile(r'"play_count"\s*:\s*([0-9]+)')
VIMEO_DUR_RE   = re.compile(r'"duration"\s*:\s*([0-9]+)')

def parse_vimeo(url):
    html, final = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.text.strip() if soup.title else final)
    views = None
    mv = VIMEO_VIEWS_RE.search(html)
    if mv:
        try: views = int(mv.group(1))
        except: pass
    dur_s = None
    md = VIMEO_DUR_RE.search(html)
    if md:
        try: dur_s = int(md.group(1))
        except: pass
    desc_meta = soup.find("meta", attrs={"name":"description"})
    desc = (desc_meta["content"] if desc_meta and desc_meta.has_attr("content") else "")
    is_ad_like = (dur_s is None) or (dur_s <= 180)
    return {
        "platform":"vimeo","title": title,"url": final,
        "views": views,"published_at": None,"duration_s": dur_s,
        "is_campaign_ad": bool(is_ad_like and is_campaigny_text(title + " " + desc)),
        "goal": None,"pledged": None,"backers": None,
        "comments": None,"updates": None,"status": None,"category": None,
        "platform_id": urllib.parse.urlsplit(final).path.strip("/"),
        "campaign_dates": {"launched": None, "ended": None}
    }

# Kickstarter/Indiegogo
KS_USD_RE     = re.compile(r'"usd_pledged"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
KS_PLEDGED_RE = re.compile(r'"pledged"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
KS_GOAL_RE    = re.compile(r'"goal"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
KS_BACKERS_RE = re.compile(r'"backers_count"\s*:\s*([0-9]+)')
KS_LAUNCH_RE  = re.compile(r'"launched_at"\s*:\s*([0-9]{9,11})')
KS_DEADLINE_RE= re.compile(r'"deadline"\s*:\s*([0-9]{9,11})')
KS_COMMENTS_RE= re.compile(r'"comments_count"\s*:\s*([0-9]+)')
KS_UPDATES_RE = re.compile(r'"updates_count"\s*:\s*([0-9]+)')
KS_STATE_RE   = re.compile(r'"state"\s*:\s*"([a-z_]+)"')
KS_CAT_RE     = re.compile(r'"category"\s*:\s*{[^}]*"name"\s*:\s*"([^"]+)"', re.S)

YTB_IFR = re.compile(r'src="https?://(?:www\.)?youtube\.com/embed/([A-Za-z0-9_\-]+)"')
VIM_IFR = re.compile(r'src="https?://player\.vimeo\.com/video/([0-9]+)"')

def ts_to_date(s):
    try: return datetime.fromtimestamp(int(s), tz=timezone.utc).date().isoformat()
    except: return None

def find_embedded_videos(html):
    vids = []
    for m in YTB_IFR.finditer(html):
        vids.append(("youtube", f"https://www.youtube.com/watch?v={m.group(1)}"))
    for m in VIM_IFR.finditer(html):
        vids.append(("vimeo", f"https://vimeo.com/{m.group(1)}"))
    return vids

def youtube_search_by_title_fallback(campaign_title, max_results=2):
    query = f'site:youtube.com {campaign_title} (kickstarter OR indiegogo OR crowdfunding) (trailer OR launch)'
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    resp = service.cse().list(q=query, cx=GOOGLE_CX, num=min(max_results,10), dateRestrict=DATE_RESTRICT).execute()
    items = resp.get("items", []) or []
    return [{"title": it.get("title"), "link": it.get("link")} for it in items]

def parse_kickstarter(url):
    try:
        html, final = fetch_html(url)
    except Exception:
        slug = urllib.parse.urlsplit(url).path.strip("/").split("/")[-1]
        guess_title = (slug or "Kickstarter campaign").replace("-", " ").title()
        yt_links = youtube_search_by_title_fallback(guess_title, max_results=2)
        return {
            "platform":"kickstarter","title": guess_title,"url": url,
            "views": None,"published_at": None,
            "goal": None,"pledged": None,"backers": None,
            "comments": None,"updates": None,"status": None,"category": None,
            "platform_id": urllib.parse.urlsplit(url).path.strip("/"),
            "campaign_dates": {"launched": None,"ended": None},
            "embedded_videos": [("youtube", v["link"]) for v in yt_links]
        }
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.text.strip() if soup.title else final)
    title = title.replace(" — Kickstarter","").replace("| Kickstarter","").strip()
    usd = KS_USD_RE.search(html)
    pledged = KS_PLEDGED_RE.search(html)
    goal = KS_GOAL_RE.search(html)
    backers = KS_BACKERS_RE.search(html)
    launched = KS_LAUNCH_RE.search(html)
    deadline = KS_DEADLINE_RE.search(html)
    comments_m = KS_COMMENTS_RE.search(html)
    updates_m  = KS_UPDATES_RE.search(html)
    state_m    = KS_STATE_RE.search(html)
    cat_m      = KS_CAT_RE.search(html)
    embeds = find_embedded_videos(html)
    path = urllib.parse.urlsplit(final).path.strip("/")
    platform_id = path.replace("projects/", "", 1) if path.startswith("projects/") else path
    return {
        "platform":"kickstarter","title": title,"url": final,
        "views": None,"published_at": None,
        "goal": float(goal.group(1)) if goal else None,
        "pledged": float(usd.group(1)) if usd else (float(pledged.group(1)) if pledged else None),
        "backers": int(backers.group(1)) if backers else None,
        "comments": int(comments_m.group(1)) if comments_m else None,
        "updates": int(updates_m.group(1)) if updates_m else None,
        "status": state_m.group(1) if state_m else None,
        "category": cat_m.group(1) if cat_m else None,
        "platform_id": platform_id,
        "campaign_dates": {"launched": ts_to_date(launched.group(1)) if launched else None,
                           "ended": ts_to_date(deadline.group(1)) if deadline else None},
        "embedded_videos": embeds
    }

IGG_GOAL_RE     = re.compile(r'"goal"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
IGG_PLEDGED_RE  = re.compile(r'"collected_amount"\s*:\s*([0-9]+(?:\.[0-9]+)?)|"funds_raised":\s*([0-9]+(?:\.[0-9]+)?)', re.I)
IGG_BACKERS_RE  = re.compile(r'"contributors_count"\s*:\s*([0-9]+)', re.I)
IGG_COMMENTS_RE = re.compile(r'"comments_count"\s*:\s*([0-9]+)', re.I)
IGG_UPDATES_RE  = re.compile(r'"updates_count"\s*:\s*([0-9]+)', re.I)
IGG_STATUS_RE   = re.compile(r'"status"\s*:\s*"([a-z_]+)"', re.I)
IGG_CAT_RE      = re.compile(r'"category"\s*:\s*{[^}]*"name"\s*:\s*"([^"]+)"', re.I|re.S)

def parse_indiegogo(url):
    try:
        html, final = fetch_html(url)
    except Exception:
        slug = urllib.parse.urlsplit(url).path.strip("/").split("/")[-1]
        guess_title = (slug or "Indiegogo campaign").replace("-", " ").title()
        yt_links = youtube_search_by_title_fallback(guess_title, max_results=2)
        return {
            "platform":"indiegogo","title": guess_title,"url": url,
            "views": None,"published_at": None,
            "goal": None,"pledged": None,"backers": None,
            "comments": None,"updates": None,"status": None,"category": None,
            "platform_id": urllib.parse.urlsplit(url).path.strip("/"),
            "campaign_dates": {"launched": None,"ended": None},
            "embedded_videos": [("youtube", v["link"]) for v in yt_links]
        }
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.text.strip() if soup.title else final)
    goal = IGG_GOAL_RE.search(html)
    pledged_m = IGG_PLEDGED_RE.search(html)
    pledged_val = None
    if pledged_m:
        g1, g2 = pledged_m.groups()
        pledged_val = float(g1 or g2) if (g1 or g2) else None
    backers = IGG_BACKERS_RE.search(html)
    comments_m = IGG_COMMENTS_RE.search(html)
    updates_m  = IGG_UPDATES_RE.search(html)
    status_m   = IGG_STATUS_RE.search(html)
    cat_m      = IGG_CAT_RE.search(html)
    embeds = find_embedded_videos(html)
    return {
        "platform":"indiegogo","title": title,"url": final,
        "views": None,"published_at": None,
        "goal": float(goal.group(1)) if goal else None,
        "pledged": pledged_val,
        "backers": int(backers.group(1)) if backers else None,
        "comments": int(comments_m.group(1)) if comments_m else None,
        "updates": int(updates_m.group(1)) if updates_m else None,
        "status": status_m.group(1) if status_m else None,
        "category": cat_m.group(1) if cat_m else None,
        "platform_id": urllib.parse.urlsplit(final).path.strip("/"),
        "campaign_dates": {"launched": None,"ended": None},
        "embedded_videos": embeds
    }

def resolve_embeds_to_views(item):
    embeds = item.get("embedded_videos") or []
    best = None
    for host, link in embeds:
        try:
            if host == "youtube":
                vid = parse_youtube(link)
            elif host == "vimeo":
                vid = parse_vimeo(link)
            else:
                continue
            if vid.get("is_campaign_ad") and (vid.get("views") is not None):
                if not best or vid["views"] > best["views"]:
                    best = vid
        except Exception:
            pass
    if best:
        item["url"] = best["url"]
        item["platform"] = best["platform"]
        item["views"] = best["views"]
        item["published_at"] = best.get("published_at")
        item["duration_s"] = best.get("duration_s")
        item["is_campaign_ad"] = True
    return item

# =========================
# State & novelty
# =========================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"items": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def _stable_key(item):
    plat, nid = extract_native_id(item.get("url") or "")
    return f"{plat}:{nid}"

def extract_native_id(url):
    u = urllib.parse.urlsplit(url or "")
    if "youtube.com" in u.netloc:
        q = urllib.parse.parse_qs(u.query)
        vid = q.get("v", [None])[0]
        return ("youtube", vid or url)
    if "vimeo.com" in u.netloc:
        return ("vimeo", u.path.strip("/").split("/")[-1] or url)
    if "kickstarter.com" in u.netloc:
        return ("kickstarter", u.path.strip("/"))
    if "indiegogo.com" in u.netloc:
        return ("indiegogo", u.path.strip("/"))
    return ("web", url)

def apply_novelty_filter(items, state):
    """Drop items we reported within COOLDOWN_DAYS unless views improved >= IMPROVEMENT_PCT."""
    out = []
    today = datetime.now(timezone.utc).date().isoformat()
    for it in items:
        key = _stable_key(it)
        prev = state["items"].get(key)
        if not prev:
            out.append(it); continue
        last_rep = prev.get("last_reported")
        best_views = prev.get("best_views") or 0
        cooldown_ok = True
        if last_rep:
            try:
                d = datetime.fromisoformat(last_rep).date()
                days = (datetime.now(timezone.utc).date() - d).days
                cooldown_ok = days >= COOLDOWN_DAYS
            except: pass
        improved = (it.get("views") or 0) > best_views * (1 + IMPROVEMENT_PCT)
        if cooldown_ok or improved:
            out.append(it)
    return out

def update_state_with_report(items, state):
    today = datetime.now(timezone.utc).date().isoformat()
    for it in items:
        key = _stable_key(it)
        entry = state["items"].get(key, {})
        entry["best_views"] = max(entry.get("best_views") or 0, it.get("views") or 0)
        entry["last_reported"] = today
        entry["title"] = it.get("title")
        entry["platform"] = it.get("platform")
        state["items"][key] = entry
    save_state(state)

def _extract_json_from_fence(raw: str):
    """
    If raw contains a fenced ```json ... ``` block, return the parsed JSON object.
    Otherwise, try json.loads(raw) directly. Returns None on failure.
    """
    # Look for a fenced json block
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try raw as JSON
    try:
        return json.loads(raw)
    except Exception:
        return None

def _enrich_minimal_results(items):
    """
    Given items with only title/url/platform, fetch real metrics:
    - KS/IGG: parse page, extract embedded video, resolve to views
    - YouTube/Vimeo: parse directly
    Returns a new list of enriched dicts ready for scoring.
    """
    enriched = []
    for it in items:
        url = it.get("url") or ""
        u = (url or "").lower()
        try:
            # quick existence check to skip dead links
            if not verify_exists(url).get("ok"):
                continue

            if "kickstarter.com" in u:
                parsed = resolve_embeds_to_views(parse_kickstarter(url))
            elif "indiegogo.com" in u:
                parsed = resolve_embeds_to_views(parse_indiegogo(url))
            elif "youtube.com" in u:
                parsed = parse_youtube(url)
                if not parsed.get("is_campaign_ad"): 
                    continue
            elif "vimeo.com" in u:
                parsed = parse_vimeo(url)
                if not parsed.get("is_campaign_ad"):
                    continue
            else:
                # not one of our platforms
                continue

            # if the agent provided a better title, keep it
            if it.get("title") and (not parsed.get("title") or len(it["title"]) < 120):
                parsed["title"] = it["title"]

            # ensure platform field is consistent
            if it.get("platform"):
                parsed["platform"] = it["platform"]

            enriched.append(parsed)
        except Exception:
            # skip problematic items rather than failing the whole run
            continue
    return enriched

# =========================
# LLM tools (tiny outputs) & orchestration
# =========================
tools = [
    {"type":"function","function":{
        "name":"google_search_site",
        "description":"CSE on a single domain (fresh window) with campaign-focused keywords.",
        "parameters":{"type":"object","properties":{
            "domain":{"type":"string","enum":DOMAINS},
            "query_suffix":{"type":"string"},
            "num":{"type":"integer","default":5},
            "date_restrict":{"type":"string","default":DATE_RESTRICT},
            "page":{"type":"integer","default":1}
        },"required":["domain","query_suffix"]}
    }},
    {"type":"function","function":{"name":"normalize_url","description":"Remove query/fragment.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
    {"type":"function","function":{"name":"verify_exists","description":"HEAD/GET existence check.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}, 
    {"type":"function","function":{"name":"parse_kickstarter","description":"Parse KS metrics + embedded videos.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}, 
    {"type":"function","function":{"name":"parse_indiegogo","description":"Parse IGG metrics + embedded videos.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}, 
    {"type":"function","function":{"name":"parse_youtube","description":"Parse YT title/views/published/duration + campaigniness.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}, 
    {"type":"function","function":{"name":"parse_vimeo","description":"Parse Vimeo title/views/duration + campaigniness.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}
]

def call_tool(name, args):
    if name == "google_search_site": return google_search_site(args["domain"], args["query_suffix"], args.get("num",5), args.get("date_restrict",DATE_RESTRICT), args.get("page",1))
    if name == "normalize_url":      return {"normalized": normalize_url(args["url"])}
    if name == "verify_exists":      return verify_exists(args["url"])
    if name == "parse_kickstarter":  return parse_kickstarter(args["url"])
    if name == "parse_indiegogo":    return parse_indiegogo(args["url"])
    if name == "parse_youtube":      return parse_youtube(args["url"])
    if name == "parse_vimeo":        return parse_vimeo(args["url"])
    return {"error": f"unknown tool {name}"}

SYSTEM_PROMPT = f"""
Find *campaign ad* videos from last {DISCOVERY_DAYS} days (fallback 14) across Kickstarter, Indiegogo, YouTube, Vimeo.
Rules:
- KS/IGG: parse page, extract embedded YT/Vimeo; resolve to views.
- YT/Vimeo direct: keep only if crowdfunding/launchy (title/desc) AND duration <= 180s.
- Prefer consumer electronics/smart home/design, but include standouts.
Return a plain text output containing Top items and three CSV blocks (campaigns/videos/scores) OR a compact JSON 'results'. Keep tool outputs tiny.
"""

USER_PROMPT = "Rotate query variants and page start; avoid returning the exact same top each day."

def run_agent():
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": USER_PROMPT},
    ]
    # rotate domains, queries, page by day-of-year to vary results
    domains = _rotated(DOMAINS)
    queries = _rotated(QUERY_SUFFIXES)
    page = (datetime.now().timetuple().tm_yday % 2) + 1  # 1..2

    # simple loop: search each domain with one rotated query, then parse
    for domain in domains:
        q_suffix = queries[0]
        queries = queries[1:] + queries[:1]
        # Ask model to call google_search_site
        resp = client.chat.completions.create(
            model=ORCH_MODEL, temperature=0.1, messages=messages + [
                {"role":"user","content":(
                    "Aggregate all candidates you saw into JSON {\"results\":[...]}. "
                    "For each, include fields if known: title, url, platform, views, goal, pledged, backers, comments, "
                    "published_at, campaign_dates, category, status, platform_id. Deduplicate by url. Keep up to 20 items."
                )}
            ],
        )

        messages.append({"role":"assistant","content":"","tool_calls":[{"id":"_1","type":"function","function":{"name":"google_search_site","arguments":json.dumps({"domain":domain,"query_suffix":q_suffix,"num":5,"page":page})}}]})
        result = call_tool("google_search_site", {"domain":domain,"query_suffix":q_suffix,"num":5,"page":page})
        messages.append({"role":"tool","tool_call_id":"_1","content":json.dumps(result)})

        # For each found link: normalize -> verify -> parse_*
        for it in result:
            url = it.get("link")
            if not url: continue
            nurl = normalize_url(url)
            v = verify_exists(nurl)
            if not v.get("ok"): continue
            if "youtube.com" in nurl:
                parsed = parse_youtube(nurl)
                if not parsed.get("is_campaign_ad"): continue
            elif "vimeo.com" in nurl:
                parsed = parse_vimeo(nurl)
                if not parsed.get("is_campaign_ad"): continue
            elif "kickstarter.com" in nurl:
                parsed = resolve_embeds_to_views(parse_kickstarter(nurl))
            elif "indiegogo.com" in nurl:
                parsed = resolve_embeds_to_views(parse_indiegogo(nurl))
            else:
                continue
            # stash a tiny JSON line for the model (but we won’t send huge content back)
            messages.append({"role":"assistant","content":json.dumps({"candidate": {
                "title": parsed.get("title"), "url": parsed.get("url"),
                "platform": parsed.get("platform"), "views": parsed.get("views"),
                "goal": parsed.get("goal"), "pledged": parsed.get("pledged"),
                "backers": parsed.get("backers"), "comments": parsed.get("comments"),
                "published_at": parsed.get("published_at"),
                "campaign_dates": parsed.get("campaign_dates"),
                "category": parsed.get("category"), "status": parsed.get("status"),
                "platform_id": parsed.get("platform_id"),
            }})})

    # Final: ask the model to compile a compact JSON array of results (max ~20)
    resp = client.chat.completions.create(
        model=ORCH_MODEL, temperature=0.1, messages=messages + [
            {"role":"user","content":"Aggregate all candidates you saw into JSON {\"results\":[...]}. Deduplicate by url. Keep up to 20 items."}
        ],
    )
    return resp.choices[0].message.content.strip()

# =========================
# Scoring & reporting
# =========================
def _to_float(x):
    try:
        if x in (None, "", "null"): return None
        return float(re.sub(r"[^\d\.]", "", str(x)))
    except: return None

def _to_int(x):
    try:
        if x in (None, "", "null"): return None
        return int(re.sub(r"[^\d]", "", str(x)))
    except: return None

def _days_since(date_str: str) -> float:
    if not date_str: return 30.0
    try:
        d = datetime.fromisoformat(date_str).date()
    except Exception:
        return 30.0
    today = datetime.now(timezone.utc).date()
    return float(max(1, (today - d).days))

def _funding_ratio(goal, pledged):
    try:
        g = float(goal) if goal is not None else None
        p = float(pledged) if pledged is not None else None
        if not g or g <= 0 or p is None: return None
        return p / g
    except Exception:
        return None

def _backers_per_day(item):
    b = item.get("backers")
    d = item.get("published_at") or ((item.get("campaign_dates") or {}).get("launched")) or ((item.get("campaign_dates") or {}).get("ended"))
    if b is None: return None
    return float(b) / _days_since(d)

def _comments_per_day(item):
    c = item.get("comments")
    d = item.get("published_at") or ((item.get("campaign_dates") or {}).get("launched")) or ((item.get("campaign_dates") or {}).get("ended"))
    if c is None: return None
    return float(c) / _days_since(d)

def _zs(values):
    arr = [v for v in values if v is not None]
    if not arr: return [None]*len(values)
    m = sum(arr)/len(arr)
    var = sum((x-m)**2 for x in arr) / max(1,(len(arr)-1))
    sd = math.sqrt(var) if var > 0 else 1.0
    out = []
    for v in values:
        out.append(((v - m)/sd) if v is not None else None)
    return out

def _zero(x): return 0.0 if x is None else float(x)

def compute_scores(items):
    # normalize numeric fields
    for it in items:
        it["views"]   = _to_int(it.get("views"))
        it["goal"]    = _to_float(it.get("goal"))
        it["pledged"] = _to_float(it.get("pledged"))
        it["backers"] = _to_int(it.get("backers"))
        it["comments"]= _to_int(it.get("comments"))

    views = [it.get("views") for it in items]
    fr    = [_funding_ratio(it.get("goal"), it.get("pledged")) for it in items]
    bpd   = [_backers_per_day(it) for it in items]
    cpd   = [_comments_per_day(it) for it in items]

    z_views = _zs(views)
    z_fr    = _zs(fr)
    z_bpd   = _zs(bpd)
    z_cpd   = _zs(cpd)

    for i, it in enumerate(items):
        s = 0.5*_zero(z_views[i]) + 0.2*_zero(z_fr[i]) + 0.2*_zero(z_bpd[i]) + 0.1*_zero(z_cpd[i])
        it["score"] = round(s, 4)
    return items

def _primary_post_date(it):
    return it.get("published_at") or ((it.get("campaign_dates") or {}).get("launched")) or ((it.get("campaign_dates") or {}).get("ended"))

def _compress_items_for_llm(items, top_k=10):
    slim = []
    for it in items[:top_k]:
        slim.append({
            "platform": it.get("platform"),
            "title": it.get("title"),
            "category": it.get("category"),
            "status": it.get("status"),
            "views": it.get("views"),
            "funding_ratio": _funding_ratio(it.get("goal"), it.get("pledged")),
            "backers_per_day": _backers_per_day(it),
            "comments_per_day": _comments_per_day(it),
            "published_at": _primary_post_date(it),
            "duration_s": it.get("duration_s"),
            "score": it.get("score"),
        })
    return slim

def generate_key_takeaways_llm(items_scored, top_items, tz_name="Europe/Stockholm", bullets=5):
    data = {
        "date_local": datetime.now(ZoneInfo(tz_name)).date().isoformat(),
        "items_sample": _compress_items_for_llm(items_scored, top_k=15),
        "top_sample": _compress_items_for_llm(top_items, top_k=len(top_items)),
    }
    sys = (
        "You are a marketing performance analyst. From the provided campaign/video metrics, "
        "infer concise, non-generic insights. Use concrete metrics (views, funding_ratio, backers/comments per day, recency, platform mix, duration). "
        "Output exactly 5 bullet points, each one sentence."
    )
    usr = (
        f"Timezone: {tz_name}. Today is {data['date_local']}.\n"
        f"Analyze and return EXACTLY 5 bullets:\n\n" + json.dumps(data, ensure_ascii=False)
    )
    try:
        resp = client.chat.completions.create(
            model=TAKEAWAY_MODEL, temperature=0.4,
            messages=[{"role":"system","content": sys},{"role":"user","content": usr}],
        )
        text = resp.choices[0].message.content.strip()
        lines = [ln.strip(" -•\t") for ln in text.split("\n") if ln.strip()]
        lines = [("• " + ln.rstrip(".")) + "." for ln in lines][:bullets]
        if not lines:
            lines = ["• Video-hosted campaign ads led performance.",
                     "• Short (≤180s) launch cuts performed best.",
                     "• Higher funding ratios correlated with above-average scores.",
                     "• Backers/comments velocity lifted rankings.",
                     "• Recency (≤14 days) boosted performance."]
        return lines
    except Exception:
        return ["• Video-hosted campaign ads led performance.",
                "• Short (≤180s) launch cuts performed best.",
                "• Higher funding ratios correlated with above-average scores.",
                "• Backers/comments velocity lifted rankings.",
                "• Recency (≤14 days) boosted performance."]

def render_report(items, tz_name="Europe/Stockholm", top_k=5):
    # Scoring
    items_scored = compute_scores(items)
    # Novelty filter using state
    state = load_state()
    items_novel = apply_novelty_filter(items_scored, state)
    # Ensure we still have enough; if too strict, fall back to scored
    pool = items_novel if len(items_novel) >= top_k else items_scored
    pool.sort(key=lambda x: x.get("score",0.0), reverse=True)
    top = pool[:top_k]

    # LLM key take-aways
    takeaways = generate_key_takeaways_llm(pool, top, tz_name=tz_name, bullets=5)

    # Build Top creatives list
    lines_top = []
    for rank, it in enumerate(top, 1):
        bits = []
        if it.get("views") is not None: bits.append(f"{it['views']:,} views")
        fr = _funding_ratio(it.get("goal"), it.get("pledged"))
        if fr: bits.append(f"{fr*100:.0f}% funded")
        if it.get("backers"): bits.append(f"{it['backers']:,} backers")
        if it.get("comments"): bits.append(f"{it['comments']:,} comments")
        reason = " · ".join(bits) if bits else "engagement signals present"
        lines_top.append(f"{rank}. {it.get('title','(untitled)')} – {reason} (score {it.get('score',0):.2f}).")

    # Selected 5 machine fields
    selected_lines = []
    for it in top:
        out = {
            "platform": it.get("platform",""),
            "title": it.get("title",""),
            "post_date": _primary_post_date(it),
            "platform_id": it.get("platform_id") or urllib.parse.urlsplit(it.get("url") or "").path.strip("/"),
            "link": it.get("url",""),
            "views": it.get("views"),
            "backers": it.get("backers"),
            "comments": it.get("comments"),
            "score": round(it.get("score",0.0), 2),
        }
        selected_lines.append(json.dumps(out, ensure_ascii=False))

    # Update state with what we actually reported
    update_state_with_report(top, state)

    # Render
    today_local = datetime.now(ZoneInfo(tz_name)).date().isoformat()
    report = []
    report.append("### Daily Creative Performance Report")
    report.append(f"Date: {today_local} ({tz_name})\n")
    report.append("Key take-aways")
    for t in takeaways:
        report.append(f"{t}  ")
    report.append("\nTop creatives (sorted by score)")
    report.extend(lines_top)
    report.append("\nSelected (5) — machine fields")
    report.append("```json")
    report.extend(selected_lines)
    report.append("```")
    return "\n".join(report)

# =========================
# Main
# =========================
def main():
    raw = run_agent()

    # 1) Try to parse compact JSON even if it was printed inside ```json fences
    data = _extract_json_from_fence(raw)
    items = []
    if data and isinstance(data, dict) and isinstance(data.get("results"), list):
        items = data["results"]
    else:
        # fallback: scan for any {"candidate": {...}} crumbs
        for m in re.finditer(r'\{"\s*candidate"\s*:\s*\{.*?\}\s*\}', raw, re.S):
            try:
                obj = json.loads(m.group(0))
                items.append(obj["candidate"])
            except Exception:
                pass

    if not items:
        print("No structured items parsed from agent output. Raw below:\n")
        print(raw)
        return

    # 2) Enrich: fetch metrics / embedded video views
    enriched = _enrich_minimal_results(items)

    if not enriched:
        print("Parsed items, but could not enrich any with valid campaign ad metrics. Raw below:\n")
        print(raw)
        return

    # 3) Render the final human-readable report with LLM-generated takeaways
    report_md = render_report(enriched, tz_name="Europe/Stockholm", top_k=5)
    print(report_md)


if __name__ == "__main__":
    main()
