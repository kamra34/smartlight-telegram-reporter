# ads_tracker.py
# Run:
#   python ads_tracker.py discover --top 5
#   python ads_tracker.py notify
#   python ads_tracker.py enrich-state
#
# Env:
#   GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_CUSTOM_SEARCH_CX
#   OPENAI_API_KEY (optional, for LLM extraction; set USE_LLM=true)
#   TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (for notify)
#   TZ (default Europe/Stockholm)

import os, re, json, time, math, random, urllib.parse, requests
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Optional libs
try:
    from googleapiclient.discovery import build
except Exception:
    build = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import cloudscraper

# =========================
# Config
# =========================
load_dotenv()

@dataclass(frozen=True)
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    google_api_key: str = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
    google_cx: str      = os.getenv("GOOGLE_CUSTOM_SEARCH_CX", "")
    timezone: str       = os.getenv("TZ", "Europe/Stockholm")
    state_file: str     = os.getenv("ADS_STATE_FILE", "ads_state.json")
    cooldown_days: int  = int(os.getenv("COOLDOWN_DAYS", "7"))
    improvement_pct: float = float(os.getenv("IMPROVEMENT_PCT", "0.15"))
    http_timeout: int   = int(os.getenv("HTTP_TIMEOUT", "18"))
    http_retries: int   = int(os.getenv("HTTP_RETRIES", "2"))
    use_llm: bool       = os.getenv("USE_LLM", "true").lower() == "true"
    notify_min: int  = int(os.getenv("NOTIFY_MIN", "3"))
    notify_max: int  = int(os.getenv("NOTIFY_MAX", "5"))
    window_days: int = int(os.getenv("WINDOW_DAYS", "150"))

CFG = Config()
CLIENT = OpenAI(api_key=CFG.openai_api_key) if (CFG.use_llm and OpenAI and CFG.openai_api_key) else None

# =========================
# HTTP helpers (Cloudflare-safe)
# =========================
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

def http_get(url: str, timeout=CFG.http_timeout, retries=CFG.http_retries) -> requests.Response:
    last_exc = None
    for i in range(retries + 1):
        try:
            r = _scraper.get(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            if r.status_code == 403:
                r = requests.get(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1))
    raise last_exc

def http_head(url: str, timeout=CFG.http_timeout, retries=CFG.http_retries) -> requests.Response:
    last_exc = None
    for i in range(retries + 1):
        try:
            r = _scraper.head(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            if r.status_code >= 400:
                r = _scraper.get(url, headers=_headers(), timeout=timeout, stream=True, allow_redirects=True)
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1))
    raise last_exc

def verify_exists(url) -> Dict[str, Any]:
    try:
        r = http_head(url)
        ok = r.status_code < 400
        return {"ok": ok, "final_url": getattr(r, "url", url), "content_type": r.headers.get("Content-Type","")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_html(url: str) -> Tuple[str,str]:
    r = http_get(url)
    return r.text, getattr(r, "url", url)

def normalize_url(url: str):
    try:
        u = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, "", ""))
    except Exception:
        return url

def canonicalize_campaign_url(url: str) -> str:
    """Normalize KS/IGG URLs to main campaign page, stripping /rewards, /posts, /community, etc."""
    try:
        u = urllib.parse.urlsplit(url)
        host = u.netloc.lower()
        path = u.path or "/"
        parts = [p for p in path.split("/") if p]
        if "kickstarter.com" in host:
            if parts and parts[0] == "projects":
                parts = parts[:3]  # /projects/{owner}/{slug}
                path = "/" + "/".join(parts)
        elif "indiegogo.com" in host:
            if parts and parts[0] == "projects":
                parts = parts[:2]  # /projects/{slug}
                path = "/" + "/".join(parts)
        return urllib.parse.urlunsplit((u.scheme, host, path, "", ""))
    except Exception:
        return url

def is_indiegogo_project_url(url: str) -> bool:
    try:
        u = urllib.parse.urlsplit(url)
        host = u.netloc.lower()
        parts = [p for p in (u.path or "/").split("/") if p]
        return ("indiegogo.com" in host) and (len(parts) >= 2) and (parts[0] == "projects")
    except Exception:
        return False

# =========================
# Google Custom Search
# =========================
DOMAINS = ["kickstarter.com", "indiegogo.com", "youtube.com", "vimeo.com"]
QUERY_SUFFIXES = [
    '(launch OR "now live" OR trailer) (lighting OR "smart home" OR gadget OR electronics)',
    '(Kickstarter OR Indiegogo) (trailer OR launch) (design OR lighting OR home)',
    '(crowdfunding campaign) (video OR trailer) (lighting OR smart home OR gadget)',
]

def rotated(seq):
    if not seq: return seq
    k = datetime.now().timetuple().tm_yday % len(seq)
    return seq[k:] + seq[:k]

def google_search_site(domain: str, query_suffix: str, num: int = 5, date_restrict: str = "d7", page: int = 1):
    if not build:
        raise RuntimeError("google-api-python-client not available")
    if not (CFG.google_api_key and CFG.google_cx):
        raise RuntimeError("Missing Google CSE creds")
    svc = build("customsearch", "v1", developerKey=CFG.google_api_key)
    q = f"site:{domain} {query_suffix}".strip()
    start = (page - 1) * 10 + 1
    try:
        resp = svc.cse().list(q=q, cx=CFG.google_cx, num=min(num,10), dateRestrict=date_restrict, start=start).execute()
    except Exception:
        resp = svc.cse().list(q=q, cx=CFG.google_cx, num=min(num,10), dateRestrict="d14", start=start).execute()
    items = resp.get("items", []) or []
    return [{"title": it.get("title"), "link": it.get("link"), "displayLink": it.get("displayLink")} for it in items]

# =========================
# Parsers (YT/Vimeo)
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

def parse_youtube(url: str) -> Dict[str, Any]:
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
    is_camp = is_campaigny_text((title or "") + " " + (desc or ""))

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

def parse_vimeo(url: str) -> Dict[str, Any]:
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
    is_ad_like = (dur_s is not None and dur_s <= 180)
    return {
        "platform":"vimeo","title": title,"url": final,
        "views": views,"published_at": None,"duration_s": dur_s,
        "is_campaign_ad": bool(is_ad_like and is_campaigny_text((title or "") + " " + (desc or ""))),
        "goal": None,"pledged": None,"backers": None,
        "comments": None,"updates": None,"status": None,"category": None,
        "platform_id": urllib.parse.urlsplit(final).path.strip("/"),
        "campaign_dates": {"launched": None, "ended": None}
    }

# =========================
# Kickstarter/Indiegogo + helpers
# =========================
YTB_IFR = re.compile(r'src="https?://(?:www\.)?youtube\.com/embed/([A-Za-z0-9_\-]+)"')
VIM_IFR = re.compile(r'src="https?://player\.vimeo\.com/video/([0-9]+)"')

def find_embedded_videos(html: str) -> List[Tuple[str,str]]:
    vids = []
    for m in YTB_IFR.finditer(html):
        vids.append(("youtube", f"https://www.youtube.com/watch?v={m.group(1)}"))
    for m in VIM_IFR.finditer(html):
        vids.append(("vimeo", f"https://vimeo.com/{m.group(1)}"))
    return vids

def ts_to_date(s: str) -> Optional[str]:
    try: return datetime.fromtimestamp(int(s), tz=timezone.utc).date().isoformat()
    except: return None

# ---------- LLM extraction (URL + HTML) ----------
def llm_extract_campaign(url: str, html: str) -> Dict[str, Any]:
    """
    Give the LLM the URL + HTML to extract fields:
      pledged, goal, backers, comments, views (if on page),
      video_urls (list), llm_score (0..1), confidence (0..1)
    """
    if not (CFG.use_llm and CLIENT):
        return {}
    system = (
        "You are an information extraction agent for crowdfunding campaign pages "
        "(Kickstarter or Indiegogo). Extract numeric metrics if visible on the page HTML."
    )
    user = (
        "Given the campaign URL and HTML below, return ONLY a compact JSON object with keys:\n"
        " pledged:number?, goal:number?, backers:int?, comments:int?, views:int?,\n"
        " video_urls:string[]?, llm_score:number?[0..1], confidence:number?[0..1].\n"
        "Rules:\n"
        "- Use NUMBERS only (no currency symbols, no text).\n"
        "- If a key is not present on the page, omit it.\n"
        f"URL: {url}\n\nHTML START\n{html[:150_000]}\nHTML END"
    )
    try:
        resp = CLIENT.chat.completions.create(
            model=CFG.openai_model, temperature=0.0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else json.loads(raw)
        out: Dict[str, Any] = {}
        def to_int(x):
            try: return int(re.sub(r"[^\d]", "", str(x)))
            except: return None
        def to_float(x):
            try: return float(re.sub(r"[^\d\.]", "", str(x)))
            except: return None
        if "pledged" in data:  out["pledged"]  = to_float(data["pledged"])
        if "goal" in data:     out["goal"]     = to_float(data["goal"])
        if "backers" in data:  out["backers"]  = to_int(data["backers"])
        if "comments" in data: out["comments"] = to_int(data["comments"])
        if "views" in data:    out["views"]    = to_int(data["views"])
        if "llm_score" in data:
            try:
                s = float(data["llm_score"])
                out["llm_score"] = max(0.0, min(1.0, s))
            except:
                pass
        if "confidence" in data:
            try:
                c = float(data["confidence"])
                out["confidence"] = max(0.0, min(1.0, c))
            except:
                pass
        vids = data.get("video_urls") or []
        out["video_urls"] = [v for v in vids if isinstance(v, str)]
        return out
    except Exception:
        return {}

# ---------- Kickstarter ----------
def parse_kickstarter(url: str) -> Dict[str, Any]:
    def _parse(html: str, final_url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.text.strip() if soup.title else final_url)
        title = title.replace(" — Kickstarter","").replace("| Kickstarter","").strip()
        text = html

        def first_int(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    try: return int(re.sub(r"[^\d]", "", m.group(1)))
                    except: pass
            return None

        def first_float(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    try: return float(re.sub(r"[^\d\.]", "", m.group(1)))
                    except: pass
            return None

        def first_str(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m: return m.group(1)
            return None

        pledged = first_float(r'"usd_pledged"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"pledged"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        goal    = first_float(r'"goal"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        backers = first_int(r'"backers_count"\s*:\s*([0-9]+)',
                            r'"backersCount"\s*:\s*([0-9]+)',
                            r'"backers"\s*:\s*([0-9]+)')
        comments= first_int(r'"comments_count"\s*:\s*([0-9]+)',
                            r'"commentsCount"\s*:\s*([0-9]+)')
        updates = first_int(r'"updates_count"\s*:\s*([0-9]+)',
                            r'"updatesCount"\s*:\s*([0-9]+)')
        state   = first_str(r'"state"\s*:\s*"([a-z_]+)"')
        category= first_str(r'"category"\s*:\s*{[^}]*"name"\s*:\s*"([^"]+)"',
                            r'"category"\s*:\s*"([^"]+)"')

        launched_ts = first_int(r'"launched_at"\s*:\s*([0-9]{9,11})')
        deadline_ts = first_int(r'"deadline"\s*:\s*([0-9]{9,11})')

        path = urllib.parse.urlsplit(final_url).path.strip("/")
        platform_id = path.replace("projects/", "", 1) if path.startswith("projects/") else path

        embeds = find_embedded_videos(html)

        parsed = {
            "platform":"kickstarter","title": title,"url": final_url,
            "views": None,"published_at": None,
            "goal": goal,"pledged": pledged,"backers": backers,
            "comments": comments,"updates": updates,"status": state,
            "category": category,"platform_id": platform_id,
            "campaign_dates": {"launched": ts_to_date(launched_ts) if launched_ts else None,
                               "ended": ts_to_date(deadline_ts) if deadline_ts else None},
            "embedded_videos": embeds
        }

        # LLM fallback if still empty
        if parsed["pledged"] is None and parsed["backers"] is None and parsed["comments"] is None:
            extra = llm_extract_campaign(final_url, html)
            for k in ("pledged","goal","backers","comments"):
                if parsed.get(k) is None and extra.get(k) is not None:
                    parsed[k] = extra[k]
            # merge extra videos
            for v in extra.get("video_urls") or []:
                if "youtube.com" in v or "vimeo.com" in v:
                    host = "youtube" if "youtube.com" in v else "vimeo"
                    if (host, v) not in parsed["embedded_videos"]:
                        parsed["embedded_videos"].append((host, v))
            if parsed.get("views") is None and extra.get("views") is not None:
                parsed["views"] = extra["views"]

        return parsed

    canon = canonicalize_campaign_url(url)
    try:
        html, final = fetch_html(canon)
        parsed = _parse(html, final)
        if (parsed.get("pledged") is None and parsed.get("backers") is None and parsed.get("comments") is None
            and canon != url):
            html2, final2 = fetch_html(url)
            parsed2 = _parse(html2, final2)
            def sig(p): return sum(x is not None for x in (p.get("pledged"), p.get("backers"), p.get("comments")))
            return parsed2 if sig(parsed2) > sig(parsed) else parsed
        return parsed
    except Exception:
        return {"platform":"kickstarter","title":"Kickstarter Campaign","url": canon,
                "views":None,"published_at":None,"goal":None,"pledged":None,"backers":None,
                "comments":None,"updates":None,"status":None,"category":None,
                "platform_id": urllib.parse.urlsplit(canon).path.strip("/"),
                "campaign_dates":{"launched":None,"ended":None}}

# ---------- Indiegogo ----------
def parse_indiegogo(url: str) -> Dict[str, Any]:
    def _parse(html: str, final_url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.text.strip() if soup.title else final_url)
        text = html

        # __NUXT__ JSON (when present)
        nuxt_json = None
        m_nuxt = re.search(r"window\.__NUXT__\s*=\s*(\{.*?\});", text, re.S)
        if m_nuxt:
            try: nuxt_json = json.loads(m_nuxt.group(1))
            except Exception: nuxt_json = None

        def first_int(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S | re.I)
                if m:
                    try: return int(re.sub(r"[^\d]", "", m.group(1)))
                    except: pass
            return None

        def first_float(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S | re.I)
                if m:
                    try: return float(re.sub(r"[^\d\.]", "", m.group(1)))
                    except: pass
            return None

        def first_str(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S | re.I)
                if m: return m.group(1)
            return None

        goal    = first_float(r'"goal"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"goalAmount"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        pledged = first_float(r'"collected_amount"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"funds_raised"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"collectedAmount"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        backers = first_int(r'"contributors_count"\s*:\s*([0-9]+)',
                            r'"contributorsCount"\s*:\s*([0-9]+)',
                            r'"backers"\s*:\s*([0-9]+)')
        comments= first_int(r'"comments_count"\s*:\s*([0-9]+)',
                            r'"commentsCount"\s*:\s*([0-9]+)')
        updates = first_int(r'"updates_count"\s*:\s*([0-9]+)',
                            r'"updatesCount"\s*:\s*([0-9]+)')
        status  = first_str(r'"status"\s*:\s*"([a-z_]+)"')
        category= first_str(r'"category"\s*:\s*{[^}]*"name"\s*:\s*"([^"]+)"',
                            r'"category"\s*:\s*"([^"]+)"')

        # visible markup fallback
        if backers is None:
            m = re.search(r'Contributors[^0-9]*([\d,\.]+)', soup.get_text(" ", strip=True), re.I)
            if m:
                try: backers = int(re.sub(r"[^\d]", "", m.group(1)))
                except: pass
        if comments is None:
            m = re.search(r'Comments[^0-9]*([\d,\.]+)', soup.get_text(" ", strip=True), re.I)
            if m:
                try: comments = int(re.sub(r"[^\d]", "", m.group(1)))
                except: pass

        # __NUXT__ JSON dig
        if nuxt_json:
            try:
                data = json.dumps(nuxt_json)
                if pledged is None:
                    m = re.search(r'"collectedAmount"\s*:\s*([0-9]+(?:\.[0-9]+)?)', data)
                    if m: pledged = float(m.group(1))
                if goal is None:
                    m = re.search(r'"goalAmount"\s*:\s*([0-9]+(?:\.[0-9]+)?)', data)
                    if m: goal = float(m.group(1))
                if backers is None:
                    m = re.search(r'"contributorsCount"\s*:\s*([0-9]+)', data)
                    if m: backers = int(m.group(1))
                if comments is None:
                    m = re.search(r'"commentsCount"\s*:\s*([0-9]+)', data)
                    if m: comments = int(m.group(1))
            except Exception:
                pass

        embeds = find_embedded_videos(html)

        parsed = {
            "platform":"indiegogo","title": title,"url": final_url,
            "views": None,"published_at": None,
            "goal": goal,"pledged": pledged,"backers": backers,
            "comments": comments,"updates": updates,"status": status,
            "category": category,"platform_id": urllib.parse.urlsplit(final_url).path.strip("/"),
            "campaign_dates": {"launched": None,"ended": None},
            "embedded_videos": embeds,
            "llm_score": None
        }

        # LLM URL+HTML to fill gaps & add llm_score/views/video_urls
        extra = llm_extract_campaign(final_url, html)
        for k in ("pledged","goal","backers","comments","views","llm_score"):
            if extra.get(k) is not None and (parsed.get(k) in (None, 0)):
                parsed[k] = extra[k]
        for v in extra.get("video_urls") or []:
            if "youtube.com" in v or "vimeo.com" in v:
                host = "youtube" if "youtube.com" in v else "vimeo"
                if (host, v) not in parsed["embedded_videos"]:
                    parsed["embedded_videos"].append((host, v))

        return parsed

    canon = canonicalize_campaign_url(url)
    try:
        html, final = fetch_html(canon)
        parsed = _parse(html, final)
        if (parsed.get("pledged") is None and parsed.get("backers") is None and parsed.get("comments") is None
            and canon != url):
            html2, final2 = fetch_html(url)
            parsed2 = _parse(html2, final2)
            def sig(p): return sum(x is not None for x in (p.get("pledged"), p.get("backers"), p.get("comments")))
            return parsed2 if sig(parsed2) > sig(parsed) else parsed
        return parsed
    except Exception:
        return {"platform":"indiegogo","title":"Indiegogo Campaign","url": canon,
                "views":None,"published_at":None,"goal":None,"pledged":None,"backers":None,
                "comments":None,"updates":None,"status":None,"category":None,
                "platform_id": urllib.parse.urlsplit(canon).path.strip("/"),
                "campaign_dates":{"launched":None,"ended":None}}

# =========================
# Enrichment (resolve embeds → best YT/Vimeo views)
# =========================
def resolve_embeds_to_views(item: Dict[str, Any]) -> Dict[str, Any]:
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
        # keep campaign URL; store ad video separately
        item["ad_url"] = best["url"]
        item["ad_platform"] = best["platform"]
        item["views"] = best["views"]
        item["published_at"] = best.get("published_at")
        item["duration_s"] = best.get("duration_s")
        item["is_campaign_ad"] = True
    return item


def enrich_minimal_results(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    enriched = []
    for it in items:
        url = it.get("url") or ""
        url = canonicalize_campaign_url(normalize_url(url))
        try:
            if not verify_exists(url).get("ok"):
                continue
            u = url.lower()
            if "kickstarter.com" in u:
                parsed = resolve_embeds_to_views(parse_kickstarter(url))
            elif "indiegogo.com" in u:
                if not is_indiegogo_project_url(url):
                    continue
                parsed = resolve_embeds_to_views(parse_indiegogo(url))
                if (parsed.get("pledged") is None and parsed.get("backers") is None and parsed.get("comments") is None
                    and not parsed.get("embedded_videos")):
                    continue
            elif "youtube.com" in u:
                parsed = parse_youtube(url)
                if not parsed.get("is_campaign_ad"):
                    continue
            elif "vimeo.com" in u:
                parsed = parse_vimeo(url)
                if not parsed.get("is_campaign_ad"):
                    continue
            else:
                continue

            if it.get("title") and (not parsed.get("title") or len(it["title"]) < 120):
                parsed["title"] = it["title"]
            if it.get("platform"):
                parsed["platform"] = it["platform"]

            enriched.append(parsed)
        except Exception:
            continue
    return enriched

# =========================
# Gate & scoring
# =========================
def _primary_date_for_window(it: Dict[str, Any]) -> Optional[str]:
    return (it.get("published_at")
            or ((it.get("campaign_dates") or {}).get("launched"))
            or ((it.get("campaign_dates") or {}).get("ended")))

def is_within_window(it: Dict[str, Any], days: int = CFG.window_days) -> bool:
    ds = _primary_date_for_window(it)
    if not ds:
        # if unknown, keep it — Google CSE already used recent restrict,
        # and we’ll still dedupe by key
        return True
    try:
        d = datetime.fromisoformat(ds).date()
    except Exception:
        return True
    today = datetime.now(timezone.utc).date()
    return (today - d).days <= max(1, days)

def hard_gate(item: Dict[str,Any]) -> bool:
    host = (item.get("platform") or "").lower()
    if host not in ("youtube","vimeo","kickstarter","indiegogo"): return False
    if host in ("youtube","vimeo"):
        d = item.get("duration_s")
        if not d or d > 180: return False
        if not is_campaigny_text((item.get("title") or "") + " " + (item.get("desc") or "")):
            return False
    return True

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

def compute_scores(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
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
        det = max(0.0, s)  # clamp at 0
        if it.get("llm_score") is not None:
            it["score"] = float(it["llm_score"])
        else:
            it["score"] = round(max(0.0, min(1.0, 1 / (1 + math.exp(-det)))), 4)
    return items

# =========================
# State
# =========================
def pick_unsent_items(state: dict, max_items: int = 5) -> list[tuple[str, dict]]:
    """
    Returns up to max_items unsent items, ordered by:
      1) score desc
      2) best_views desc
      3) last_enriched desc (string ISO date; fine for ordering)
    """
    items = []
    for key, entry in state.get("items", {}).items():
        if entry.get("sent"):
            continue
        items.append((key, entry))
    items.sort(key=lambda kv: (
        float(kv[1].get("score") or 0.0),
        kv[1].get("best_views") or 0,
        kv[1].get("last_enriched") or ""
    ), reverse=True)
    return items[:max_items]

def extract_native_id(url: str) -> Tuple[str,str]:
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

def stable_key(item: Dict[str, Any]) -> str:
    plat = (item.get("platform") or "").lower()
    pid  = (item.get("platform_id") or "").strip()
    url  = item.get("url") or ""
    if plat in ("kickstarter", "indiegogo") and pid:
        return f"{plat}:{pid}"
    plat2, nid = extract_native_id(url)
    return f"{plat2}:{nid}"


def load_state() -> Dict[str,Any]:
    if os.path.exists(CFG.state_file):
        with open(CFG.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"items": {}}
    for k, v in data.get("items", {}).items():
        if "sent" not in v:
            v["sent"] = False
    return data

def save_state(state: Dict[str,Any]) -> None:
    with open(CFG.state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def apply_novelty_filter(items: List[Dict[str,Any]], state: Dict[str,Any]) -> List[Dict[str,Any]]:
    out = []
    for it in items:
        key = stable_key(it)
        prev = state["items"].get(key)
        if not prev:
            out.append(it); continue
        last_rep = prev.get("last_reported")
        best_views = prev.get("best_views") or 0
        cooldown_ok = True
        if last_rep:
            try:
                days = (datetime.now(timezone.utc).date() - datetime.fromisoformat(last_rep).date()).days
                cooldown_ok = days >= CFG.cooldown_days
            except:
                pass
        improved = (it.get("views") or 0) > best_views * (1 + CFG.improvement_pct)
        if cooldown_ok or improved:
            out.append(it)
    return out

# ==== NEW: persistence helpers ====
PERSIST_FIELDS = [
    "title","platform","url","platform_id",
    "ad_url","ad_platform",
    "views","backers","comments","goal","pledged",
    "published_at","duration_s","category","status",
    "campaign_dates","llm_score","score"
]

def _funding_ratio_value(goal, pledged):
    try:
        g = float(goal) if goal not in (None,"") else None
        p = float(pledged) if pledged not in (None,"") else None
        if not g or g <= 0 or p is None: return None
        return p / g
    except Exception:
        return None

def _snapshot_item(it: dict) -> dict:
    snap = {k: it.get(k) for k in PERSIST_FIELDS}
    # numeric cleanup…
    for n in ("views","backers","comments"):
        v = snap.get(n)
        if isinstance(v, str):
            try: snap[n] = int(re.sub(r"[^\d]", "", v)) if v.strip() else None
            except: pass
    for n in ("goal","pledged","llm_score","score"):
        v = snap.get(n)
        if isinstance(v, str):
            try: snap[n] = float(re.sub(r"[^\d\.]", "", v)) if v.strip() else None
            except: pass
    # funding ratio (as float, same style as your example)
    g, p = snap.get("goal"), snap.get("pledged")
    if g not in (None, 0) and p is not None:
        try:
            snap["funding_ratio"] = (float(p) / float(g)) * 100.0  # percent like your example (162.63)
        except Exception:
            snap["funding_ratio"] = None
    else:
        snap["funding_ratio"] = None
    # prefer llm_score if present
    if snap.get("llm_score") is not None:
        snap["score"] = float(snap["llm_score"])
    return snap

def persist_discover_results(enriched_items: list[dict], state: dict) -> None:
    """Merge snapshots into state['items'] only. No 'daily' key."""
    today = datetime.now(timezone.utc).date().isoformat()
    state.setdefault("items", {})

    for it in enriched_items:
        key = stable_key(it)
        entry = state["items"].get(key, {})
        snap = _snapshot_item(it)
        # merge w/o clobbering non-empty values
        for k, v in snap.items():
            if v is not None:
                entry[k] = v
            else:
                entry.setdefault(k, v)
        entry.setdefault("sent", False)
        entry["last_enriched"] = today
        entry["best_views"] = max(entry.get("best_views") or 0, it.get("views") or 0)
        state["items"][key] = entry

    save_state(state)

# =========================
# Agent (thin planner)
# =========================
def agent_discover(max_total=20) -> Dict[str,Any]:
    results = []
    domains = rotated(DOMAINS)
    queries = rotated(QUERY_SUFFIXES)
    page = (datetime.now().timetuple().tm_yday % 2) + 1

    for domain in domains:
        q_suffix = queries[0]; queries = queries[1:] + queries[:1]
        try:
            candidates = google_search_site(domain, q_suffix, num=6, page=page)
        except Exception:
            continue

        for it in candidates:
            url = it.get("link")
            if not url: continue
            nurl = canonicalize_campaign_url(normalize_url(url))
            v = verify_exists(nurl)
            if not v.get("ok"): continue

            try:
                lower = nurl.lower()
                if "youtube.com" in lower:   parsed = parse_youtube(nurl)
                elif "vimeo.com" in lower:   parsed = parse_vimeo(nurl)
                elif "kickstarter.com" in lower: parsed = parse_kickstarter(nurl)
                elif "indiegogo.com" in lower:
                    if not is_indiegogo_project_url(nurl):
                        continue
                    parsed = parse_indiegogo(nurl)
                else: continue
            except Exception:
                continue

            if it.get("title") and (not parsed.get("title") or len(it["title"]) < 120):
                parsed["title"] = it["title"]

            if hard_gate(parsed):
                results.append({
                    "title": parsed.get("title"),
                    "url": parsed.get("url"),
                    "platform": parsed.get("platform"),
                    "views": parsed.get("views"),
                    "goal": parsed.get("goal"),
                    "pledged": parsed.get("pledged"),
                    "backers": parsed.get("backers"),
                    "comments": parsed.get("comments"),
                    "published_at": parsed.get("published_at"),
                    "campaign_dates": parsed.get("campaign_dates"),
                    "category": parsed.get("category"),
                    "status": parsed.get("status"),
                    "platform_id": parsed.get("platform_id"),
                    "duration_s": parsed.get("duration_s"),
                    "llm_score": parsed.get("llm_score"),
                })

            if len(results) >= max_total: break
        if len(results) >= max_total: break

    seen = set(); dedup = []
    for r in results:
        u = r.get("url")
        if u in seen: continue
        seen.add(u); dedup.append(r)
    return {"results": dedup[:max_total]}

# =========================
# Reporting FROM STATE
# =========================
def _primary_post_date(entry: dict):
    return entry.get("published_at") or ((entry.get("campaign_dates") or {}).get("launched")) or ((entry.get("campaign_dates") or {}).get("ended"))

def _format_top_line(rank: int, entry: dict) -> str:
    bits = []
    if entry.get("backers") is not None: bits.append(f"{entry['backers']:,} backers")
    if entry.get("comments") is not None: bits.append(f"{entry['comments']:,} comments")
    fr = entry.get("funding_ratio")
    if fr: bits.append(f"{fr*100:.0f}% funded")
    if entry.get("views") is not None: bits.append(f"{entry['views']:,} views")
    reason = " · ".join(bits) if bits else "engagement signals present"
    return f"{rank}. {entry.get('title','(untitled)')} – {reason} (score {float(entry.get('score',0.0)):.2f})."

def build_daily_report_from_state(state: dict, date_str: Optional[str] = None, top_k: int = 5) -> str:
    tz_name = CFG.timezone
    if not date_str:
        date_str = datetime.now(ZoneInfo(tz_name)).date().isoformat()

    items = state.get("items", {})
    daily = state.get("daily", {})
    keys_order = daily.get(date_str, [])

    if not keys_order:
        todays = [ (k,v) for k,v in items.items() if v.get("last_enriched") == date_str ]
        todays.sort(key=lambda kv: float(kv[1].get("score") or 0.0), reverse=True)
        keys_order = [k for k,_ in todays[:top_k]]

    ordered_entries = [items[k] for k in keys_order if k in items][:top_k]

    lines_top = []
    for idx, entry in enumerate(ordered_entries, 1):
        lines_top.append(_format_top_line(idx, entry))

    selected_lines = []
    for entry in ordered_entries:
        out = {
            "platform": entry.get("platform",""),
            "title": entry.get("title",""),
            "post_date": _primary_post_date(entry),
            "platform_id": entry.get("platform_id") or urllib.parse.urlsplit(entry.get("url") or "").path.strip("/"),
            "link": entry.get("url",""),
            "views": entry.get("views"),
            "backers": entry.get("backers"),
            "comments": entry.get("comments"),
            "score": round(float(entry.get("score",0.0)), 2),
        }
        selected_lines.append(json.dumps(out, ensure_ascii=False))

    report = []
    report.append("Top creatives (sorted by score)")
    report.extend(lines_top)
    report.append("\nSelected (5) — machine fields")
    report.append("```json")
    report.extend(selected_lines)
    report.append("```")
    return "\n".join(report)

# =========================
# Pipeline
# =========================
def discover(top_k=5, add_new_only: bool = True, update_existing: bool = False):
    """Discover and persist; by default only adds *new* campaigns within window."""
    data = agent_discover(max_total=20)
    raw_items = data.get("results", [])
    if not raw_items:
        print("No candidates found."); return

    enriched = enrich_minimal_results(raw_items)
    if not enriched:
        print("Candidates found, but none enriched to valid campaign ads."); return

    # keep only window
    enriched = [it for it in enriched if is_within_window(it, days=CFG.window_days)]
    if not enriched:
        print("No items within discovery window."); return

    # score (kept; sets score=llm_score when present)
    items_scored = compute_scores(enriched)

    state = load_state()

    # filter to *new* keys unless we want to refresh
    if add_new_only and not update_existing:
        items_scored = [it for it in items_scored if stable_key(it) not in state.get("items", {})]

    if not items_scored:
        print("No new items to add (all were already in state).")
        return

    # optionally cap how many we persist this run
    items_scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    to_persist = items_scored[:top_k]

    persist_discover_results(to_persist, state)
    print(f"Persisted {len(to_persist)} new items.")


def send_to_telegram(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": False})
    r.raise_for_status()

def notify():
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat  = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID missing")

    # Load and pick unsent items
    state = load_state()
    candidates = pick_unsent_items(state, max_items=CFG.notify_max)

    # If we have too few, run discover to refill, then try again
    if len(candidates) < CFG.notify_min:
        # Run a small discover pass to populate state
        try:
            discover(top_k=max(CFG.notify_max, 5))
        except Exception as e:
            print(f"Discover failed during notify fallback: {e}")
        state = load_state()
        candidates = pick_unsent_items(state, max_items=CFG.notify_max)

    if not candidates:
        print("No unsent items available in state.")
        return

    # Compose and send per-item messages
    sent_any = False
    for key, entry in candidates:
        # Build a concise message
        title = entry.get("title") or "(untitled)"
        platform = entry.get("platform") or ""
        url = entry.get("url") or ""
        backers = entry.get("backers")
        comments = entry.get("comments")
        views = entry.get("views")
        fr = entry.get("funding_ratio")
        score = entry.get("score")

        bits = []
        if backers is not None:  bits.append(f"{int(backers):,} backers")
        if comments is not None: bits.append(f"{int(comments):,} comments")
        if views is not None:    bits.append(f"{int(views):,} views")
        if fr:                   bits.append(f"{float(fr)*100:.0f}% funded")
        if score is not None:    bits.append(f"score {float(score):.2f}")
        metrics_line = " · ".join(bits) if bits else "engagement signals present"

        text = (
            f"{title}\n"
            f"Platform: {platform}\n"
            f"Link: {url}\n"
            f"{metrics_line}"
        )

        try:
            send_to_telegram(token, chat, text)
            # Mark sent and persist
            entry["sent"] = True
            state["items"][key] = entry
            sent_any = True
        except Exception as e:
            print(f"Telegram send failed for {key}: {e}")

    if sent_any:
        save_state(state)
        print(f"Delivered {len([1 for _,e in candidates if e.get('sent')])} items and marked sent=True.")
    else:
        print("No items were delivered.")


# ---------- Enrich existing items in state ----------
def _enrich_one_entry(entry: dict) -> dict:
    url = entry.get("url") or ""
    if not url:
        return entry
    url = canonicalize_campaign_url(normalize_url(url))
    try:
        verify = verify_exists(url)
        if not verify.get("ok"):
            return entry
        lower = url.lower()
        if "kickstarter.com" in lower:
            parsed = resolve_embeds_to_views(parse_kickstarter(url))
        elif "indiegogo.com" in lower:
            if not is_indiegogo_project_url(url):
                return entry
            parsed = resolve_embeds_to_views(parse_indiegogo(url))
        elif "youtube.com" in lower:
            parsed = parse_youtube(url)
        elif "vimeo.com" in lower:
            parsed = parse_vimeo(url)
        else:
            return entry
        snap = _snapshot_item(parsed)
        entry.update(snap)
        entry["best_views"] = max(entry.get("best_views") or 0, parsed.get("views") or 0)
        return entry
    except Exception:
        return entry

def enrich_state_items():
    state = load_state()
    items = state.get("items", {})
    if not items:
        print("State has no items to enrich."); return
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    for key, entry in list(items.items()):
        before = (entry.get("backers"), entry.get("comments"), entry.get("views"))
        entry = _enrich_one_entry(entry)
        after  = (entry.get("backers"), entry.get("comments"), entry.get("views"))
        entry["last_enriched"] = today
        items[key] = entry
        if before != after:
            count += 1
    save_state(state)
    print(f"Enriched {count} items and saved state.")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ads tracker: discover & notify")
    ap.add_argument("cmd", choices=["discover","notify","enrich-state"], help="Pipeline command to run")
    ap.add_argument("--top", type=int, default=5, help="Top-K items in report (discover)")
    args = ap.parse_args()
    if args.cmd == "discover":
        discover(top_k=args.top)
    elif args.cmd == "notify":
        notify()
    elif args.cmd == "enrich-state":
        enrich_state_items()
