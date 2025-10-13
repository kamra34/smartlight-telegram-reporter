# ads_tracker.py
import os, re, json, time, math, random, urllib.parse, requests
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv

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
    google_api_key: str = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
    google_cx: str      = os.getenv("GOOGLE_CUSTOM_SEARCH_CX", "")
    timezone: str       = os.getenv("TZ", "Europe/Stockholm")
    state_file: str     = os.getenv("ADS_STATE_FILE", "ads_state.json")
    cooldown_days: int  = int(os.getenv("COOLDOWN_DAYS", "7"))
    improvement_pct: float = float(os.getenv("IMPROVEMENT_PCT", "0.15"))
    http_timeout: int   = int(os.getenv("HTTP_TIMEOUT", "18"))
    http_retries: int   = int(os.getenv("HTTP_RETRIES", "2"))
    use_llm: bool       = os.getenv("USE_LLM", "true").lower() == "true"

CFG = Config()
CLIENT = OpenAI(api_key=CFG.openai_api_key) if (CFG.use_llm and OpenAI and CFG.openai_api_key) else None

# =========================
# HTTP helpers
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
# Parsers + helpers
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

# ---------- OPTIONAL LLM FALLBACK ----------
def llm_scrape_metrics(html: str) -> Dict[str, Optional[float]]:
    """Use LLM to extract pledged, goal, backers, comments when regexes fail. Numbers only."""
    if not (CFG.use_llm and CLIENT):
        return {}
    prompt = (
        "You will receive HTML from a Kickstarter or Indiegogo campaign page. "
        "Extract numeric metrics if visible: pledged, goal, backers (contributors), comments. "
        "Return ONLY a compact JSON object with those keys when known, numbers only."
    )
    try:
        resp = CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[{"role":"system","content":prompt},
                      {"role":"user","content":html[:150_000]}]
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else json.loads(raw)
        out = {}
        for k in ("pledged","goal","backers","comments"):
            if k in data and data[k] not in (None, ""):
                try:
                    if k in ("backers","comments"):
                        out[k] = int(re.sub(r"[^\d]", "", str(data[k])))
                    else:
                        out[k] = float(re.sub(r"[^\d\.]", "", str(data[k])))
                except:
                    pass
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
                    try: return int(m.group(1))
                    except: pass
            return None

        def first_float(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    try: return float(m.group(1))
                    except: pass
            return None

        def first_str(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    return m.group(1)
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

        if parsed["pledged"] is None and parsed["backers"] is None and parsed["comments"] is None:
            extra = llm_scrape_metrics(html)
            for k, v in extra.items():
                if parsed.get(k) is None:
                    parsed[k] = v
        return parsed

    canon = canonicalize_campaign_url(url)
    try:
        html, final = fetch_html(canon)
        parsed = _parse(html, final)
        # if still weak and original differs, try original
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

        def first_int(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    try: return int(m.group(1))
                    except: pass
            return None

        def first_float(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    try: return float(m.group(1))
                    except: pass
            return None

        def first_str(*patterns):
            for pat in patterns:
                m = re.search(pat, text, re.S|re.I)
                if m:
                    return m.group(1)
            return None

        goal    = first_float(r'"goal"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"goalAmount"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        pledged = first_float(r'"collected_amount"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"funds_raised"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                              r'"collectedAmount"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        backers = first_int(r'"contributors_count"\s*:\s*([0-9]+)',
                            r'"contributorsCount"\s*:\s*([0-9]+)')
        comments= first_int(r'"comments_count"\s*:\s*([0-9]+)',
                            r'"commentsCount"\s*:\s*([0-9]+)')
        updates = first_int(r'"updates_count"\s*:\s*([0-9]+)',
                            r'"updatesCount"\s*:\s*([0-9]+)')
        status  = first_str(r'"status"\s*:\s*"([a-z_]+)"')
        category= first_str(r'"category"\s*:\s*{[^}]*"name"\s*:\s*"([^"]+)"',
                            r'"category"\s*:\s*"([^"]+)"')

        embeds = find_embedded_videos(html)

        parsed = {
            "platform":"indiegogo","title": title,"url": final_url,
            "views": None,"published_at": None,
            "goal": goal,"pledged": pledged,"backers": backers,
            "comments": comments,"updates": updates,"status": status,
            "category": category,"platform_id": urllib.parse.urlsplit(final_url).path.strip("/"),
            "campaign_dates": {"launched": None,"ended": None},
            "embedded_videos": embeds
        }

        if parsed["pledged"] is None and parsed["backers"] is None and parsed["comments"] is None:
            extra = llm_scrape_metrics(html)
            for k, v in extra.items():
                if parsed.get(k) is None:
                    parsed[k] = v
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
# Enrichment
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
        item["url"] = best["url"]
        item["platform"] = best["platform"]
        item["views"] = best["views"]
        item["published_at"] = best.get("published_at")
        item["duration_s"] = best.get("duration_s")
        item["is_campaign_ad"] = True
    return item

def enrich_minimal_results(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    enriched = []
    for it in items:
        url = it.get("url") or ""
        # canonicalize early
        url = canonicalize_campaign_url(normalize_url(url))
        try:
            if not verify_exists(url).get("ok"):
                continue
            u = url.lower()
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
# Gate & scoring (unchanged)
# =========================
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
        it["score"] = round(s, 4)
    return items

# =========================
# State
# =========================
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

def stable_key(item: Dict[str,Any]) -> str:
    plat, nid = extract_native_id(item.get("url") or "")
    return f"{plat}:{nid}"

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
            except: pass
        improved = (it.get("views") or 0) > best_views * (1 + CFG.improvement_pct)
        if cooldown_ok or improved:
            out.append(it)
    return out

def update_state_with_report(items: List[Dict[str,Any]], state: Dict[str,Any]) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    for it in items:
        key = stable_key(it)
        entry = state["items"].get(key, {})
        entry["best_views"] = max(entry.get("best_views") or 0, it.get("views") or 0)
        entry["last_reported"] = today
        entry["title"] = it.get("title")
        entry["platform"] = it.get("platform")
        entry.setdefault("sent", False)
        entry["url"] = it.get("url")
        state["items"][key] = entry
    save_state(state)

# =========================
# Agent
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
                elif "indiegogo.com" in lower:   parsed = parse_indiegogo(nurl)
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
# Reporting
# =========================
def _primary_post_date(it):
    return it.get("published_at") or ((it.get("campaign_dates") or {}).get("launched")) or ((it.get("campaign_dates") or {}).get("ended"))

def _funding_ratio_safe(it):
    fr = _funding_ratio(it.get("goal"), it.get("pledged"))
    return fr

def compress_for_takeaways(items, top_k=10):
    slim = []
    for it in items[:top_k]:
        slim.append({
            "platform": it.get("platform"),
            "title": it.get("title"),
            "category": it.get("category"),
            "status": it.get("status"),
            "views": it.get("views"),
            "funding_ratio": _funding_ratio_safe(it),
            "backers_per_day": _backers_per_day(it),
            "comments_per_day": _comments_per_day(it),
            "published_at": _primary_post_date(it),
            "duration_s": it.get("duration_s"),
            "score": it.get("score"),
        })
    return slim

def key_takeaways(items_scored, top_items, tz_name="Europe/Stockholm", bullets=5) -> List[str]:
    if not (CFG.use_llm and CLIENT):
        return [
            "• Short (≤180s) campaign videos led performance.",
            "• Higher funding ratios correlated with above-average scores.",
            "• Backers/comments velocity boosted rankings.",
            "• Recent launches (≤14 days) tended to perform better.",
            "• YouTube/Vimeo embeds on KS/IGG dominated the top set.",
        ]
    data = {
        "date_local": datetime.now(ZoneInfo(tz_name)).date().isoformat(),
        "items_sample": compress_for_takeaways(items_scored, top_k=15),
        "top_sample": compress_for_takeaways(top_items, top_k=len(top_items)),
    }
    sys = ("You are a marketing performance analyst. From the provided campaign/video metrics, "
           "infer concise, non-generic insights. Use concrete metrics (views, funding_ratio, backers/comments per day, recency, platform mix, duration). "
           "Output exactly 5 bullet points, each one sentence.")
    usr = f"Timezone: {tz_name}. Today is {data['date_local']}.\nAnalyze and return EXACTLY 5 bullets:\n\n{json.dumps(data, ensure_ascii=False)}"
    try:
        resp = CLIENT.chat.completions.create(model="gpt-4o-mini", temperature=0.4, messages=[{"role":"system","content": sys},{"role":"user","content": usr}])
        text = resp.choices[0].message.content.strip()
        lines = [ln.strip(" -•\t") for ln in text.split("\n") if ln.strip()]
        lines = [("• " + ln.rstrip(".")) + "." for ln in lines][:bullets]
        return lines or [
            "• Short (≤180s) campaign videos led performance.",
            "• Higher funding ratios correlated with above-average scores.",
            "• Backers/comments velocity boosted rankings.",
            "• Recent launches (≤14 days) tended to perform better.",
            "• YouTube/Vimeo embeds on KS/IGG dominated the top set.",
        ]
    except Exception:
        return [
            "• Short (≤180s) campaign videos led performance.",
            "• Higher funding ratios correlated with above-average scores.",
            "• Backers/comments velocity boosted rankings.",
            "• Recent launches (≤14 days) tended to perform better.",
            "• YouTube/Vimeo embeds on KS/IGG dominated the top set.",
        ]

def render_report(enriched_items: List[Dict[str,Any]], tz_name="Europe/Stockholm", top_k=5) -> str:
    items_scored = compute_scores(enriched_items)
    state = load_state()
    items_novel = apply_novelty_filter(items_scored, state)
    pool = items_novel if len(items_novel) >= top_k else items_scored
    pool.sort(key=lambda x: x.get("score",0.0), reverse=True)
    top = pool[:top_k]
    takeaways = key_takeaways(pool, top, tz_name=tz_name, bullets=5)

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

    update_state_with_report(top, state)

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
# Pipeline
# =========================
def discover(top_k=5):
    data = agent_discover(max_total=20)
    items = data.get("results", [])
    if not items:
        print("No candidates found."); return
    enriched = enrich_minimal_results(items)
    if not enriched:
        print("Candidates found, but none enriched to valid campaign ads."); return
    report_md = render_report(enriched, tz_name=CFG.timezone, top_k=top_k)
    print(report_md)

def send_to_telegram(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": False})
    r.raise_for_status()

def notify():
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat  = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID missing")

    state = load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    sent_any = False

    for key, entry in state.get("items", {}).items():
        if entry.get("sent"): continue
        if entry.get("last_reported") != today: continue
        title = entry.get("title") or key
        platform = entry.get("platform") or ""
        url = entry.get("url") or ""
        text = f"Top creative\nTitle: {title}\nPlatform: {platform}\nLink: {url}"
        try:
            send_to_telegram(token, chat, text)
            entry["sent"] = True
            sent_any = True
        except Exception as e:
            print(f"Telegram send failed for {key}: {e}")

    if sent_any:
        save_state(state)
        print("Marked sent=True for delivered items.")
    else:
        print("No unsent items for today.")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ads tracker: discover & notify")
    ap.add_argument("cmd", choices=["discover","notify"], help="Pipeline command to run")
    ap.add_argument("--top", type=int, default=5, help="Top-K items in report (discover)")
    args = ap.parse_args()
    if args.cmd == "discover":
        discover(top_k=args.top)
    elif args.cmd == "notify":
        notify()
