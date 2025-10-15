# parsers.py
import re, json, urllib.parse
from typing import Dict, Any, List, Tuple, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from config import CFG, CLIENT
from net import fetch_html, canonicalize_campaign_url, normalize_url, is_indiegogo_project_url

# ----- helpers -----
def ts_to_date(s: str) -> Optional[str]:
    try: return datetime.fromtimestamp(int(s), tz=timezone.utc).date().isoformat()
    except: return None

def is_campaigny_text(txt: str) -> bool:
    if not txt: return False
    txtl = txt.lower()
    kws = ["kickstarter","indiegogo","crowdfund","crowdfunding","now live","back us","pre-order","preorder","launch on","campaign"]
    return any(k in txtl for k in kws)

YTB_IFR = re.compile(r'src="https?://(?:www\.)?youtube\.com/embed/([A-Za-z0-9_\-]+)"')
VIM_IFR = re.compile(r'src="https?://player\.vimeo\.com/video/([0-9]+)"')
def find_embedded_videos(html: str) -> List[Tuple[str,str]]:
    vids = []
    for m in YTB_IFR.finditer(html):
        vids.append(("youtube", f"https://www.youtube.com/watch?v={m.group(1)}"))
    for m in VIM_IFR.finditer(html):
        vids.append(("vimeo", f"https://vimeo.com/{m.group(1)}"))
    return vids

def _safe_unicode_escape(raw):
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except Exception:
        return raw.replace("\\", "")

# ----- YouTube/Vimeo -----
YTG_VIEWS_RE = re.compile(r'"viewCount"\s*:\s*"\s*([0-9,]+)\s*"|{"simpleText"\s*:\s*"([\d,\.]+)\s*views"', re.I)
YTG_PUB_RE   = re.compile(r'"publishDate":\s*"(\d{4}-\d{2}-\d{2})"')
YTG_LEN_RE   = re.compile(r'"lengthSeconds"\s*:\s*"(\d+)"')
DESC_RE      = re.compile(r'"shortDescription":\s*"([^"]+)"', re.S)

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

# ----- LLM extraction -----
def llm_extract_campaign(url: str, html: str) -> Dict[str, Any]:
    """Extract pledged/goal/backers/comments/views/video_urls/llm_score/confidence via LLM."""
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
            except: pass
        if "confidence" in data:
            try:
                c = float(data["confidence"])
                out["confidence"] = max(0.0, min(1.0, c))
            except: pass
        vids = data.get("video_urls") or []
        out["video_urls"] = [v for v in vids if isinstance(v, str)]
        return out
    except Exception:
        return {}

# ----- resolve embeds to best views -----
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
        item["ad_url"] = best["url"]
        item["ad_platform"] = best["platform"]
        item["views"] = best["views"]
        item["published_at"] = best.get("published_at")
        item["duration_s"] = best.get("duration_s")
        item["is_campaign_ad"] = True
    return item



def detect_platform(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc.lower()
    if "kickstarter.com" in host: return "kickstarter"
    if "indiegogo.com"  in host: return "indiegogo"
    if "youtube.com"    in host: return "youtube"
    if "vimeo.com"      in host: return "vimeo"
    return "web"

def extract_platform_id(platform: str, url: str) -> str:
    u = urllib.parse.urlsplit(url)
    path = (u.path or "/").strip("/")
    if platform == "youtube":
        q = urllib.parse.parse_qs(u.query)
        return q.get("v", [path])[-1] or path
    return path or url

def llm_only_extract_campaign(url: str, title_hint: str | None = None) -> Dict[str, Any]:
    """
    Fetch the page once, run LLM extraction only, and return a unified item.
    No regex parsing, no embed resolution.
    """
    canon = canonicalize_campaign_url(url)
    html, final = fetch_html(canon)
    extra = llm_extract_campaign(final, html)  # may include pledged/goal/backers/comments/views/llm_score/video_urls
    # print(f"LLM Extractions for {final}, {html}: {extra}")
    platform = detect_platform(final)
    platform_id = extract_platform_id(platform, final)

    # Build the lean item solely from LLM (+ minimal URL-derived fields)
    out = {
        "platform": platform,
        "title": title_hint or "(untitled)",
        "url": final,
        "platform_id": platform_id,
        "category": None,
        "status": None,
        "campaign_dates": {"launched": None, "ended": None},
        # LLM numeric fields (may be missing)
        "pledged": extra.get("pledged"),
        "goal": extra.get("goal"),
        "backers": extra.get("backers"),
        "comments": extra.get("comments"),
        "views": extra.get("views"),
        "llm_score": extra.get("llm_score"),
        # score computed later; we keep duration/published_at absent in LLM-only mode
        "published_at": None,
        "duration_s": None,
    }
    return out


# ----- gadget filtering -----
GADGET_POS_KWS = {
    "smart","iot","sensor","bluetooth","wireless","home assistant","home automation","smart home",
    "lighting","light","lamp","charger","battery","power bank","robot","vacuum","air purifier",
    "camera","dashcam","projector","headphone","earbud","earbuds","audio","speaker","portable speaker",
    "keyboard","mouse","wearable","watch","smartwatch","tracker","e-ink","display","monitor","dock","hub",
    "router","wi-fi","wifi","nfc","rfid","arduino","raspberry pi","esp32","microcontroller","drone",
    "phone","android","ios","magnetic","charging","led","rgb","flashlight","torch","gimbal","stabilizer",
    "microphone","3d printer","laser engraver","gadget","electronics","hardware",
}
GADGET_NEG_KWS = {
    "novel","book","comic","zine","poetry","cookbook",
    "film","movie","documentary","short film","web series",
    "music","album","vinyl","band","song",
    "tabletop","board game","card game","rpg",
    "video game artbook","art book","graphic novel",
    "theatre","dance","fashion design","apparel",
    "restaurant","cafe","food truck","bakery",
    "charity","relief","fundraiser",
}
KS_ALLOW_CATS = {"technology","product design","gadgets","hardware","design","electronics"}
KS_DENY_CATS  = {"publishing","comics","games","music","film & video","art","food","fashion","journalism"}
IGG_ALLOW_CATS = {"tech & innovation","audio","home","productivity","transportation","travel & outdoors","wearables"}
IGG_DENY_CATS  = {"film","education","local businesses","community","music","art","comics","culture"}

def _text_blob(*parts) -> str:
    return " ".join([p for p in parts if p]).lower()

def _kw_hit(blob: str, kws: set[str]) -> bool:
    return any(k in blob for k in kws)

def is_gadget_by_category(platform: str, category: Optional[str]) -> Optional[bool]:
    if not category:
        return None
    c = category.strip().lower()
    if platform == "kickstarter":
        if c in KS_DENY_CATS:  return False
        if c in KS_ALLOW_CATS: return True
    elif platform == "indiegogo":
        if c in IGG_DENY_CATS:  return False
        if c in IGG_ALLOW_CATS: return True
    return None

def is_gadget_by_keywords(title: str, category: Optional[str]=None, desc: Optional[str]=None) -> bool:
    blob = _text_blob(title, category or "", desc or "")
    if _kw_hit(blob, GADGET_NEG_KWS):
        return False
    return _kw_hit(blob, GADGET_POS_KWS)

def llm_is_gadget(title: str, category: Optional[str], url: str) -> Optional[bool]:
    if not (CFG.gadget_llm_fallback and CLIENT):
        return None
    system = "Decide if the campaign is a consumer gadget/electronics/smart-home product (yes/no)."
    user = f"Title: {title}\nCategory: {category or ''}\nURL: {url}\nAnswer 'yes' or 'no' only."
    try:
        resp = CLIENT.chat.completions.create(
            model=CFG.openai_model, temperature=0.0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        ans = resp.choices[0].message.content.strip().lower()
        if "yes" in ans: return True
        if "no"  in ans: return False
    except Exception:
        pass
    return None

def is_gadget_candidate(item: Dict[str,Any]) -> bool:
    if not CFG.gadget_only:
        return True
    plat = (item.get("platform") or "").lower()
    cat  = item.get("category")
    title = item.get("title") or ""

    llm_decision = llm_is_gadget(title, cat, item.get("url") or "")
    if llm_decision is not None:
        print(f"LLM Decides Gadget for {item.get("url")}: {llm_decision}")
        return llm_decision

    cat_decision = is_gadget_by_category(plat, cat)
    if cat_decision is not None:
        print(f"Category Decides Gadget: {cat_decision}")
        return cat_decision

    KW_decision = is_gadget_by_keywords(title, category=cat)
    if KW_decision:
        print(f"Keywords Decide Gadget: {KW_decision}")
        return True

    return False
