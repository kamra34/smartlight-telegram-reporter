# scrapers/indiegogo_scraper.py
import os, re, json, math, html
from typing import Any, Dict, Optional
import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv

try:
    from price_parser import Price  # optional
except Exception:
    Price = None

load_dotenv()
from openai import OpenAI
_OPENAI = OpenAI()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def _headers():
    return {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.indiegogo.com/",
        "Upgrade-Insecure-Requests": "1",
    }

def _http_get(url: str, timeout: int = 25) -> requests.Response:
    s = requests.Session()
    s.headers.update(_headers())
    s.get("https://www.indiegogo.com/", timeout=timeout)
    r = s.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

# ---------- helpers (shared-ish) ----------
JSY = re.compile(r"(?:\bvar\b|\bfunction\b|\bwindow\.\w+|{.+}|;)", re.I)
HIDDEN_STYLE = re.compile(r"(display\s*:\s*none|visibility\s*:\s*hidden)", re.I)
NAV_WORDS = ("story","campaign","updates","comments","community","faq","perks","back it","contribute","funding")

def _clean_soup(soup: BeautifulSoup) -> None:
    for tag in soup(["script","style","noscript","template","svg","path","meta","link","iframe","head"]):
        tag.decompose()
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for el in soup.find_all(True):
        try:
            if el.has_attr("hidden"): el.decompose(); continue
            if el.get("aria-hidden","").lower() == "true": el.decompose(); continue
            style = el.get("style","")
            if style and HIDDEN_STYLE.search(style): el.decompose(); continue
        except Exception:
            pass

def _is_visible_text(s: str) -> bool:
    if not s or len(s.strip()) < 2: return False
    if JSY.search(s): return False
    if s.count('&quot;') >= 4 or re.search(r'[{[]\s*".*":', s): return False
    punct = sum(ch in "{}[];=<>&" for ch in s)
    if punct and punct/len(s) > 0.12: return False
    return True

def _context_snippet(node_text: str, parent, max_len: int = 360) -> str:
    t = node_text.strip()
    bits = [t]
    if parent:
        try:
            ptxt = parent.get_text(" ", strip=True)
        except Exception:
            ptxt = ""
        if ptxt and ptxt != t:
            bits.append(ptxt)
        for sib in [parent.previous_sibling, parent.next_sibling]:
            if sib is None: continue
            try:
                st = getattr(sib, "get_text", lambda *a, **k: "")(" ", strip=True)
            except Exception:
                st = ""
            if st:
                bits.append(st)
    return html.unescape(" | ".join(b for b in bits if b))[:360]

def _usefulness_score(s: str) -> float:
    sc = 0.0
    if re.search(r"[$€£¥]|USD|EUR|SEK|NOK|DKK|GBP|JPY|CHF|CAD|AUD|NZD|SGD|HKD|kr", s, re.I): sc += 3
    if re.search(r"\d", s): sc += 1
    if "%" in s: sc += 0.5
    low = s.lower()
    if sum(1 for w in NAV_WORDS if w in low) >= 1 and not re.search(r"\d", s): sc -= 2.5
    sc += max(0, 6 - math.log2(len(s) + 1))
    return sc

def _collect_meta_and_jsonld(html_text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    out: Dict[str, Any] = {"meta": [], "jsonld": [], "boot": []}

    # meta summaries
    for attrs in ({"property":"og:description"}, {"name":"twitter:description"}, {"name":"description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            c = (tag["content"] or "").strip()
            if c: out["meta"].append(c)

    # JSON-LD
    for tag in soup.find_all("script", attrs={"type":"application/ld+json"}):
        txt = (tag.string or tag.text or "").strip()
        if txt and "{" in txt:
            out["jsonld"].append(txt)

    # Hydration / window initial state patterns seen on IGG
    for tag in soup.find_all("script"):
        txt = (tag.string or tag.text or "").strip()
        if not txt or "{" not in txt: 
            continue
        if re.search(r"window\.(?:__INITIAL_STATE__|__NEXT_DATA__|__APOLLO_STATE__)", txt):
            out["boot"].append(txt[:4000])
        # data-hydration attribute on script tags
        if tag.has_attr("data-hydration") and "{" in txt:
            out["boot"].append(txt[:4000])
    return out

def _narrow_visible_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    _clean_soup(soup)
    chunks = []
    for node in soup.find_all(string=True):
        s = (node or "").strip()
        if not _is_visible_text(s): 
            continue
        if not any(ch.isalpha() for ch in s) and len(s) < 5:
            continue
        chunk = _context_snippet(s, getattr(node, "parent", None), 360)
        if chunk:
            chunks.append(chunk)

    seen, uniq = set(), []
    for c in chunks:
        k = c[:160]
        if k in seen: continue
        seen.add(k); uniq.append(c)
    uniq.sort(key=lambda s: -_usefulness_score(s))
    return "\n---\n".join(uniq[:10])

def _normalize_amount(raw: Optional[str]):
    if not raw:
        return None, None
    s = str(raw)
    if Price:
        try:
            p = Price.fromstring(s)
            if p and (p.amount is not None or p.amount_text):
                try: amt = float(str(p.amount or p.amount_text).replace(",", "."))
                except Exception: amt = None
                return amt, (p.currency or None)
        except Exception:
            pass
    s2 = s.replace("\u202f"," ").replace("\u00a0"," ").strip()
    cur = None
    mcur = re.search(r"(USD|EUR|SEK|NOK|DKK|GBP|JPY|CHF|CAD|AUD|NZD|SGD|HKD|[$€£¥]|kr)", s2, re.I)
    if mcur: cur = mcur.group(1)
    nums = re.findall(r"[\d][\d\s.,]*", s2)
    if nums:
        rawnum = nums[0].strip()
        last_com = rawnum.rfind(',')
        last_dot = rawnum.rfind('.')
        dec = ',' if last_com > last_dot else '.'
        tmp = rawnum.replace(' ', '').replace('\u202f','').replace('\u00a0','')
        if dec == ',': tmp = tmp.replace('.', '').replace(',', '.')
        else:          tmp = tmp.replace(',', '')
        try: amt = float(tmp)
        except Exception: amt = None
        return amt, cur
    return None, cur

# ---------- LLM extraction ----------
_PROMPT = """You are a multilingual data extractor for Indiegogo pages.
Input includes: meta descriptions, JSON-LD or boot JSON blobs, and top visible text snippets.

Extract:
- pledged_raw: string (as shown; raised so far)
- goal_raw: string (funding goal/target)
- backers: integer (a.k.a. contributions)
- comments: integer
- views: integer or null (rare on IGG; return null if not shown)

Rules:
- Use explicit values only; do not compute from percentages.
- 'Contributions' maps to 'backers'.
- Prefer structured data if clearly present.
Return only JSON with the required keys.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "pledged_raw": {"type": ["string","null"]},
        "goal_raw":    {"type": ["string","null"]},
        "backers":     {"type": ["integer","null"]},
        "comments":    {"type": ["integer","null"]},
        "views":       {"type": ["integer","null"]},
    },
    "required": ["pledged_raw","goal_raw","backers","comments","views"],
    "additionalProperties": False
}

def extract_indiegogo_stats(url: str) -> Dict[str, Any]:
    r = _http_get(url)
    html_text = r.text

    # Gather structured hints + visible snippets
    bits = _collect_meta_and_jsonld(html_text)
    narrowed = []
    if bits["meta"]:
        narrowed.append("MetaSummaries:\n" + "\n---\n".join(bits["meta"]))
    if bits["jsonld"]:
        narrowed.append("JSONLD:\n" + "\n\n".join(bits["jsonld"][:2]))
    if bits["boot"]:
        narrowed.append("BootBlobs:\n" + "\n\n".join(bits["boot"][:2]))
    narrowed.append("TopTextSnippets:\n" + _narrow_visible_text(html_text))
    payload = "\n\n".join(narrowed)[:12000]

    resp = _OPENAI.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role":"system","content":"You are a precise, multilingual information extractor."},
            {"role":"user","content": _PROMPT + "\n\nTEXT:\n" + payload}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "indiegogo_stats", "schema": _SCHEMA, "strict": True}
        }
    )
    data = json.loads(resp.choices[0].message.content)

    pledged_amount, pledged_currency = _normalize_amount(data.get("pledged_raw"))
    goal_amount, goal_currency       = _normalize_amount(data.get("goal_raw"))

    return {
        "pledged_raw": data.get("pledged_raw"),
        "goal_raw":    data.get("goal_raw"),
        "pledged_amount": pledged_amount,
        "pledged_currency": pledged_currency,
        "goal_amount": goal_amount,
        "goal_currency": goal_currency,
        "backers": data.get("backers"),
        "comments": data.get("comments"),
        "views": data.get("views"),
        "_sources": {"platform": "indiegogo"}
    }

link = "https://www.indiegogo.com/projects/odin-3-the-ultimate-6-120hz-oled-gaming-handheld#/"
stats = extract_indiegogo_stats(link)
print(stats)