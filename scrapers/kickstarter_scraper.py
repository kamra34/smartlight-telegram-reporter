# kickstarter_scraper.py
import os, re, json, math, time, html, random, urllib.parse
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv

# --- Optional deps (safe to miss) ---
try:
    import cloudscraper  # type: ignore
except Exception:
    cloudscraper = None

try:
    from price_parser import Price  # type: ignore
except Exception:
    Price = None

try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _PLAYWRIGHT_OK = True
except Exception:
    _PLAYWRIGHT_OK = False

# --- OpenAI client (LLM extraction) ---
load_dotenv()
from openai import OpenAI
_OPENAI = OpenAI()

# ==============================
# Config
# ==============================
MODEL_NAME      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT_S  = 25
HTTP_RETRIES    = 3
UA              = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def _headers() -> Dict[str, str]:
    return {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.kickstarter.com/",
        "Upgrade-Insecure-Requests": "1",
    }

# ==============================
# HTTP
# ==============================
def http_get(url: str, timeout: int = HTTP_TIMEOUT_S, retries: int = HTTP_RETRIES) -> requests.Response:
    """
    Browser-y GET with cookie warm-up; falls back to cloudscraper if present.
    """
    last_err: Optional[Exception] = None

    # Browsery session with homepage warm-up (reduces 403s)
    try:
        sess = requests.Session()
        sess.headers.update(_headers())
        sess.get("https://www.kickstarter.com/", timeout=timeout)
        time.sleep(0.3 + random.random() * 0.5)

        for i in range(retries):
            try:
                r = sess.get(url, timeout=timeout, allow_redirects=True)
                if r.status_code == 403 and i < retries - 1:
                    time.sleep(0.6 * (2 ** i))
                    continue
                r.raise_for_status()
                return r
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (2 ** i))
    except Exception as e:
        last_err = e

    # cloudscraper fallback (optional)
    if cloudscraper is not None:
        try:
            scraper = cloudscraper.create_scraper(browser={"custom": UA})
            for i in range(retries):
                try:
                    r = scraper.get(url, headers=_headers(), timeout=timeout, allow_redirects=True)
                    if r.status_code == 403 and i < retries - 1:
                        time.sleep(0.6 * (2 ** i))
                        continue
                    r.raise_for_status()
                    return r
                except Exception as e:
                    last_err = e
                    time.sleep(0.6 * (2 ** i))
        except Exception as e:
            last_err = e

    raise last_err or RuntimeError("Failed to GET URL")

def canonicalize_campaign_url(url: str) -> str:
    try:
        u = urllib.parse.urlsplit(url)
        host = u.netloc.lower()
        path = u.path or "/"
        parts = [p for p in path.split("/") if p]
        if "kickstarter.com" in host and parts[:1] == ["projects"]:
            parts = parts[:3]  # /projects/{creator}/{slug}
            path = "/" + "/".join(parts)
        return urllib.parse.urlunsplit((u.scheme or "https", host, path, "", ""))
    except Exception:
        return url

# ==============================
# Fast path: <url>.json
# ==============================
def try_kickstarter_project_json(url: str) -> Optional[Dict[str, Any]]:
    jurl = url.rstrip("/") + ".json"
    try:
        r = http_get(jurl)
        if r.status_code == 200:
            data = r.json()
            src = data.get("project", data) if isinstance(data, dict) else {}
            if not isinstance(src, dict):
                return None

            def pick(*names):
                for n in names:
                    if n in src and src[n] is not None:
                        return src[n]
                return None

            out = {
                "pledged":  pick("pledged", "pledged_amount", "usd_pledged"),
                "goal":     pick("goal", "goal_amount", "static_usd_goal"),
                "backers":  pick("backers_count", "backer_count", "num_backers"),
                "comments": pick("comments_count", "comment_count"),
            }
            if any(v is not None for v in out.values()):
                return out
    except Exception:
        pass
    return None

# ==============================
# Playwright capture (GraphQL/Apollo)
# ==============================
def _deep_find(obj: Any, keys: set) -> Optional[Any]:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in keys and v is not None:
                    return v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None

def fetch_live_stats_via_graphql(url: str) -> Optional[Dict[str, Any]]:
    """
    If Playwright is available, open the page, capture /graph responses,
    and/or read Apollo cache to get pledged/goal/backers/comments during hydration.
    """
    if not _PLAYWRIGHT_OK:
        return None

    result: Dict[str, Any] = {}

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    try:
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()

        gql_payloads: list[str] = []

        def on_response(resp):
            try:
                if "/graph" in resp.url and resp.request.method == "POST":
                    ct = (resp.headers.get("content-type") or "").lower()
                    if "application/json" in ct:
                        t = resp.text()
                        if t and t.strip().startswith("{"):
                            gql_payloads.append(t)
            except Exception:
                pass

        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(2500)

        # Try GraphQL responses first (newest first)
        for payload in reversed(gql_payloads):
            try:
                j = json.loads(payload)
            except Exception:
                continue
            data = j.get("data") if isinstance(j, dict) else None
            if not isinstance(data, dict):
                continue

            pledged  = _deep_find(data, {"pledged", "pledgedAmount", "currentAmount"})
            goal     = _deep_find(data, {"goal", "goalAmount", "targetAmount"})
            backers  = _deep_find(data, {"backersCount", "backerCount", "numBackers"})
            comments = _deep_find(data, {"commentsCount", "commentCount"})

            if any(v is not None for v in (pledged, goal, backers, comments)):
                if pledged is not None:  result["pledged"]  = pledged
                if goal is not None:     result["goal"]     = goal
                if backers is not None:  result["backers"]  = backers
                if comments is not None: result["comments"] = comments
                return result

        # Fallback: read Apollo cache
        try:
            raw = page.evaluate("""() => {
                try {
                    const a = (window.__APOLLO_STATE__ || window.__APOLLO_CACHE__ || window.__INITIAL_STATE__ || {});
                    return JSON.stringify(a);
                } catch(e) { return "{}"; }
            }""")
            if raw and raw.strip().startswith("{"):
                j = json.loads(raw)
                pledged  = _deep_find(j, {"pledged", "pledgedAmount", "currentAmount"})
                goal     = _deep_find(j, {"goal", "goalAmount", "targetAmount"})
                backers  = _deep_find(j, {"backersCount", "backer_count", "backerCount", "numBackers"})
                comments = _deep_find(j, {"commentsCount", "comment_count"})
                if any(v is not None for v in (pledged, goal, backers, comments)):
                    if pledged is not None:  result["pledged"]  = pledged
                    if goal is not None:     result["goal"]     = goal
                    if backers is not None:  result["backers"]  = backers
                    if comments is not None: result["comments"] = comments
                    return result
        except Exception:
            pass

        # Last resort: return fully rendered HTML for narrowing
        try:
            html_rendered = page.content()
            return {"__rendered_html__": html_rendered}
        except Exception:
            pass
    finally:
        browser.close()
        pw.stop()

    return None

# ==============================
# HTML narrowing (language-agnostic)
# ==============================
JSY = re.compile(r"(?:\bvar\b|\bfunction\b|\bwindow\.\w+|{.+}|;)", re.I)
HIDDEN_STYLE = re.compile(r"(display\s*:\s*none|visibility\s*:\s*hidden)", re.I)
NAV_WORDS = ("campaign","creator","community","rewards","faq","updates","comments","back this project")

def collect_meta_summaries(html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    vals = []
    for attrs in (
        {"property": "og:description"},
        {"name": "twitter:description"},
        {"name": "description"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            c = (tag["content"] or "").strip()
            if c:
                vals.append(c)
    return vals

def clean_soup(soup: BeautifulSoup) -> None:
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

def is_visible_text(s: str) -> bool:
    if not s or len(s.strip()) < 2: return False
    if JSY.search(s): return False
    if s.count('&quot;') >= 4 or re.search(r'[{[]\s*".*":', s): return False
    punct = sum(ch in "{}[];=<>&" for ch in s)
    if punct and punct/len(s) > 0.12: return False
    return True

def context_snippet(node_text: str, parent, max_len: int = 360) -> str:
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
            if sib is None: 
                continue
            try:
                st = getattr(sib, "get_text", lambda *a, **k: "")(" ", strip=True)
            except Exception:
                st = ""
            if st:
                bits.append(st)
    snippet = " | ".join(b for b in bits if b)
    return html.unescape(snippet)[:360]

def parse_boot_json(soup: BeautifulSoup) -> list[str]:
    blobs: list[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        txt = (tag.string or tag.text or "").strip()
        if txt and "{" in txt:
            blobs.append(txt)
    for tag in soup.find_all("script"):
        txt = (tag.string or tag.text or "").strip()
        if not txt or "{" not in txt:
            continue
        if re.search(r"window\.(?:__APOLLO_STATE__|__APOLLO_CACHE__|__INITIAL_STATE__|__NEXT_DATA__)\s*=", txt):
            blobs.append(txt[:3500])
    return blobs[:3]

def deep_parse_boot_values(boot_blobs: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in boot_blobs:
        # Try full JSON
        try:
            j = json.loads(raw)
            out.setdefault("pledged",  _deep_find(j, {"pledged","pledged_amount","current_amount","amount_pledged","usd_pledged"}))
            out.setdefault("goal",     _deep_find(j, {"goal","goal_amount","static_usd_goal","target_amount"}))
            out.setdefault("backers",  _deep_find(j, {"backers_count","backer_count","num_backers"}))
            out.setdefault("comments", _deep_find(j, {"comments_count","comment_count"}))
        except Exception:
            pass
        # Try assignment form: window.X = {...}
        for m in re.finditer(r"=\s*({.*?})\s*;?\s*$", raw, re.S | re.M):
            js = m.group(1)
            try:
                j = json.loads(js)
                out.setdefault("pledged",  _deep_find(j, {"pledged","pledged_amount","current_amount","amount_pledged","usd_pledged"}))
                out.setdefault("goal",     _deep_find(j, {"goal","goal_amount","static_usd_goal","target_amount"}))
                out.setdefault("backers",  _deep_find(j, {"backers_count","backer_count","num_backers"}))
                out.setdefault("comments", _deep_find(j, {"comments_count","comment_count"}))
            except Exception:
                continue
    return out

def usefulness_score(s: str) -> float:
    sc = 0.0
    if re.search(r"[$€£¥]|USD|EUR|SEK|NOK|DKK|GBP|JPY|CHF|CAD|AUD|NZD|SGD|HKD|kr", s, re.I): sc += 3
    if re.search(r"\d", s): sc += 1
    if "%" in s: sc += 0.5
    low = s.lower()
    if sum(1 for w in NAV_WORDS if w in low) >= 1 and not re.search(r"\d", s): sc -= 2.5
    sc += max(0, 6 - math.log2(len(s) + 1))
    return sc

def collect_meta_summaries_and_narrow(html_text: str, cap_chars: int = 12000) -> str:
    meta_bits = collect_meta_summaries(html_text)

    soup = BeautifulSoup(html_text, "html.parser")
    clean_soup(soup)

    boot_blobs = parse_boot_json(soup)
    boot_text  = "\n\n".join(boot_blobs)

    chunks = []
    for node in soup.find_all(string=True):
        s = (node or "").strip()
        if not is_visible_text(s): 
            continue
        if not any(ch.isalpha() for ch in s) and len(s) < 5:
            continue
        chunk = context_snippet(s, getattr(node, "parent", None), 360)
        if chunk:
            chunks.append(chunk)

    seen, uniq = set(), []
    for c in chunks:
        k = c[:160]
        if k in seen: 
            continue
        seen.add(k); uniq.append(c)

    uniq.sort(key=lambda s: -usefulness_score(s))
    top_chunks = uniq[:10]

    parts = []
    if meta_bits:
        parts.append("MetaSummaries:\n" + "\n---\n".join(meta_bits))
    if boot_text:
        parts.append("StructuredBlobs:\n" + boot_text)
    if top_chunks:
        parts.append("TopTextSnippets:\n" + "\n---\n".join(top_chunks))

    payload = "\n\n".join(parts).strip()
    return payload[:cap_chars]

# ==============================
# LLM: schema + normalization
# ==============================
_EXTRACT_PROMPT = """You are a multilingual data extractor. You will see compact text from a Kickstarter project page:
- Some embedded JSON from the page boot code
- A few top visible text snippets

Extract these fields (any page language):
- pledged_raw: string (as shown; may include currency symbol/code)
- goal_raw: string
- backers: integer
- comments: integer
- views: integer or null (Kickstarter pages usually do not show page views)

Rules:
- Use explicit values only; do not compute from percentages.
- If a field is not clearly present, return null (or 0 only if explicitly shown).
- Prefer embedded JSON values if clearly present.
Return only JSON matching the schema.
"""

_JSON_SCHEMA = {
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

def _normalize_amount(raw: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    if not raw:
        return None, None
    s = str(raw)
    if Price:
        try:
            p = Price.fromstring(s)
            if p and (p.amount is not None or p.amount_text):
                try:
                    amt = float(str(p.amount or p.amount_text).replace(",", "."))
                except Exception:
                    amt = None
                return amt, (p.currency or None)
        except Exception:
            pass

    s2 = s.replace("\u202f"," ").replace("\u00a0"," ").strip()
    cur = None
    mcur = re.search(r"(USD|EUR|SEK|NOK|DKK|GBP|JPY|CHF|CAD|AUD|NZD|SGD|HKD|[$€£¥]|kr)", s2, re.I)
    if mcur:
        cur = mcur.group(1)
    nums = re.findall(r"[\d][\d\s.,]*", s2)
    if nums:
        rawnum = nums[0].strip()
        last_com = rawnum.rfind(',')
        last_dot = rawnum.rfind('.')
        dec = ',' if last_com > last_dot else '.'
        tmp = rawnum.replace(' ', '').replace('\u202f','').replace('\u00a0','')
        if dec == ',':
            tmp = tmp.replace('.', '').replace(',', '.')
        else:
            tmp = tmp.replace(',', '')
        try:
            amt = float(tmp)
        except Exception:
            amt = None
        return amt, cur
    return None, cur

# ==============================
# Public API
# ==============================
def extract_kickstarter_stats(url: str, use_browser_fallback: bool = True) -> Dict[str, Any]:
    """
    Universal extractor:
      1) canonicalize URL
      2) try <url>.json
      3) GET static HTML
      4) if still missing and allowed, Playwright capture (GraphQL/Apollo or rendered HTML)
      5) Narrow + LLM extraction (schema-enforced)
      6) Fill any gaps with JSON/GraphQL/Apollo values
      7) Return raw + normalized amounts
    """
    canon = canonicalize_campaign_url(url)

    # Fast path
    pj = try_kickstarter_project_json(canon) or {}

    # Static HTML
    r = http_get(canon)
    html_text = r.text

    # Live hydration fallback (optional)
    missing_core = not any(pj.get(k) for k in ("pledged","goal","backers"))
    if use_browser_fallback and missing_core:
        live = fetch_live_stats_via_graphql(canon)
        if isinstance(live, dict):
            if "__rendered_html__" in live:
                html_text = live["__rendered_html__"]
            else:
                pj.update(live)

    # Narrow HTML + embedded JSON snippets
    narrowed = collect_meta_summaries_and_narrow(html_text)
    soup = BeautifulSoup(html_text, "html.parser")
    boot_vals = deep_parse_boot_values(parse_boot_json(soup))

    # LLM extraction (schema enforced)
    messages = [
        {"role": "system", "content": "You are a precise, multilingual information extractor."},
        {"role": "user", "content": _EXTRACT_PROMPT + "\n\nTEXT:\n" + narrowed}
    ]
    resp = _OPENAI.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "kickstarter_stats", "schema": _JSON_SCHEMA, "strict": True}
        }
    )
    data = json.loads(resp.choices[0].message.content)

    # Patch from JSON/GraphQL/Apollo when LLM lacks fields
    def patch_raw(name: str, val: Any):
        if data.get(f"{name}_raw") in (None, "") and val not in (None, ""):
            data[f"{name}_raw"] = str(val)

    def patch_num(name: str, val: Any):
        if data.get(name) in (None, 0) and isinstance(val, (int, float, str)) and str(val).strip():
            try:
                data[name] = int(float(str(val)))
            except Exception:
                pass

    # project.json / GraphQL are authoritative if present
    patch_num("backers",  pj.get("backers"))
    patch_num("comments", pj.get("comments"))
    patch_raw("pledged",  pj.get("pledged"))
    patch_raw("goal",     pj.get("goal"))

    # embedded boot values as final fallback
    patch_num("backers",  boot_vals.get("backers"))
    patch_num("comments", boot_vals.get("comments"))
    patch_raw("pledged",  boot_vals.get("pledged"))
    patch_raw("goal",     boot_vals.get("goal"))

    # Normalize amounts (language/currency agnostic)
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
        "views": data.get("views"),  # KS doesn’t expose page views; expect None
        "_sources": {
            "project_json_used": bool(pj),
            "browser_fallback_used": use_browser_fallback and missing_core,
            "playwright_available": _PLAYWRIGHT_OK,
        }
    }
