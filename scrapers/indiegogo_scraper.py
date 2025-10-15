# scrapers/indiegogo_scraper.py
from __future__ import annotations
import re, json, html
from typing import Any, Dict, Optional, Tuple, List
from collections import Counter

from bs4 import BeautifulSoup
from net import canonicalize_campaign_url, fetch_html
from parsers import llm_only_extract_campaign  # your numeric-only LLM extractor

# Prefer multi-char tokens first so "$" doesn't beat "CA$"
_CURRENCY_PATTERNS = [
    r"CAD|AUD|NZD|HKD|SGD|USD|EUR|GBP|JPY|SEK|NOK|DKK|CHF|MXN|INR|CNY|KRW|TWD|TRY|PLN|CZK|HUF|ILS|AED",
    r"CA\$", r"US\$", r"A\$", r"NZ\$", r"HK\$", r"SG\$",
    r"€", r"£", r"¥", r"₩", r"₺", r"₪", r"₱", r"₫", r"₹",
    r"\$",
    r"\bkr\b",
]
_CURRENCY_RE = re.compile("|".join(_CURRENCY_PATTERNS), re.I)
# amount-ish
_AMOUNT_RE = re.compile(r"\d[\d\s.,]*")

def _deep_find(obj: Any, keys: set[str]) -> Optional[Any]:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in keys and isinstance(v, (str, int, float)) and v not in ("", None):
                    return v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None

def _scan_json_candidates(soup: BeautifulSoup) -> List[str]:
    cands: List[str] = []
    # JSON-LD blocks
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        txt = (tag.string or tag.text or "").strip()
        if not txt or "{" not in txt: continue
        try:
            j = json.loads(txt)
        except Exception:
            continue
        cur = _deep_find(j, {"priceCurrency","currency","acceptedCurrency","currencyCode"})
        if isinstance(cur, str):
            m = _CURRENCY_RE.search(cur)
            if m: cands.append(m.group(0))

    # Window state / hydration blobs
    for tag in soup.find_all("script"):
        txt = (tag.string or tag.text or "").strip()
        if not txt or "{" not in txt: continue
        # Cheap find for "currency" fields
        for m in re.finditer(r'"(?:currency|currencyCode|priceCurrency)"\s*:\s*"([^"]+)"', txt, re.I):
            val = m.group(1)
            mm = _CURRENCY_RE.search(val)
            if mm: cands.append(mm.group(0))
    return cands

def _scan_meta_candidates(soup: BeautifulSoup) -> List[str]:
    cands: List[str] = []
    for attrs in (
        {"property":"og:price:currency"}, {"name":"twitter:data1"},
        {"name":"twitter:label1"}, {"name":"description"}, {"property":"og:description"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if not tag: continue
        val = (tag.get("content") or "").strip()
        if not val: continue
        for m in _CURRENCY_RE.finditer(val):
            cands.append(m.group(0))
    return cands

def _scan_text_near_amounts(html_text: str) -> List[str]:
    """
    Heuristic: find currency tokens that appear immediately before/after numbers.
    E.g., 'CA$ 12,345', '$9,999', '12.345 kr'
    """
    cands: List[str] = []
    # currency before amount
    for m in re.finditer(rf"({_CURRENCY_RE.pattern})\s*({_AMOUNT_RE.pattern})", html_text, re.I):
        tok = m.group(1)
        cands.append(tok)
    # amount before currency
    for m in re.finditer(rf"({_AMOUNT_RE.pattern})\s*({_CURRENCY_RE.pattern})", html_text, re.I):
        tok = m.group(2)
        cands.append(tok)
    return cands

def _choose_currency(cands: List[str]) -> Optional[str]:
    if not cands:
        return None
    # normalize casing for codes; keep symbols as-is
    norm = []
    for c in cands:
        cc = c.upper() if re.fullmatch(r"[A-Z]{3}", c, re.I) else c
        norm.append(cc)
    # Prefer specific tokens over bare "$"
    # Rank: length desc, then frequency
    freq = Counter(norm)
    ranked = sorted(freq.items(), key=lambda kv: (len(kv[0]), kv[1]), reverse=True)
    return ranked[0][0]

def _fmt_raw(amount: Optional[float], currency: Optional[str]) -> Optional[str]:
    if amount is None:
        return None
    try:
        txt = f"{float(amount):,.0f}" if float(amount).is_integer() else f"{float(amount):,.2f}"
    except Exception:
        return None
    return f"{currency} {txt}" if currency else txt

def extract_indiegogo_stats(url: str) -> Dict[str, Any]:
    """
    Unified schema:
      pledged_raw, goal_raw, pledged_amount, pledged_currency,
      goal_amount, goal_currency, backers, comments, views, _sources
    """
    canon = canonicalize_campaign_url(url)
    html_text, final = fetch_html(canon)

    # 1) Numeric-only pass via your LLM helper (works reliably on IGG)
    llm = llm_only_extract_campaign(final, title_hint=None) or {}

    pledged_amount = llm.get("pledged")
    goal_amount    = llm.get("goal")
    backers        = llm.get("backers")
    comments       = llm.get("comments")
    views          = llm.get("views")

    # 2) Currency detection (multi-source)
    soup = BeautifulSoup(html_text, "html.parser")
    cands = []
    cands += _scan_json_candidates(soup)
    cands += _scan_meta_candidates(soup)
    cands += _scan_text_near_amounts(html_text)
    currency = _choose_currency(cands)

    # 3) Raw strings synthesized from numeric + currency
    pledged_raw = _fmt_raw(pledged_amount, currency)
    goal_raw    = _fmt_raw(goal_amount, currency)

    return {
        "pledged_raw": pledged_raw,
        "goal_raw":    goal_raw,
        "pledged_amount": pledged_amount,
        "pledged_currency": currency,
        "goal_amount": goal_amount,
        "goal_currency": currency,
        "backers": backers,
        "comments": comments,
        "views": views,
        "_sources": {
            "indiegogo_llm_only": True,
            "currency_candidates": cands[:10],  # keep a short trace for debugging
        }
    }
