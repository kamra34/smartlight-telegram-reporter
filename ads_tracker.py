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

import os
from datetime import datetime
from typing import Dict, Any, List
from config import CFG
from state_ops import (
    load_state, save_state, stable_key,
    compute_scores,
    persist_discover_results, pick_unsent_items, enrich_state_items_inplace,
    _primary_post_date
)
from parsers import (
    is_gadget_candidate, detect_platform
)
from scrapers import extract_stats 

# ---- Google CSE ----
try:
    from googleapiclient.discovery import build as gbuild
except Exception:
    gbuild = None

DOMAINS = ["kickstarter.com", "indiegogo.com", "youtube.com", "vimeo.com"]
QUERY_SUFFIXES = [
    '(launch OR "now live" OR trailer) (lighting OR "smart home" OR gadget OR electronics)',
    '(Kickstarter OR Indiegogo) (trailer OR launch) (design OR lighting OR home)',
    '(crowdfunding campaign) (video OR trailer) (lighting OR smart home OR gadget)',
]
CORE_FIELDS = [
    "pledged", "goal", "backers", "comments", "views",
    "pledged_raw", "goal_raw", "pledged_currency", "goal_currency",
    "funding_ratio", "platform", "title", "url"
]

def rotated(seq):
    if not seq: return seq
    k = datetime.now().timetuple().tm_yday % len(seq)
    return seq[k:] + seq[:k]

def google_search_site(domain: str, query_suffix: str, num: int = 5, date_restrict: str = "d7", page: int = 1):
    if not gbuild:
        raise RuntimeError("google-api-python-client not available")
    if not (CFG.google_api_key and CFG.google_cx):
        raise RuntimeError("Missing Google CSE creds")
    svc = gbuild("customsearch", "v1", developerKey=CFG.google_api_key)
    q = f"site:{domain} {query_suffix}".strip()
    start = (page - 1) * 10 + 1
    try:
        resp = svc.cse().list(q=q, cx=CFG.google_cx, num=min(num,10), dateRestrict=date_restrict, start=start).execute()
    except Exception:
        resp = svc.cse().list(q=q, cx=CFG.google_cx, num=min(num,10), dateRestrict="d14", start=start).execute()
    items = resp.get("items", []) or []
    return [{"title": it.get("title"), "link": it.get("link"), "displayLink": it.get("displayLink")} for it in items]

# ---- agent discover ----
def agent_discover(max_total=20) -> Dict[str,Any]:
    """
    Return CSE hits only (title/link). LLM-only extraction happens in discover().
    """
    results = []
    domains = rotated(DOMAINS)
    queries = rotated(QUERY_SUFFIXES)
    page = (datetime.now().timetuple().tm_yday % 2) + 1

    for domain in domains:
        q_suffix = queries[0]; queries = queries[1:] + queries[:1]
        try:
            candidates = google_search_site(domain, q_suffix, num=15, page=page)
        except Exception:
            continue

        for it in candidates:
            if it.get("link"):
                results.append({"title": it.get("title"), "link": it.get("link")})
            if len(results) >= max_total:
                break
        if len(results) >= max_total:
            break

    # Dedupe by link
    seen, dedup = set(), []
    for r in results:
        u = r.get("link")
        if u and u not in seen:
            seen.add(u); dedup.append(r)
    return {"results": dedup[:max_total]}

# ---- discover pipeline ----
def discover(top_k=5, add_new_only: bool = True, update_existing: bool = False):
    """
    Discover pipeline:
      - Search candidates (Google CSE)
      - Skip non-gadgets (no fetch)
      - For gadgets, route to platform scraper (Kickstarter/Indiegogo)
      - Score, dedupe, persist top_k
    """
    data = agent_discover(max_total=15)
    hits = data.get("results", [])
    if not hits:
        print("No candidates found.")
        return

    state = load_state()
    items: List[Dict[str, Any]] = []

    def prefer(a, b):
        """Pick a if it's set; otherwise b."""
        return a if a not in (None, "", 0) else b

    for it in hits:
        url = it.get("link") or it.get("url") or ""
        title = it.get("title") or ""
        if not url:
            continue

        platform = detect_platform(url)
        tmp_item = {"platform": platform, "category": None, "title": title, "url": url}
        if not is_gadget_candidate(tmp_item):
            continue  # skip non-gadgets early

        # 1) Try platform scraper (Kickstarter/Indiegogo)
        stats: Dict[str, Any] = {}
        try:
            stats = extract_stats(url, platform=platform) or {}
            print(f"[discover] scraped {platform}: {stats.get('pledged_raw')} / {stats.get('goal_raw')} "
                  f"({stats.get('backers')} backers, {stats.get('comments')} comments)")
        except Exception as e:
            print(f"[discover] platform scraper failed for {url}: {e}")

        # 2) Merge into a single item (we keep llm-only off for now; can add later)
        item: Dict[str, Any] = {
            "url": url,
            "title": title,
            "platform": platform,
            # prefer platform-scraped numerics
            "pledged":  stats.get("pledged_amount"),
            "goal":     stats.get("goal_amount"),
            "backers":  stats.get("backers"),
            "comments": stats.get("comments"),
            "views":    stats.get("views"),
            # keep raw/currency for display/debug
            "pledged_raw":      stats.get("pledged_raw"),
            "pledged_currency": stats.get("pledged_currency"),
            "goal_raw":         stats.get("goal_raw"),
            "goal_currency":    stats.get("goal_currency"),
            # llm_score not used here but included in CORE_FIELDS for future merge
            "llm_score": None,
        }

        # 3) Compute funding ratio if possible
        try:
            if item.get("pledged") and item.get("goal") and float(item["goal"]) > 0:
                item["funding_ratio"] = float(item["pledged"]) / float(item["goal"])
            else:
                item["funding_ratio"] = None
        except Exception:
            item["funding_ratio"] = None

        # 4) Ensure ALL core fields exist on the item (prevents dropping at persist-time)
        for k in CORE_FIELDS:
            item.setdefault(k, None)

        items.append(item)

    if not items:
        print("No gadget items extracted.")
        return

    # Score (state_ops.compute_scores prefers llm_score when present)
    items_scored = compute_scores(items)

    # Only NEW items unless updating
    if add_new_only and not update_existing:
        items_scored = [it for it in items_scored if stable_key(it) not in state.get("items", {})]

    if not items_scored:
        print("No new items to add (all were already in state).")
        return

    # Rank and persist
    items_scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    to_persist = items_scored[:top_k]

    # Project to guarantee core fields reach persistence even if saver whitelists
    def _project_for_state(it: Dict[str, Any]) -> Dict[str, Any]:
        projected = {k: it.get(k, None) for k in CORE_FIELDS}
        # include any extra keys compute_scores added (e.g., 'score')
        projected.update({k: v for k, v in it.items() if k not in projected})
        return projected

    to_persist = [_project_for_state(x) for x in to_persist]

    # Debug sample (optional)
    # print(json.dumps({stable_key(x): {k: x.get(k) for k in CORE_FIELDS} for x in to_persist}, indent=2))

    persist_discover_results(to_persist, state)
    print(f"Persisted {len(to_persist)} new items.")


# ---- notify ----
def send_to_telegram(token: str, chat_id: str, text: str):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": False})
    r.raise_for_status()

def notify():
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat  = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID missing")

    state = load_state()

    # How many to send? Prefer config, default to 3
    max_to_send = int(os.getenv("NOTIFY_MAX", str(getattr(CFG, "notify_max", 3))))
    max_to_send = max(1, min(max_to_send, 10))  # sane bounds

    candidates = pick_unsent_items(state, max_items=max_to_send)

    def fmt_money(amount, currency, raw_fallback):
        """
        Render like 'CA$ 9,712' when possible.
        - amount: numeric
        - currency: 'USD', 'EUR', 'CA$', '$', 'kr', etc.
        - raw_fallback: original raw string from scraper
        """
        if amount is None:
            return None
        try:
            amt_txt = f"{float(amount):,.0f}" if float(amount).is_integer() else f"{float(amount):,.2f}"
        except Exception:
            return raw_fallback or None
        if currency:
            return f"{currency} {amt_txt}"
        return raw_fallback or amt_txt

    # If no unsent items, send a fallback message and exit
    if not candidates:
        try:
            send_to_telegram(token, chat, "No new campaigns to notify today.")
        except Exception as e:
            print(f"Telegram send failed for fallback message: {e}")
        return

    sent_any = False
    for key, entry in candidates:
        title     = entry.get("title") or "(untitled)"
        platform  = entry.get("platform") or ""
        url       = entry.get("url") or ""

        backers   = entry.get("backers")
        comments  = entry.get("comments")
        views     = entry.get("views")
        fr        = entry.get("funding_ratio")
        score     = entry.get("score")

        pledged   = entry.get("pledged")
        goal      = entry.get("goal")
        p_cur     = entry.get("pledged_currency")
        g_cur     = entry.get("goal_currency")
        p_raw     = entry.get("pledged_raw")
        g_raw     = entry.get("goal_raw")

        pledged_txt = fmt_money(pledged, p_cur, p_raw)
        goal_txt    = fmt_money(goal,    g_cur, g_raw)

        bits = []
        if backers is not None:
            try: bits.append(f"{int(backers):,} backers")
            except: bits.append(f"{backers} backers")
        if pledged_txt: bits.append(f"pledged: {pledged_txt}")
        if goal_txt:    bits.append(f"goal: {goal_txt}")
        if fr:          bits.append(f"{float(fr)*100:.0f}% funded")
        # if comments is not None:
        #     try: bits.append(f"{int(comments):,} comments")
        #     except: bits.append(f"{comments} comments")
        # if views is not None:
        #     try: bits.append(f"{int(views):,} views")
        #     except: bits.append(f"{views} views")
        # if score is not None:
        #     try: bits.append(f"score {float(score):.2f}")
        #     except: bits.append(f"score {score}")

        metrics_line = " · ".join(bits) if bits else "engagement signals present"

        text = (
            f"{title}\n"
            f"Platform: {platform}\n"
            f"Link: {url}\n"
            f"{metrics_line}"
        )

        try:
            send_to_telegram(token, chat, text)
            entry["sent"] = True
            state["items"][key] = entry
            sent_any = True
        except Exception as e:
            print(f"Telegram send failed for {key}: {e}")

    if sent_any:
        save_state(state)
        print(f"Delivered {len([1 for _, e in candidates if state['items'][_].get('sent')])} items and marked sent=True.")
    else:
        print("No items were delivered.")


# ---- enrich existing items ----
def enrich_state_items():
    state = load_state()
    if not state.get("items"):
        print("State has no items to enrich."); return
    count = enrich_state_items_inplace(state)
    print(f"Enriched {count} items and saved state.")

# ---- CLI ----
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
