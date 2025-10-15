# state_ops.py
import os, re, json, math, urllib.parse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import CFG
from net import normalize_url, canonicalize_campaign_url, verify_exists, is_indiegogo_project_url
from parsers import is_campaigny_text

# ---------- state I/O ----------
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

# ---------- identity / keys ----------
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
    pid  = (item.get("platform_id") or "")
    pid = pid.strip() if isinstance(pid, str) else str(pid)
    url  = item.get("url") or ""
    if plat in ("kickstarter", "indiegogo") and pid:
        return f"{plat}:{pid}"
    plat2, nid = extract_native_id(url)
    return f"{plat2}:{nid}"

# ---------- gating / scoring / window ----------
def _primary_date_for_window(it: Dict[str, Any]) -> Optional[str]:
    return (it.get("published_at")
            or ((it.get("campaign_dates") or {}).get("launched"))
            or ((it.get("campaign_dates") or {}).get("ended")))

def is_within_window(it: Dict[str, Any], days: int = CFG.window_days) -> bool:
    ds = _primary_date_for_window(it)
    if not ds:
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
        det = max(0.0, s)
        if it.get("llm_score") is not None:
            it["score"] = float(it["llm_score"])
        else:
            it["score"] = round(max(0.0, min(1.0, 1 / (1 + math.exp(-det)))), 4)
    return items

# ---------- persistence helpers ----------
PERSIST_FIELDS = [
    "title","platform","url","platform_id",
    "ad_url","ad_platform",
    "views","backers","comments","goal","pledged",
    # --- NEW: keep raw and currency fields from scrapers ---
    "pledged_raw","goal_raw","pledged_currency","goal_currency",
    # -------------------------------------------------------
    "published_at","duration_s","category","status",
    "campaign_dates","llm_score","score"
]

def _snapshot_item(it: dict) -> dict:
    snap = {k: it.get(k) for k in PERSIST_FIELDS}

    # Normalize numeric strings for core metrics
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

    # Compute funding ratio
    g, p = snap.get("goal"), snap.get("pledged")
    if g not in (None, 0) and p is not None:
        try:
            snap["funding_ratio"] = (float(p) / float(g))
        except Exception:
            snap["funding_ratio"] = None
    else:
        snap["funding_ratio"] = None

    # If llm_score present, prefer it for score
    if snap.get("llm_score") is not None:
        snap["score"] = float(snap["llm_score"])

    return snap

def persist_discover_results(enriched_items: list[dict], state: dict) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    state.setdefault("items", {})
    for it in enriched_items:
        key = stable_key(it)
        entry = state["items"].get(key, {})

        snap = _snapshot_item(it)

        # Merge: write non-None values; ensure keys exist with None otherwise
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

def pick_unsent_items(state: dict, max_items: int = 5) -> list[tuple[str, dict]]:
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

def enrich_state_items_inplace(state: dict) -> int:
    items = state.get("items", {})
    if not items: return 0
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0

    # Lazy import to avoid circulars / heavy imports at module load
    from scrapers import extract_stats
    from parsers import parse_kickstarter, parse_indiegogo, parse_youtube, parse_vimeo, resolve_embeds_to_views

    def _enrich_one_entry(entry: dict) -> dict:
        from net import verify_exists
        url = entry.get("url") or ""
        if not url:
            return entry
        url = canonicalize_campaign_url(normalize_url(url))
        try:
            verify = verify_exists(url)
            if not verify.get("ok"):
                return entry

            lower = url.lower()
            parsed = {}
            # First: resolve embeds/views via your existing parser flow
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
                # Unknown platform: nothing to do
                return entry

            # Snap from legacy parsers (gives us views/comments/backers/goal/pledged when available)
            snap = _snapshot_item(parsed)
            entry.update(snap)
            entry["best_views"] = max(entry.get("best_views") or 0, parsed.get("views") or 0)

            # Second: for KS/IGG, call the robust scraper to fill missing fields + currencies/raw
            if "kickstarter.com" in lower or "indiegogo.com" in lower:
                try:
                    stats = extract_stats(url)
                except Exception:
                    stats = {}

                if stats:
                    # Map scraper fields directly onto entry before snapshot recompute
                    # Numeric
                    if stats.get("pledged_amount") is not None: entry["pledged"] = stats["pledged_amount"]
                    if stats.get("goal_amount")    is not None: entry["goal"]    = stats["goal_amount"]
                    if stats.get("backers")        is not None: entry["backers"] = stats["backers"]
                    if stats.get("comments")       is not None: entry["comments"]= stats["comments"]
                    # Raw/Currency
                    if stats.get("pledged_raw")      is not None: entry["pledged_raw"]      = stats["pledged_raw"]
                    if stats.get("goal_raw")         is not None: entry["goal_raw"]         = stats["goal_raw"]
                    if stats.get("pledged_currency") is not None: entry["pledged_currency"] = stats["pledged_currency"]
                    if stats.get("goal_currency")    is not None: entry["goal_currency"]    = stats["goal_currency"]

                    # Re-snapshot to ensure funding_ratio/score normalization stays consistent
                    snap2 = _snapshot_item(entry)
                    entry.update(snap2)

            return entry
        except Exception:
            return entry

    for key, entry in list(items.items()):
        before = (entry.get("backers"), entry.get("comments"), entry.get("views"), entry.get("pledged"), entry.get("goal"))
        entry = _enrich_one_entry(entry)
        after  = (entry.get("backers"), entry.get("comments"), entry.get("views"), entry.get("pledged"), entry.get("goal"))
        entry["last_enriched"] = today
        items[key] = entry
        if before != after:
            count += 1

    save_state(state)
    return count


# ---------- selection/report ----------
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
