import os, io, time, requests, pandas as pd
from datetime import datetime
from dateutil import tz
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def tg_send_text(text, parse_mode=None):
    chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
    for chunk in chunks:
        data = {
            "chat_id": CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True
        }
        if parse_mode:  # only add if not None
            data["parse_mode"] = parse_mode
        r = requests.post(f"{API}/sendMessage", data=data, timeout=60)
        if not r.ok:
            raise RuntimeError(f"sendMessage failed: {r.status_code} {r.text}")
        time.sleep(0.3)


def tg_send_csv_bytes(filename, csv_bytes, caption=""):
    files = {"document": (filename, csv_bytes, "text/csv")}
    data = {"chat_id": CHAT_ID, "caption": caption}
    r = requests.post(f"{API}/sendDocument", data=data, files=files, timeout=120)
    if not r.ok:
        raise RuntimeError(f"sendDocument failed: {r.status_code} {r.text}")

now = datetime.now(tz.gettz("Europe/Stockholm"))
today_iso = now.strftime("%Y-%m-%d")

import time, json, re, requests

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- timing start ----
t0 = time.perf_counter()

# SYSTEM_PROMPT = f"""
# Search for the most‑viewed and highest‑engagement recent ads or launch videos across Kickstarter, Indiegogo, YouTube (consumer electronics/smart home/design), TikTok Creative Center (home & living/electronics), Facebook Ads Library (where accessible), and Vimeo. Focus on last 7 days (fallback to 14). Include lighting/smart‑home first, but don’t limit to them—add any standout creative examples.

# For each result, collect and maintain structured fields and output **three CSV blocks** daily in this order:

# 1) campaigns(id, platform, title, url, category, status, goal, pledged, backers, comments, updates, first_seen, last_seen)
#    - id: stable id (platform+native id)
#    - goal/pledged in USD if possible; otherwise include currency code prefix (e.g., EUR 12,345).
#    - first_seen/last_seen: ISO 8601.

# 2) videos(id, platform, campaign_id, host, host_video_id, url, views, likes, published_at, last_checked)
#    - host ∈ {youtube, vimeo, tiktok, meta}.
#    - metrics are integers; published_at/last_checked ISO 8601.

# 3) scores(campaign_id, date, score, rank, features_json)
#    - score formula: 0.5·views_z + 0.2·funding_ratio_z + 0.2·backers_per_day_z + 0.1·comments_per_day_z; include features_json with raw features and z‑scores.

# Then provide a **Top 10** list with direct **video links** (one primary link per item), plus a creative teardown for each: 3‑sec hook, story arc (problem→solution→proof→CTA), visual motifs (lighting, moves, typography), sound/VO notes, pacing, and the single biggest takeaway.

# Deduplicate, skip NSFW, prefer EU/US examples when volume is high. Rank consistently day over day. If a project was seen before, update last_seen and delta metrics.

# Send a concise message with today’s date, Top 10 titles + primary links, and attach the three CSV blocks as code blocks. Also return all contents here in chat. Finish with 3 trend observations and 3 testable ideas for SmartLight.
# """

SYSTEM_PROMPT = f"""
You are a marketing research agent. Produce a DAILY creative performance report for the most-viewed and highest-engagement ads and launch videos.

TODAY (timezone Europe/Stockholm): {today_iso}

SOURCES: Kickstarter, Indiegogo, YouTube, TikTok (home & living / smart-home / lighting), Facebook Ads Library, Vimeo. Focus on the last 7–14 days. Prioritize EU/US examples.

STRUCTURE YOUR OUTPUT EXACTLY AS FOLLOWS:

### Daily Creative Performance Report
Date: {today_iso} (Europe/Stockholm)

Key take-aways
• 3–5 concise bullets summarizing main insights (use "• " not dashes).
• Make them narrative but short.

Top creatives (sorted by score)
1. Title – short one-line creative summary and score insight.
2. …
3. …

---
#### Creative details
Return a JSON array. Never invent or guess URLs.

For each item include:
- platform: TikTok | YouTube | Kickstarter | Indiegogo | Vimeo | Facebook
- title
- post_date (ISO-8601)
- platform_id:
  * TikTok: numeric video_id (e.g., 7412345678901234567)
  * YouTube: video_id (e.g., PHbloom729)
  * Kickstarter: project slug "owner/slug" (e.g., glowcube/smart-rgb-desk-lamp)
  * Indiegogo: project slug (e.g., lumibar-retractable-led-strip)
  * Vimeo: video_id
  * Facebook: page id OR ad id (if unknown, leave blank)
- link: ONLY IF you are 100% certain and it resolves publicly. Otherwise, set "" (empty string).
- views
- likes OR backers (as applicable)
- comments
- score

Example:
```json
[
  {{"platform":"YouTube","title":"Philips Hue Gradient Bloom Review","post_date":"2025-10-02","platform_id":"PHbloom729","link":"https://www.youtube.com/watch?v=PHbloom729","views":850000,"likes":27000,"comments":1300,"score":0.72}},
  {{"platform":"Kickstarter","title":"GlowCube Smart RGB Desk Lamp","post_date":"2025-10-04","platform_id":"glowcube/smart-rgb-desk-lamp","link":"","views":150000,"backers":3500,"comments":128,"score":0.88}}
]

DATA RULES:

- If not 100% sure about the URL, leave "link": "" and fill "platform_id".
- Do NOT invent links or domains. Use the correct project/user/video slug or ID.
- Prefer YouTube/Kickstarter/Indiegogo/Vimeo where URLs are stable; TikTok/Facebook links are often region-locked.
- Integers for numeric metrics.
- ISO-8601 for dates.
- Total message ≤ 3500 chars.
- No CSV or attachments.
- Keep formatting readable in plain text (avoid Markdown in the report body).
- Output exactly ONE ```json code block for Creative details.
- Do not output any other JSON arrays (no empty [] placeholders).
"""
def verify_url_live(url):
    """Returns True if the URL loads with status 200–399"""
    try:
        r = requests.get(url, timeout=15, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0"
        })
        return 200 <= r.status_code < 400
    except requests.RequestException:
        return False

def canonical_url(item):
    plat = str(item.get("platform") or "").strip().lower()
    
    pid  = str(item.get("platform_id") or "").strip()
    link = str(item.get("link") or "").strip()

    # If already provided, verify it
    if link and verify_url_live(link):
        return link

    candidates = []

    if plat == "youtube" and pid:
        candidates.append(f"https://www.youtube.com/watch?v={pid}")
    if plat == "kickstarter" and pid:
        if "/" in pid:
            owner, slug = pid.split("/", 1)
            candidates.append(f"https://www.kickstarter.com/projects/{owner}/{slug}")
        else:
            candidates.append(f"https://www.kickstarter.com/projects/{pid}")
    if plat == "indiegogo" and pid:
        candidates.append(f"https://www.indiegogo.com/projects/{pid}")
    if plat == "vimeo" and pid.isdigit():
        candidates.append(f"https://vimeo.com/{pid}")
    if plat == "tiktok" and pid.isdigit():
        candidates.append(f"https://www.tiktok.com/@_/video/{pid}")
    if plat == "facebook" and "ads" not in pid.lower():
        # e.g., page IDs (not perfect)
        candidates.append(f"https://www.facebook.com/{pid}")

    # Try each candidate until one works
    for candidate in candidates:
        if verify_url_live(candidate):
            return candidate

    return ""  # none worked


#----- call OpenAI -----

resp = client.responses.create(
model="o3",
input=[{"role": "system", "content": SYSTEM_PROMPT}]
)

md_report = resp.output_text or ""

#----- extract first json ... block safely -----

# ----- extract the LAST ```json ... ``` block (ignore empty ones) -----
blocks = re.findall(r"```json\s*(.*?)\s*```", md_report, flags=re.DOTALL | re.IGNORECASE)
details_json = "[]"
for b in reversed(blocks):
    if b.strip() and b.strip() != "[]":
        details_json = b.strip()
        break

items = []
try:
    items = json.loads(details_json)
except json.JSONDecodeError:
    items = []

#----- verify/canonicalize links -----

# ----- canonicalize links -----
for it in items:
    it["link"] = canonical_url(it)

# ----- rebuild ONLY the chosen block (replace the last block) -----
pretty = json.dumps(items, ensure_ascii=False, separators=(",",":"))

def replace_last_json_block(text, new_block):
    matches = list(re.finditer(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL|re.IGNORECASE))
    if not matches:
        return text
    last = matches[-1]
    start, end = last.span()
    return text[:start] + f"```json\n{new_block}\n```" + text[end:]

md_report = replace_last_json_block(md_report, pretty)


#----- send to Telegram (header markdown, body plain) -----
header = f"*SmartLight Bot* — {today_iso}\nTop ads daily report."
tg_send_text(header, parse_mode="Markdown")
tg_send_text(md_report, parse_mode=None)


#---- timing end ----

elapsed = time.perf_counter() - t0
mins = int(elapsed // 60)
secs = elapsed - mins * 60
print(f"Done. Elapsed: {mins}m {secs:.1f}s")