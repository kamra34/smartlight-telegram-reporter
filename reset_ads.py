import json

STATE_FILE = "ads_state.json"

with open(STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)

items = state.get("items", {})
count = 0

for key, entry in items.items():
    if entry.get("sent") is not False:
        entry["sent"] = False
        count += 1

with open(STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2, ensure_ascii=False)

print(f"✅ Reset 'sent' to False for {count} items.")
