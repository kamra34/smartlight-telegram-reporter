# config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str   = os.getenv("OPENAI_MODEL", "gpt-4o")
    google_api_key: str = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
    google_cx: str      = os.getenv("GOOGLE_CUSTOM_SEARCH_CX", "")
    timezone: str       = os.getenv("TZ", "Europe/Stockholm")
    state_file: str     = os.getenv("ADS_STATE_FILE", "ads_state.json")
    cooldown_days: int  = int(os.getenv("COOLDOWN_DAYS", "7"))
    improvement_pct: float = float(os.getenv("IMPROVEMENT_PCT", "0.15"))
    http_timeout: int   = int(os.getenv("HTTP_TIMEOUT", "18"))
    http_retries: int   = int(os.getenv("HTTP_RETRIES", "2"))
    use_llm: bool       = os.getenv("USE_LLM", "true").lower() == "true"
    notify_min: int     = int(os.getenv("NOTIFY_MIN", "3"))
    notify_max: int     = int(os.getenv("NOTIFY_MAX", "5"))
    window_days: int    = int(os.getenv("WINDOW_DAYS", "150"))
    gadget_only: bool   = os.getenv("GADGET_ONLY", "true").lower() == "true"
    gadget_llm_fallback: bool = os.getenv("GADGET_LLM_FALLBACK", "true").lower() == "true"

CFG = Config()

# Optional clients
try:
    from openai import OpenAI
    CLIENT = OpenAI(api_key=CFG.openai_api_key) if (CFG.use_llm and CFG.openai_api_key) else None
except Exception:
    CLIENT = None
