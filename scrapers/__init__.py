# scrapers/__init__.py
from typing import Dict, Any, Optional
from parsers import detect_platform
from .kickstarter_scraper import extract_kickstarter_stats
from .indiegogo_scraper import extract_indiegogo_stats

def extract_stats(url: str, platform: Optional[str] = None) -> Dict[str, Any]:
    """
    Unified entrypoint. Returns:
    {
      pledged_raw, goal_raw, pledged_amount, pledged_currency,
      goal_amount, goal_currency, backers, comments, views,
      _sources: {...}
    }
    """
    plat = (platform or detect_platform(url) or "").lower()
    if "kickstarter" in plat:
        print("Kickstarter Scraper...")
        return extract_kickstarter_stats(url)
    if "indiegogo" in plat:
        print("Indiegogo Scraper...")
        return extract_indiegogo_stats(url)
    # Unknown platform → return empty shape
    return {
        "pledged_raw": None, "goal_raw": None,
        "pledged_amount": None, "pledged_currency": None,
        "goal_amount": None,   "goal_currency": None,
        "backers": None, "comments": None, "views": None,
        "_sources": {"platform": plat or "unknown"}
    }
