# net.py
import random, time, urllib.parse, requests
from typing import Dict, Any, Tuple
import cloudscraper
from config import CFG

HEADERS_BASE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

def _headers():
    h = HEADERS_BASE.copy()
    h["User-Agent"] = random.choice(UA_POOL)
    return h

_scraper = cloudscraper.create_scraper(browser={'browser':'chrome','platform':'windows','mobile':False})

def http_get(url: str, timeout=CFG.http_timeout, retries=CFG.http_retries) -> requests.Response:
    last_exc = None
    for i in range(retries + 1):
        try:
            r = _scraper.get(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            if r.status_code == 403:
                r = requests.get(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1))
    raise last_exc

def http_head(url: str, timeout=CFG.http_timeout, retries=CFG.http_retries) -> requests.Response:
    last_exc = None
    for i in range(retries + 1):
        try:
            r = _scraper.head(url, headers=_headers(), timeout=timeout, allow_redirects=True)
            if r.status_code >= 400:
                r = _scraper.get(url, headers=_headers(), timeout=timeout, stream=True, allow_redirects=True)
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * (i + 1))
    raise last_exc

def verify_exists(url) -> Dict[str, Any]:
    try:
        r = http_head(url)
        ok = r.status_code < 400
        return {"ok": ok, "final_url": getattr(r, "url", url), "content_type": r.headers.get("Content-Type","")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def fetch_html(url: str) -> Tuple[str,str]:
    r = http_get(url)
    return r.text, getattr(r, "url", url)

def normalize_url(url: str):
    try:
        u = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, "", ""))
    except Exception:
        return url

def canonicalize_campaign_url(url: str) -> str:
    """Normalize KS/IGG URLs to main campaign page."""
    try:
        u = urllib.parse.urlsplit(url)
        host = u.netloc.lower()
        path = u.path or "/"
        parts = [p for p in path.split("/") if p]
        if "kickstarter.com" in host:
            if parts and parts[0] == "projects":
                parts = parts[:3]
                path = "/" + "/".join(parts)
        elif "indiegogo.com" in host:
            if parts and parts[0] == "projects":
                parts = parts[:2]
                path = "/" + "/".join(parts)
        return urllib.parse.urlunsplit((u.scheme, host, path, "", ""))
    except Exception:
        return url

def is_indiegogo_project_url(url: str) -> bool:
    try:
        u = urllib.parse.urlsplit(url)
        host = u.netloc.lower()
        parts = [p for p in (u.path or "/").split("/") if p]
        return ("indiegogo.com" in host) and (len(parts) >= 2) and (parts[0] == "projects")
    except Exception:
        return False
