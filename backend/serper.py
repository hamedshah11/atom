import os
import time
import requests
from typing import List, Dict, Any

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

class SerperError(RuntimeError):
    pass

def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 300:
        raise SerperError(f"Serper HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def web_search_serper(query: str, num: int = 5, retries: int = 2, backoff: float = 1.2) -> List[Dict[str, str]]:
    """
    Minimal Google Serper search.
    Returns: list of dicts: {"title", "link", "snippet"}
    """
    if not SERPER_API_KEY:
        return []

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = _post_json(url, headers, payload)
            results = []
            for item in (data.get("organic") or [])[:num]:
                results.append({
                    "title": item.get("title", "")[:200],
                    "link": item.get("link", "")[:400],
                    "snippet": item.get("snippet", "")[:300]
                })
            return results
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise SerperError(f"Serper failed after retries: {e}") from e
    return []
