import time
import os
from functools import wraps

def retry_on_failure(max_retries=None, delay=None):
    """
    Retry decorator for agent functions that may fail due to API issues
    
    Args:
        max_retries: Number of retry attempts (default from env or 2)
        delay: Seconds to wait between retries (default from env or 3.0)
    """
    if max_retries is None:
        max_retries = int(os.getenv("MAX_RETRIES", "2"))
    if delay is None:
        delay = float(os.getenv("RETRY_DELAY", "3.0"))
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            last_result = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is an error message
                    if result and isinstance(result, str):
                        if result.startswith("⚠️"):
                            # It's an error, but record it
                            last_result = result
                            
                            # Don't retry on authentication errors
                            if "Authentication" in result or "API Key" in result:
                                return result
                            
                            # Retry on other errors
                            if attempt < max_retries:
                                print(f"[RETRY] Attempt {attempt + 1} failed, retrying in {delay}s...")
                                time.sleep(delay)
                                continue
                            else:
                                return result
                        else:
                            # Success!
                            return result
                    else:
                        # Empty or invalid result
                        if attempt < max_retries:
                            print(f"[RETRY] Empty result, retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                        else:
                            return result or "⚠️ No response generated after all retries"
                        
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        print(f"[RETRY] Attempt {attempt + 1} failed with {type(e).__name__}, retrying...")
                        time.sleep(delay)
                    else:
                        return f"⚠️ Failed after {max_retries + 1} attempts: {str(e)[:200]}"
            
            # Should not reach here, but just in case
            if last_result:
                return last_result
            return f"⚠️ All retries exhausted. Last error: {str(last_error)[:200]}"
        
        return wrapper
    return decorator
