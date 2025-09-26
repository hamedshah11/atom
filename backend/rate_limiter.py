import time
from typing import Optional
from functools import wraps

class RateLimiter:
    """Simple rate limiter to avoid hitting API limits"""
    
    def __init__(self, min_delay: float = 1.0):
        self.min_delay = min_delay
        self.last_call_time: Optional[float] = None
    
    def wait_if_needed(self):
        """Wait if necessary to maintain minimum delay between calls"""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
        self.last_call_time = time.time()
    
    def __call__(self, func):
        """Decorator to rate limit function calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper

# Global rate limiter instance
# Adjust delay based on your OpenAI tier:
# - Free tier: 3-5 seconds
# - Tier 1: 1-2 seconds  
# - Tier 2+: 0.5-1 second
api_rate_limiter = RateLimiter(min_delay=1.5)
