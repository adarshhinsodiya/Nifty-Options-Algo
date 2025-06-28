import time
from typing import Callable, Any, Dict
from functools import wraps


class RateLimiter:
    """
    Rate limiter to prevent API throttling
    """
    
    def __init__(self, throttle_ms: int = 200):
        """
        Initialize the rate limiter    
        
        Args:
            throttle_ms: Minimum time between API calls in milliseconds
        """
        self.throttle_seconds = throttle_ms / 1000.0
        self.last_call_time: Dict[str, float] = {}
    
    def limit(self, key: str = "default") -> None:
        """
        Apply rate limiting for a specific key
        
        Args:
            key: Identifier for the rate limit (e.g., API endpoint)
        """
        current_time = time.time()
        
        if key in self.last_call_time:
            elapsed = current_time - self.last_call_time[key]
            if elapsed < self.throttle_seconds:
                sleep_time = self.throttle_seconds - elapsed
                time.sleep(sleep_time)
        
        self.last_call_time[key] = time.time()


def rate_limited(limiter=None, key="default"):
    """
    Decorator for rate-limited functions
    
    Args:
        limiter: Either a RateLimiter instance or a lambda function that returns one
        key: Identifier for the rate limit
        
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the actual rate limiter instance
            actual_limiter = limiter(args[0]) if callable(limiter) else limiter
            actual_limiter.limit(key)
            return func(*args, **kwargs)
        return wrapper
    return decorator
