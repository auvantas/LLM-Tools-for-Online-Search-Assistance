"""Core components for crawl4ai_lite"""
import time
import os
import json
from typing import Any, Optional
from datetime import datetime, timedelta

class RetryPolicy:
    """Simple retry policy with exponential backoff"""
    def __init__(self, attempts: int = 3, delay: int = 10, exponential_backoff: bool = True):
        self.attempts = attempts
        self.delay = delay
        self.exponential_backoff = exponential_backoff

    def execute(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        for attempt in range(self.attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.attempts - 1:
                    raise e
                wait_time = self.delay * (2 ** attempt if self.exponential_backoff else 1)
                time.sleep(wait_time)

class DiskCache:
    """Simple disk-based cache with TTL"""
    def __init__(self, ttl: int = 3600, cache_dir: str = ".crawl4ai_cache"):
        self.ttl = ttl
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Get the cache file path for a key"""
        return os.path.join(self.cache_dir, f"{hash(key)}.json")

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None

        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Check if cache is expired
        cached_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cached_time > timedelta(seconds=self.ttl):
            os.remove(cache_path)
            return None

        return cache_data['value']

    def set(self, key: str, value: Any):
        """Set a value in cache"""
        cache_path = self._get_cache_path(key)
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'value': value
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
