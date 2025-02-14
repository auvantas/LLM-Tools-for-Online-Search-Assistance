"""Async web crawler for crawl4ai_lite"""
import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
import random
from .core import RetryPolicy, DiskCache

class AsyncWebCrawler:
    """Asynchronous web crawler with caching and retry logic"""
    def __init__(self, 
                 max_concurrent: int = 5,
                 timeout: int = 30,
                 cache_ttl: int = 3600,
                 retry_attempts: int = 3):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.cache = DiskCache(ttl=cache_ttl)
        self.retry_policy = RetryPolicy(attempts=retry_attempts)
        self.semaphore = None
        self.session = None
        
        # Common user agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        ]

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': random.choice(self.user_agents)}
            )
        return self.session

    async def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch content from a URL with caching"""
        # Check cache first
        cached_result = self.cache.get(url)
        if cached_result:
            return cached_result

        async with self.semaphore:
            session = await self._get_session()
            try:
                async with session.get(url) as response:
                    content = await response.text()
                    result = {
                        'url': url,
                        'status': response.status,
                        'content': content,
                        'headers': dict(response.headers)
                    }
                    # Cache successful results
                    if response.status == 200:
                        self.cache.set(url, result)
                    return result
            except Exception as e:
                return {
                    'url': url,
                    'status': 0,
                    'error': str(e),
                    'content': None,
                    'headers': {}
                }

    async def crawl(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Crawl multiple URLs concurrently"""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def crawl_sync(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for crawl method"""
        loop = asyncio.get_event_loop()
        try:
            results = loop.run_until_complete(self.crawl(urls))
            loop.run_until_complete(self.close())
            return results
        except Exception as e:
            if self.session and not self.session.closed:
                loop.run_until_complete(self.close())
            raise e
