"""Extraction components for crawl4ai_lite"""
from typing import Any, Dict, List, Optional, Callable
import re
from bs4 import BeautifulSoup

class Wrapper:
    """Simple wrapper for HTML content extraction"""
    def __init__(self):
        self.soup = None

    def parse(self, html: str):
        """Parse HTML content"""
        self.soup = BeautifulSoup(html, 'html.parser')
        return self

    def extract_text(self) -> str:
        """Extract text content from HTML"""
        if not self.soup:
            return ""
        return self.soup.get_text(separator=' ', strip=True)

    def extract_links(self) -> List[str]:
        """Extract links from HTML"""
        if not self.soup:
            return []
        return [a.get('href', '') for a in self.soup.find_all('a', href=True)]

class SemanticExtractor:
    """Extract semantic information from text"""
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\+?1?\d{9,15}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'date': r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b'
        }

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract semantic information from text"""
        results = {}
        for key, pattern in self.patterns.items():
            results[key] = re.findall(pattern, text)
        return results

class LLMExtractor:
    """Extract information using LLM"""
    def __init__(self, llm_api: Callable[[str], str]):
        self.llm_api = llm_api

    def extract(self, text: str, instruction: str) -> str:
        """Extract information using LLM based on instruction"""
        prompt = f"{instruction}\n\nText: {text}\n\nExtracted information:"
        return self.llm_api(prompt)
