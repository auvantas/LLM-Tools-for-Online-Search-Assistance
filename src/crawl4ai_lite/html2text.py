"""HTML to text conversion for crawl4ai_lite"""
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

class HTML2Text:
    """Convert HTML to structured text"""
    def __init__(self, parse_lists: bool = True, parse_links: bool = True, parse_images: bool = True):
        self.parse_lists = parse_lists
        self.parse_links = parse_links
        self.parse_images = parse_images
        self.soup = None

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace"""
        return ' '.join(text.split())

    def _process_lists(self) -> List[str]:
        """Process HTML lists"""
        items = []
        if not self.soup:
            return items

        for list_tag in self.soup.find_all(['ul', 'ol']):
            for item in list_tag.find_all('li'):
                items.append(self._clean_text(item.get_text()))
        return items

    def _process_links(self) -> List[Dict[str, str]]:
        """Process HTML links"""
        links = []
        if not self.soup:
            return links

        for a in self.soup.find_all('a', href=True):
            links.append({
                'text': self._clean_text(a.get_text()),
                'href': a['href']
            })
        return links

    def _process_images(self) -> List[Dict[str, str]]:
        """Process HTML images"""
        images = []
        if not self.soup:
            return images

        for img in self.soup.find_all('img', src=True):
            images.append({
                'alt': img.get('alt', ''),
                'src': img['src']
            })
        return images

    def convert(self, html: str) -> Dict:
        """Convert HTML to structured text"""
        self.soup = BeautifulSoup(html, 'html.parser')
        result = {
            'text': self._clean_text(self.soup.get_text())
        }

        if self.parse_lists:
            result['lists'] = self._process_lists()
        if self.parse_links:
            result['links'] = self._process_links()
        if self.parse_images:
            result['images'] = self._process_images()

        return result
