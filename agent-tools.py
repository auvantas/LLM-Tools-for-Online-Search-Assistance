from typing import Dict, List, Any, Optional
import google.generativeai as genai
from groq import Groq
import streamlit as st
from crawl4ai import AsyncWebCrawler
import os
import json
import pandas as pd
from datetime import datetime
import re
from urllib.parse import urlparse
import random

# ========== Configuration Constants ==========
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    # ... (keep all existing user agents from the second script)
]

TIMEOUT_SETTINGS = {
    "page_load": 30,
    "script": 10
}

HEADLESS_OPTIONS = [
    "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1920,1080",
    "--disable-search-engine-choice-screen", "--disable-blink-features=AutomationControlled"
]

HEADLESS_OPTIONS_DOCKER = [
    "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
    "--disable-software-rasterizer", "--disable-setuid-sandbox", "--remote-debugging-port=9222",
    "--disable-search-engine-choice-screen"
]

NUMBER_SCROLL = 2

SYSTEM_MESSAGE = """You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
with no additional commentary, explanations, or extraneous information. 
You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
Please process the following text and provide the output in pure JSON format with no words before or after the JSON:"""

PROMPT_PAGINATION = """
You are an assistant that extracts pagination elements from markdown content of websites. Your goal is to act as a universal pagination scraper for URLs from all websites.

Please extract:
- A list of page URLs for pagination that follow a numbered pattern. 
- Generate subsequent URLs even if only a partial pattern exists.

Provide output as a JSON object with the following structure:
{
    "page_urls": ["url1", "url2", "url3",...,"urlN"]
}

Do not include any additional text or explanations.
Initial URL: {url}
Page content:
"""

# ========== Main Application ==========
class LLMAgentOrchestrator:
    def __init__(self, groq_api_key: str, google_api_key: str):
        self.groq = Groq(api_key=groq_api_key)
        genai.configure(api_key=google_api_key)
        self.models = {
            "GEMMA2_9B_IT": "gemma2-9b-it",
            "LLAMA_3_70B_VERSATILE": "llama-3.3-70b-versatile",
            "LLAMA_GUARD_3_8B": "llama-guard-3-8b",
            "MIXTRAL_8X7B": "mixtral-8x7b-32768",
            "GEMINI_FLASH": genai.GenerativeModel('gemini-2.0-flash-exp')
        }
        
        self.search = self.SearchAgent(self)
        self.data = self.DataAgent(self)
        self.streamlit = self.StreamlitAgent(self)

    class SearchAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def web_search(self, query: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Search: {query}", tools=[genai.Tool.from_google_search()]
            ).text

    class DataAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def clean_data(self, html: str, fields: List[str] = None) -> Dict:
            """Enhanced data cleaning with field-specific extraction"""
            prompt = SYSTEM_MESSAGE
            if fields:
                prompt += f"\nExtract the following fields: {', '.join(fields)}."
            prompt += f"\n\nPage content:\n{html}"
            
            return json.loads(self.parent._groq_call("MIXTRAL_8X7B", prompt))

        def find_pagination(self, html: str, url: str) -> List[str]:
            """Find additional pages using LLM-powered analysis"""
            prompt = PROMPT_PAGINATION.format(url=url) + html[:5000]  # Limit content size
            try:
                result = json.loads(self.parent._groq_call("MIXTRAL_8X7B", prompt))
                return result.get('page_urls', [])
            except json.JSONDecodeError:
                return []

    class StreamlitAgent:
        def __init__(self, parent):
            self.parent = parent
            self._init_session_state()
            
        def _init_session_state(self):
            defaults = {
                'scraping_state': 'idle',
                'results': None,
                'urls': [],
                'fields': [],
                'model_selection': 'gpt-4-turbo',
                'processed_urls': set()
            }
            for key, val in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = val

        def setup_ui(self):
            st.set_page_config(page_title="AI Web Scraper", page_icon="🕷️", layout="wide")
            st.title("AI-Powered Web Scraper 🕷️")
            
            with st.sidebar:
                st.title("Configuration")
                self._api_key_inputs()
                self._url_input()
                self._field_selection()
                st.slider("Scroll Count", 1-5, value=NUMBER_SCROLL, key='scroll_count')

        def _api_key_inputs(self):
            with st.expander("API Keys", expanded=False):
                st.session_state['openai_api_key'] = st.text_input("OpenAI Key", type="password")
                st.session_state['gemini_api_key'] = st.text_input("Gemini Key", type="password")
                st.session_state['groq_api_key'] = st.text_input("Groq Key", type="password")

        def _url_input(self):
            url_input = st.text_input("Enter URL(s) separated by space")
            st.session_state['urls'] = url_input.strip().split()
            
        def _field_selection(self):
            if st.toggle("Enable Field Extraction"):
                st.session_state['fields'] = st_tags(
                    label='Fields to Extract:',
                    text='Press enter to add',
                    maxtags=-1,
                    key='fields_input'
                )

        def run_scraper(self):
            if st.sidebar.button("Start Scraping"):
                if not self._validate_inputs():
                    return
                
                st.session_state['scraping_state'] = 'processing'
                self._execute_scraping()

            if st.session_state['scraping_state'] == 'processing':
                with st.spinner("Scraping in progress..."):
                    self._process_urls()
                    st.session_state['scraping_state'] = 'completed'
                    st.rerun()

            if st.session_state['scraping_state'] == 'completed':
                self._display_results()

        def _validate_inputs(self):
            if not st.session_state['urls']:
                st.error("Please enter at least one URL")
                return False
            return True

        def _execute_scraping(self):
            self.parent.search.web_search("Scraping initialization")
            st.session_state['results'] = {
                'data': [],
                'output_folder': self._generate_output_folder()
            }

        def _process_urls(self):
            crawler = AsyncWebCrawler(
                strategy="comprehensive",
                user_agent=random.choice(USER_AGENTS),
                timeout=TIMEOUT_SETTINGS['page_load'],
                scroll_count=st.session_state['scroll_count'],
                browser_options=HEADLESS_OPTIONS
            )

            url_queue = st.session_state['urls'].copy()
            while url_queue:
                url = url_queue.pop(0)
                if url in st.session_state['processed_urls']:
                    continue
                
                try:
                    result = crawler.crawl(url)
                    cleaned = self.parent.data.clean_data(result.raw_html, st.session_state['fields'])
                    st.session_state['results']['data'].append(cleaned)
                    
                    # Find and queue pagination URLs
                    new_urls = self.parent.data.find_pagination(result.raw_html, url)
                    url_queue.extend([u for u in new_urls if u not in st.session_state['processed_urls']])
                    
                    st.session_state['processed_urls'].add(url)
                except Exception as e:
                    st.error(f"Error processing {url}: {str(e)}")

        def _display_results(self):
            st.subheader("Scraping Results")
            for idx, data in enumerate(st.session_state['results']['data']):
                with st.expander(f"Result {idx+1}"):
                    st.json(data)
            
            if st.session_state['results']['data']:
                st.download_button(
                    "Download All Data",
                    data=json.dumps(st.session_state['results']['data'], indent=2),
                    file_name=f"{st.session_state['results']['output_folder']}.json"
                )

        def _generate_output_folder(self):
            domain = re.sub(r'^www\.', '', urlparse(st.session_state['urls'][0]).netloc)
            clean_domain = re.sub(r'\W+', '_', domain)
            return f"{clean_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _groq_call(self, model: str, prompt: str) -> Any:
        return self.groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.models[model],
            temperature=0.7,
            max_tokens=4000
        ).choices[0].message.content

if __name__ == "__main__":
    agent = LLMAgentOrchestrator(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    agent.streamlit.setup_ui()
    agent.streamlit.run_scraper()
