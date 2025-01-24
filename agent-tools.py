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
from pydantic import BaseModel, Field

# ========== Configuration Constants ==========
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:85.0) Gecko/20100101 Firefox/85.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36"
]

TIMEOUT_SETTINGS = {
    "page_load": 30,
    "script": 10
}

HEADLESS_OPTIONS = [
    "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1920,1080",
    "--disable-search-engine-choice-screen", "--disable-blink-features=AutomationControlled"
]

SYSTEM_MESSAGE = """You are an intelligent text extraction assistant. Extract structured information into pure JSON format 
without commentary. Process the following text:"""

PROMPT_PAGINATION = """You are an assistant that extracts pagination elements from HTML content. 
Extract pagination URLs following numbered patterns. Return JSON with 'page_urls' array of full URLs."""

# ========== Core Models ==========
class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list)

# ========== Main Application ==========
class LLMAgentOrchestrator:
    def __init__(self, groq_api_key: str, google_api_key: str):
        self.groq = Groq(api_key=groq_api_key)
        genai.configure(api_key=google_api_key)
        self.models = {
            "GEMMA2_9B_IT": "gemma2-9b-it",
            "LLAMA_3_70B_VERSATILE": "llama-3-3-70b-versatile",
            "LLAMA_GUARD_3_8B": "llama-guard-3-8b",
            "MIXTRAL_8X7B": "mixtral-8x7b-32768",
            "GEMINI_FLASH": genai.GenerativeModel('gemini-2.0-flash-exp')
        }

        # Initialize all agents
        self.search = self.SearchAgent(self)
        self.data = self.DataAgent(self)
        self.task = self.TaskAgent(self)
        self.nlp = self.NLPAgent(self)
        self.code = self.CodeAgent(self)
        self.domain = self.DomainAgent(self)
        self.viz = self.VizAgent(self)
        self.memory = self.MemoryAgent(self)
        self.multimodal = self.MultiModalAgent(self)
        self.eval = self.EvalAgent(self)
        self.streamlit = self.StreamlitAgent(self)

    # ========== Agent Implementations ==========
    class SearchAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def web_search(self, query: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Search: {query}", tools=[genai.Tool.from_google_search()]
            ).text
            
        def knowledge_retrieval(self, query: str) -> List[str]:
            return self.parent._groq_call(
                "LLAMA_3_70B_VERSATILE",
                f"Retrieve knowledge about {query} from vector DB"
            )
            
        def manage_citations(self, text: str) -> Dict:
            return self.parent._groq_call(
                "LLAMA_3_70B_VERSATILE",
                f"Extract citations from:\n{text}"
            )

    class DataAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def clean_data(self, html: str, fields: List[str] = None) -> Dict:
            prompt = SYSTEM_MESSAGE
            if fields:
                prompt += f"\nExtract fields: {', '.join(fields)}"
            prompt += f"\n\n{html[:30000]}"
            return json.loads(self.parent._groq_call("MIXTRAL_8X7B", prompt))
            
        def find_pagination(self, html: str, url: str) -> List[str]:
            try:
                prompt = f"{PROMPT_PAGINATION}\nCurrent URL: {url}\nHTML Content:\n{html[:15000]}"
                response = self.parent.groq.chat.completions.create(
                    model="llama-3-70b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                content = response.choices[0].message.content
                return PaginationData(**json.loads(content[content.find('{'):])).page_urls
            except Exception as e:
                st.error(f"Pagination error: {str(e)}")
                return []

        def eda_analysis(self, data: str) -> Dict:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Perform EDA on:\n{data}"
            ).text
            
        def detect_anomalies(self, data: str) -> Dict:
            return self.parent._groq_call(
                "LLAMA_3_70B_VERSATILE",
                f"Find anomalies in:\n{data}"
            )

    class TaskAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def decompose_task(self, task: str) -> List[str]:
            return self.parent._groq_call(
                "LLAMA_3_70B_VERSATILE",
                f"Break down task: {task}"
            )
            
        def collaborate_agents(self, task: str) -> str:
            return self.parent._groq_call(
                "GEMMA2_9B_IT",
                f"Coordinate agents for: {task}"
            )

    class NLPAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def rewrite_query(self, query: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Optimize query: {query}"
            ).text
            
        def summarize(self, text: str) -> str:
            return self.parent._groq_call(
                "MIXTRAL_8X7B",
                f"Summarize:\n{text}"
            )

    class CodeAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def generate_api(self, spec: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Create API for: {spec}"
            ).text
            
        def refactor_code(self, code: str) -> str:
            return self.parent._groq_call(
                "LLAMA_3_70B_VERSATILE",
                f"Refactor:\n{code}"
            )

    class DomainAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def analyze_finance(self, data: Dict) -> Dict:
            return self.parent._groq_call(
                "MIXTRAL_8X7B",
                f"Analyze financial data:\n{data}"
            )
            
        def research_papers(self, query: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Find papers about {query}",
                tools=[genai.Tool.from_google_search()]
            ).text

    class VizAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def create_dashboard(self, data: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                f"Generate dashboard code for:\n{data}"
            ).text

    class MemoryAgent:
        def __init__(self, parent):
            self.parent = parent
            self.context = ""
            
        def update_context(self, text: str) -> None:
            self.context = self.parent._groq_call(
                "MIXTRAL_8X7B",
                f"Update context with:\n{text}"
            )

    class MultiModalAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def analyze_image(self, image_path: str) -> str:
            return self.parent.models["GEMINI_FLASH"].generate_content(
                genai.upload_file(image_path)
            ).text

    class EvalAgent:
        def __init__(self, parent):
            self.parent = parent
            
        def benchmark_performance(self, task: str) -> Dict:
            return self.parent._groq_call(
                "LLAMA_3_70B_VERSATILE",
                f"Evaluate performance on: {task}"
            )

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
                'processed_urls': set(),
                'scroll_count': 2
            }
            for key, val in defaults.items():
                st.session_state.setdefault(key, val)

        def setup_ui(self):
            st.set_page_config(page_title="AI Web Scraper", page_icon="🕷️", layout="wide")
            st.title("AI-Powered Web Scraper 🕷️")
            
            with st.sidebar:
                st.title("Configuration")
                self._api_key_inputs()
                self._url_input()
                self._field_selection()
                st.slider("Scroll Count", 1, 5, key='scroll_count')

        def _api_key_inputs(self):
            with st.expander("API Keys", expanded=False):
                st.session_state['groq_api_key'] = st.text_input("Groq Key", type="password")
                st.session_state['gemini_api_key'] = st.text_input("Gemini Key", type="password")

        def _url_input(self):
            url_input = st.text_input("Enter URL(s) separated by spaces")
            st.session_state['urls'] = [u.strip() for u in url_input.split() if u.strip()]
            
        def _field_selection(self):
            if st.toggle("Enable Field Extraction"):
                st.session_state['fields'] = st_tags(
                    label='Fields to Extract:',
                    text='Press enter to add',
                    maxtags=15,
                    key='fields_input'
                )

        def run_scraper(self):
            if st.sidebar.button("Start Scraping") and self._validate_inputs():
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
            st.session_state.update({
                'scraping_state': 'processing',
                'results': {'data': [], 'output_folder': self._generate_output_folder()}
            })

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
