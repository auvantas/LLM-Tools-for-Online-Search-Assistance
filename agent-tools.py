from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai
from groq import Groq
import streamlit as st
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction import LLMExtractor, SemanticExtractor, Wrapper
from crawl4ai.html2text import HTML2Text
from crawl4ai.core import RetryPolicy, DiskCache
import os
import json
import pandas as pd
from datetime import datetime
import re
from urllib.parse import urlparse
import random
import plotly.express as px
from pydantic import BaseModel, Field
from streamlit_tags import st_tags

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
    "script": 10,
    "retry": 15  # Added from crawl4ai docs
}

HEADLESS_OPTIONS = [
    "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1920,1080",
    "--disable-search-engine-choice-screen", "--disable-blink-features=AutomationControlled",
    "--no-sandbox"  # Added for cloud compatibility
]

STRATEGIES = {
    "cosmic": "Full JS rendering with smart detection bypass",
    "stealth": "Headless browsing with anti-detection",
    "lightweight": "Basic HTML fetching",
    "custom": "User-defined strategy"
}

SYSTEM_MESSAGE = """You are an intelligent text extraction assistant. Extract structured information into pure JSON format 
without commentary. Process the following text:"""

PROMPT_PAGINATION = """You are an assistant that extracts pagination elements from HTML content. 
Extract pagination URLs following numbered patterns. Return JSON with 'page_urls' array of full URLs."""

# ========== Core Models ==========
class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list)
    next_page: Optional[str] = None
    page_pattern: Optional[str] = None

class EnhancedResult(BaseModel):
    content: Union[str, Dict]
    metadata: Dict = Field(default_factory=dict)
    statistics: Dict = Field(default_factory=dict)
    screenshots: List[str] = Field(default_factory=list)

# ========== Main Application with Enhancements ==========
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

        # Initialize enhanced components
        self.html_converter = HTML2Text(parse_lists=True, parse_links=True, parse_images=True)
        self.cache = DiskCache(ttl=3600, cache_dir=".crawl4ai_cache")
        
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
            cache_key = f"search_{hash(query)}"
            if cached := self.parent.cache.get(cache_key):
                return cached
                
            result = self.parent.models["GEMINI_FLASH"].generate_content(
                f"Search: {query}", tools=[genai.Tool.from_google_search()]
            ).text
            
            self.parent.cache.set(cache_key, result)
            return result
            
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
            self.wrapper = Wrapper()
            self.llm_extractor = LLMExtractor(llm_api=self._enhanced_groq_call)
            self.semantic_extractor = SemanticExtractor()

        def _enhanced_groq_call(self, prompt: str) -> Any:
            """Add caching and retry mechanism"""
            cache_key = f"groq_{hash(prompt)}"
            if cached := self.parent.cache.get(cache_key):
                return cached
            
            result = self.parent.groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.parent.models["MIXTRAL_8X7B"],
                temperature=0.7,
                max_tokens=4000
            ).choices[0].message.content
            
            self.parent.cache.set(cache_key, result)
            return result

        def clean_data(self, html: str, url: str, fields: List[str] = None) -> EnhancedResult:
            """Enhanced with crawl4ai's hybrid extraction"""
            try:
                # Convert HTML to structured text
                structured_html = self.parent.html_converter.convert(html)
                
                # Wrap content based on type
                wrapped = self.wrapper.wrap(
                    html=structured_html,
                    url=url,
                    extraction_type="article" if not fields else "custom"
                )

                # Extract using semantic rules
                base_content = self.semantic_extractor.extract(wrapped)

                # LLM extraction if fields specified
                if fields:
                    llm_content = self.llm_extractor.extract(
                        content=wrapped,
                        schema={field: "string" for field in fields},
                        model="gpt-4"
                    )
                    base_content.update(llm_content)

                return EnhancedResult(
                    content=base_content,
                    metadata=wrapped.metadata,
                    statistics={
                        "html_length": len(html),
                        "text_length": len(structured_html),
                        "elements_extracted": len(base_content)
                    }
                )
            except Exception as e:
                st.error(f"Extraction error: {str(e)}")
                return EnhancedResult(content={})

        def find_pagination(self, html: str, url: str) -> PaginationData:
            """Enhanced with crawl4ai's pattern detection"""
            try:
                result = self.semantic_extractor.extract_pagination(
                    html=html,
                    base_url=url,
                    pattern=r"page=\d+",
                    strategy="relaxed"
                )
                return PaginationData(**result)
            except Exception as e:
                st.error(f"Pagination error: {str(e)}")
                return PaginationData()

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
            self.retry_policy = RetryPolicy(
                attempts=3,
                delay=10,
                exponential_backoff=True
            )

        def _init_session_state(self):
            defaults = {
                'scraping_state': 'idle',
                'results': None,
                'urls': [],
                'fields': [],
                'processed_urls': set(),
                'scroll_count': 2,
                'strategy': 'cosmic',
                'proxy': None,
                'js_strategy': 'auto_scroll',
                'screenshots': False,
                'max_depth': 1,
                'error_log': []
            }
            for key, val in defaults.items():
                st.session_state.setdefault(key, val)

        def setup_ui(self):
            st.set_page_config(page_title="AI Web Scraper Pro", page_icon="🕷️", layout="wide")
            st.title("AI-Powered Web Scraper Pro 🕷️")
            
            with st.sidebar:
                self._main_config()
                self._advanced_config()
                self._js_config()
            
            self._results_dashboard()

        def _main_config(self):
            st.title("Configuration")
            with st.expander("API Keys", expanded=True):
                st.session_state['groq_api_key'] = st.text_input("Groq Key", type="password")
                st.session_state['gemini_api_key'] = st.text_input("Gemini Key", type="password")
            
            with st.expander("Core Settings"):
                url_input = st.text_input("Enter URL(s) separated by spaces")
                st.session_state['urls'] = [u.strip() for u in url_input.split() if u.strip()]
                
                st.selectbox(
                    "Crawling Strategy",
                    options=list(STRATEGIES.keys()),
                    index=0,
                    key='strategy',
                    help=STRATEGIES[st.session_state.strategy]
                )
                
                st.slider("Scroll Count", 1, 5, key='scroll_count')
                st.checkbox("Enable Screenshots", key='screenshots')

        def _advanced_config(self):
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Proxy Server", key='proxy')
                    st.number_input("Max Depth", 1, 5, key='max_depth')
                with col2:
                    st.checkbox("Enable Caching", True, key='caching')
                    st.checkbox("Auto-Retry Failures", True, key='auto_retry')

        def _js_config(self):
            with st.expander("JavaScript Execution"):
                st.selectbox(
                    "JS Strategy",
                    options=["auto_scroll", "click_interactive", "custom"],
                    key='js_strategy'
                )
                if st.session_state.js_strategy == "custom":
                    st.text_area("Custom JS Code", key='custom_js', height=150)

        def _results_dashboard(self):
            if st.session_state['scraping_state'] == 'completed':
                tab1, tab2, tab3 = st.tabs(["Data", "Statistics", "Logs"])
                
                with tab1:
                    self._display_results()
                    
                with tab2:
                    self._show_statistics()
                    
                with tab3:
                    self._show_error_log()

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

        def _show_statistics(self):
            stats = {
                "Total URLs Processed": len(st.session_state['processed_urls']),
                "Success Rate": f"{len(st.session_state['results']['data'])/len(st.session_state['urls'])*100:.1f}%",
                "Total Data Size": f"{sum(len(str(d)) for d in st.session_state['results']['data']) / 1024:.2f} KB"
            }
            st.json(stats)
            
        def _show_error_log(self):
            if st.session_state['error_log']:
                st.write("### Error Log")
                for error in st.session_state['error_log']:
                    st.error(f"**{error['url']}**: {error['message']}")
            else:
                st.success("No errors recorded")

        def run_scraper(self):
            if st.sidebar.button("Start Scraping") and self._validate_inputs():
                self._execute_scraping()

            if st.session_state['scraping_state'] == 'processing':
                with st.spinner("Scraping in progress..."):
                    self._process_urls()
                    st.session_state['scraping_state'] = 'completed'
                    st.rerun()

        def _process_urls(self):
            crawler = AsyncWebCrawler(
                strategy=st.session_state.strategy,
                user_agent=random.choice(USER_AGENTS),
                timeout=TIMEOUT_SETTINGS,
                scroll_count=st.session_state['scroll_count'],
                browser_options=HEADLESS_OPTIONS,
                execute_js=True,
                extract_metadata=True,
                proxy=st.session_state.proxy,
                cache=self.parent.cache if st.session_state.caching else None,
                retry_policy=self.retry_policy,
                on_error=self._handle_error,
                js_snippet=st.session_state.js_strategy if st.session_state.js_strategy != "custom" else None,
                custom_js=st.session_state.custom_js if st.session_state.js_strategy == "custom" else None
            )

            url_queue = [(url, 0) for url in st.session_state['urls']]  # (url, depth)
            while url_queue:
                url, depth = url_queue.pop(0)
                if url in st.session_state['processed_urls'] or depth > st.session_state.max_depth:
                    continue
                
                try:
                    if cached := self.parent.cache.get(url):
                        result = cached
                    else:
                        result = crawler.crawl(url)
                        if st.session_state.screenshots:
                            result.capture_screenshot()
                        self.parent.cache.set(url, result)
                    
                    cleaned = self.parent.data.clean_data(result.raw_html, url, st.session_state['fields'])
                    pagination = self.parent.data.find_pagination(result.raw_html, url)
                    
                    st.session_state['results']['data'].append(cleaned.content)
                    st.session_state['processed_urls'].add(url)
                    
                    # Add new URLs to queue
                    new_urls = [(u, depth+1) for u in pagination.page_urls 
                              if u not in st.session_state['processed_urls']]
                    url_queue.extend(new_urls)

                except Exception as e:
                    self._handle_error(url, str(e))
                    if st.session_state.auto_retry:
                        url_queue.append((url, depth))  # Retry

        def _handle_error(self, url: str, error: str):
            st.session_state['error_log'].append({
                "url": url,
                "message": error,
                "timestamp": datetime.now().isoformat()
            })

        def _generate_output_folder(self):
            # Enhanced naming convention
            domain = re.sub(r'^www\.', '', urlparse(st.session_state['urls'][0]).netloc)
            clean_domain = re.sub(r'\W+', '_', domain)
            return f"{clean_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v2"

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
