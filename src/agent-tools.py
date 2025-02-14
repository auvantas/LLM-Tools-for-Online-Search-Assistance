from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai
from groq import Groq
import streamlit as st
from crawl4ai_lite import AsyncWebCrawler
from crawl4ai_lite.extraction import LLMExtractor, SemanticExtractor, Wrapper
from crawl4ai_lite.html2text import HTML2Text
from crawl4ai_lite.core import RetryPolicy, DiskCache
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
from bs4 import BeautifulSoup
import argparse
import io
import numpy as np
import plotly.graph_objects as go

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

HEADLESS_OPTIONS = {
    "headless": True,
    "disable-gpu": True,
    "no-sandbox": True,
    "disable-dev-shm-usage": True
}

STRATEGIES = {
    "cosmic": "Advanced AI-powered extraction with pattern recognition",
    "semantic": "Content-aware extraction using NLP",
    "structured": "Rule-based extraction for well-structured pages",
    "dynamic": "JavaScript-enabled extraction for dynamic content"
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
        self.web_scraper = self.WebScraperAgent(self)

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

        def export_results(self, data: Union[Dict, List], format: str, filename: str) -> str:
            """Export results to various formats (CSV, JSON, MD, TXT)
            
            Args:
                data: The data to export
                format: Format to export to ('csv', 'json', 'md', 'txt')
                filename: Name of the output file (without extension)
                
            Returns:
                str: Path to the exported file
            """
            os.makedirs('exports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}"
            
            if format.lower() == 'json':
                output_path = f'exports/{filename}.json'
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'csv':
                output_path = f'exports/{filename}.csv'
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
                df.to_csv(output_path, index=False)
                
            elif format.lower() == 'md':
                output_path = f'exports/{filename}.md'
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, dict):
                        f.write('# Results\n\n')
                        for key, value in data.items():
                            f.write(f'## {key}\n{value}\n\n')
                    elif isinstance(data, list):
                        f.write('# Results\n\n')
                        for item in data:
                            if isinstance(item, dict):
                                for key, value in item.items():
                                    f.write(f'- **{key}**: {value}\n')
                            else:
                                f.write(f'- {item}\n')
                                
            elif format.lower() == 'txt':
                output_path = f'exports/{filename}.txt'
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f'{key}: {value}\n')
                    elif isinstance(data, list):
                        for item in data:
                            f.write(f'{str(item)}\n')
                            
            return output_path

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
            self.templates_dir = "c:/Users/samplex/Documents/Agents/Prompts"
            self.template_cache = {}
            self.load_templates()
            self._init_session_state()

        def load_templates(self):
            """Load all template files from the templates directory"""
            try:
                for file in os.listdir(self.templates_dir):
                    if file.endswith('.txt'):
                        with open(os.path.join(self.templates_dir, file), 'r') as f:
                            self.template_cache[file] = f.read()
            except Exception as e:
                st.error(f"Error loading templates: {str(e)}")

        def _init_session_state(self):
            defaults = {
                'api_keys': {},
                'sources': [],
                'custom_files': [],
                'results': None,
                'errors': [],
                'prompt': '',
                'cli_input': ''
            }
            for key, val in defaults.items():
                st.session_state.setdefault(key, val)

        def setup_ui(self):
            st.set_page_config(page_title="AI Web Scraper Pro", layout="wide")
            
            # Sidebar for configuration
            with st.sidebar:
                st.title("Configuration")
                
                # API Keys Section
                st.header("API Keys")
                groq_key = st.text_input("Groq API Key", type="password")
                google_key = st.text_input("Google API Key", type="password")
                
                # Source Integration
                st.header("Custom Sources")
                source_type = st.selectbox("Add Source", 
                    ["Website", "PDF", "Text", "Markdown"])
                
                if source_type == "Website":
                    url = st.text_input("Enter URL")
                    if url and st.button("Add Website"):
                        st.session_state.sources.append({"type": "website", "path": url})
                else:
                    uploaded_file = st.file_uploader(f"Upload {source_type} file", 
                        type=source_type.lower())
                    if uploaded_file:
                        st.session_state.custom_files.append({
                            "type": source_type.lower(),
                            "name": uploaded_file.name,
                            "content": uploaded_file.read()
                        })
                
                # Display Added Sources
                if st.session_state.sources or st.session_state.custom_files:
                    st.header("Added Sources")
                    for src in st.session_state.sources:
                        st.text(f"🌐 {src['path']}")
                    for file in st.session_state.custom_files:
                        st.text(f"📄 {file['name']}")
                    
                    if st.button("Clear Sources"):
                        st.session_state.sources = []
                        st.session_state.custom_files = []

            # Main content area
            st.title("AI Web Scraper Pro")
            
            # Dual Input Mode
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Natural Language Query")
                prompt = st.text_area("Enter your research query", 
                    value=st.session_state.prompt,
                    height=100)
                
            with col2:
                st.subheader("CLI Mode")
                cli_input = st.text_area("Enter CLI commands", 
                    value=st.session_state.cli_input,
                    height=100)
            
            # Process Button
            if st.button("Process"):
                combined_input = {
                    "prompt": prompt if prompt else None,
                    "cli": cli_input if cli_input else None,
                    "sources": st.session_state.sources,
                    "files": st.session_state.custom_files
                }
                self.process_input(combined_input)

            # Display results
            self.display_results()

        def display_results(self):
            """Display processed results in a structured format"""
            if not st.session_state.results:
                return

            # Report style selector
            report_styles = ["Executive Summary", "Technical Analysis", "Market Research", "Trend Report", "Custom"]
            selected_style = st.selectbox("Select Report Style", report_styles)
            
            # Export format selector
            export_formats = ["CSV", "Excel", "JSON", "PDF", "Markdown"]
            col1, col2 = st.columns([3, 1])
            with col1:
                st.header("Results")
            with col2:
                export_format = st.selectbox("Export Format", export_formats)
                if st.button("Export Data"):
                    self._export_data(export_format)

            # Create tabs for different views
            tabs = st.tabs(["Overview", "Sources", "Analysis", "Visualizations", "Raw Data"])
            
            with tabs[0]:
                self._display_overview(selected_style)
            with tabs[1]:
                self._display_sources()
            with tabs[2]:
                self._display_analysis()
            with tabs[3]:
                self._display_visualizations()
            with tabs[4]:
                self._display_raw_data()

        def _display_overview(self, report_style):
            """Display overview of results with customizable report style"""
            results = st.session_state.results
            
            # Summary metrics with enhanced styling
            st.markdown(f"### {report_style} Dashboard")
            metrics_container = st.container()
            col1, col2, col3, col4 = metrics_container.columns(4)
            
            with col1:
                st.metric("Sources", 
                    len(results.get("sources_processed", [])),
                    delta=None,
                    help="Total number of processed sources")
            with col2:
                success_rate = self._calculate_success_rate(results)
                st.metric("Success Rate", 
                    f"{success_rate}%",
                    delta=None,
                    help="Percentage of successful source processing")
            with col3:
                st.metric("Data Points", 
                    len(results.get("structured_data", {})),
                    delta=None,
                    help="Total number of extracted data points")
            with col4:
                processing_time = results.get("processing_time", 0)
                st.metric("Processing Time", 
                    f"{processing_time:.2f}s",
                    delta=None,
                    help="Total time taken to process all sources")

            # Quick insights with filtering
            if "prompt_analysis" in results:
                st.subheader("Query Analysis")
                with st.expander("View Details", expanded=True):
                    self._display_filtered_insights(results["prompt_analysis"])

        def _display_sources(self):
            """Display processed source information with advanced filtering"""
            results = st.session_state.results
            
            # Source type filter
            source_types = list(set(s["type"] for s in results.get("sources_processed", [])))
            selected_types = st.multiselect("Filter by Source Type", source_types, default=source_types)
            
            # Search filter
            search_term = st.text_input("Search Sources", "")
            
            filtered_sources = [
                source for source in results.get("sources_processed", [])
                if source["type"] in selected_types and
                (not search_term or search_term.lower() in str(source).lower())
            ]
            
            for source in filtered_sources:
                with st.expander(f"{source['type'].upper()}: {source.get('name', source.get('path', 'Unknown'))}"):
                    # Source metadata
                    st.markdown("#### Metadata")
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.write("Type:", source["type"])
                    with meta_cols[1]:
                        st.write("Size:", self._format_size(source.get("size", 0)))
                    with meta_cols[2]:
                        st.write("Last Modified:", source.get("last_modified", "Unknown"))
                    
                    # Source content with syntax highlighting
                    st.markdown("#### Content")
                    st.code(json.dumps(source["data"], indent=2), language="json")
            
            if results.get("errors"):
                with st.expander("Processing Errors", expanded=False):
                    for error in results["errors"]:
                        st.error(error)

        def _display_analysis(self):
            """Display analytical insights with interactive tables"""
            results = st.session_state.results
            
            if "structured_data" in results:
                data = results["structured_data"]
                
                for section in ["companies", "technologies", "trends"]:
                    if section in data:
                        st.subheader(section.title())
                        df = pd.DataFrame(data[section])
                        
                        # Add search and filter capabilities
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            search = st.text_input(f"Search {section}", key=f"search_{section}")
                        with col2:
                            sort_by = st.selectbox(
                                "Sort by",
                                df.columns.tolist(),
                                key=f"sort_{section}"
                            )
                        
                        # Filter and sort the dataframe
                        if search:
                            mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                            df = df[mask]
                        
                        df = df.sort_values(by=sort_by, ascending=False)
                        
                        # Display interactive table
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download button for each section
                        csv = df.to_csv(index=False)
                        st.download_button(
                            f"Download {section.title()} Data",
                            csv,
                            f"{section}_data.csv",
                            "text/csv",
                            key=f'download_{section}'
                        )

        def _display_visualizations(self):
            """Display interactive visualizations of the data"""
            results = st.session_state.results
            
            if "structured_data" not in results:
                st.info("No data available for visualization")
                return
            
            data = results["structured_data"]
            
            # Select visualization type
            viz_type = st.selectbox(
                "Select Visualization",
                ["Trend Analysis", "Technology Distribution", "Company Analysis"]
            )
            
            if viz_type == "Trend Analysis":
                self._plot_trend_analysis(data)
            elif viz_type == "Technology Distribution":
                self._plot_technology_distribution(data)
            elif viz_type == "Company Analysis":
                self._plot_company_analysis(data)

        def _plot_trend_analysis(self, data):
            """Plot trend analysis visualization"""
            if "trends" not in data:
                st.info("No trend data available")
                return
            
            df = pd.DataFrame(data["trends"])
            
            # Time series plot
            fig = go.Figure()
            
            for column in df.select_dtypes(include=[np.number]).columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[column],
                        name=column,
                        mode='lines+markers'
                    )
                )
                
            fig.update_layout(
                title="Trend Analysis Over Time",
                xaxis_title="Time Period",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

        def _plot_technology_distribution(self, data):
            """Plot technology distribution visualization"""
            if "technologies" not in data:
                st.info("No technology data available")
                return
            
            df = pd.DataFrame(data["technologies"])
            
            # Create a treemap
            fig = px.treemap(
                df,
                path=['category', 'name'],
                values='mentions',
                title="Technology Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        def _plot_company_analysis(self, data):
            """Plot company analysis visualization"""
            if "companies" not in data:
                st.info("No company data available")
                return
            
            df = pd.DataFrame(data["companies"])
            
            # Create a scatter plot
            fig = px.scatter(
                df,
                x='market_cap',
                y='revenue',
                size='employees',
                color='industry',
                hover_name='name',
                title="Company Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        def _export_data(self, format):
            """Export data in various formats"""
            results = st.session_state.results
            
            if format == "CSV":
                for section, data in results.get("structured_data", {}).items():
                    df = pd.DataFrame(data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        f"Download {section.title()}",
                        csv,
                        f"{section}_data.csv",
                        "text/csv"
                    )
            elif format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    for section, data in results.get("structured_data", {}).items():
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=section, index=False)
                
                st.download_button(
                    "Download Excel Report",
                    buffer.getvalue(),
                    "report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif format == "JSON":
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    "results.json",
                    "application/json"
                )
            elif format == "PDF":
                # Generate PDF report
                pdf_buffer = self._generate_pdf_report(results)
                st.download_button(
                    "Download PDF Report",
                    pdf_buffer.getvalue(),
                    "report.pdf",
                    "application/pdf"
                )
            elif format == "Markdown":
                markdown = self._generate_markdown_report(results)
                st.download_button(
                    "Download Markdown Report",
                    markdown,
                    "report.md",
                    "text/markdown"
                )

        def _generate_pdf_report(self, results):
            """Generate a PDF report"""
            buffer = io.BytesIO()
            # Add PDF generation logic here
            return buffer

        def _generate_markdown_report(self, results):
            """Generate a markdown report"""
            markdown = f"# Analysis Report\n\n"
            markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add sections
            for section, data in results.get("structured_data", {}).items():
                markdown += f"## {section.title()}\n\n"
                df = pd.DataFrame(data)
                markdown += df.to_markdown(index=False)
                markdown += "\n\n"
            
            return markdown

        def _calculate_success_rate(self, results):
            """Calculate the success rate of source processing"""
            total = len(results.get("sources_processed", [])) + len(results.get("errors", []))
            if total == 0:
                return 100
            return round((len(results.get("sources_processed", [])) / total) * 100, 1)

        def _format_size(self, size_bytes):
            """Format file size in human readable format"""
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.1f} TB"

        def _display_filtered_insights(self, insights):
            """Display filtered insights with expandable sections"""
            for category, items in insights.items():
                with st.expander(category.title()):
                    if isinstance(items, dict):
                        for key, value in items.items():
                            st.write(f"**{key}:** {value}")
                    elif isinstance(items, list):
                        for item in items:
                            st.write(f"- {item}")
                    else:
                        st.write(items)

        def process_input(self, combined_input):
            """Process both GUI and CLI inputs with custom sources"""
            try:
                # Initialize results container
                results = {
                    "structured_data": {},
                    "sources_processed": [],
                    "errors": []
                }
                
                # Process custom sources first
                for source in combined_input["sources"]:
                    try:
                        if source["type"] == "website":
                            result = self.parent.web_scraper.scrape(source["path"])
                            results["sources_processed"].append({
                                "type": "website",
                                "path": source["path"],
                                "data": result
                            })
                    except Exception as e:
                        results["errors"].append(f"Error processing {source['path']}: {str(e)}")
                
                # Process uploaded files
                for file in combined_input["files"]:
                    try:
                        result = self.process_file(file)
                        results["sources_processed"].append({
                            "type": file["type"],
                            "name": file["name"],
                            "data": result
                        })
                    except Exception as e:
                        results["errors"].append(f"Error processing {file['name']}: {str(e)}")
                
                # Process natural language prompt
                if combined_input["prompt"]:
                    results["prompt_analysis"] = self.parent.nlp.analyze_prompt(
                        combined_input["prompt"])
                
                # Process CLI commands
                if combined_input["cli"]:
                    results["cli_output"] = self.parent.task.execute_commands(
                        combined_input["cli"])
                
                # Store results in session state
                st.session_state.results = results
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

        def process_file(self, file):
            """Process uploaded files based on type"""
            if file["type"] == "pdf":
                return self._process_pdf(file)
            elif file["type"] == "text":
                return self._process_text(file)
            elif file["type"] == "markdown":
                return self._process_markdown(file)
            else:
                raise ValueError(f"Unsupported file type: {file['type']}")

        def _process_pdf(self, file):
            """Extract and process PDF content"""
            # Add PDF processing logic here
            return {"content": "PDF processing not implemented yet"}

        def _process_text(self, file):
            """Process plain text content"""
            content = file["content"].decode("utf-8")
            return {"content": content}

        def _process_markdown(self, file):
            """Process markdown content"""
            content = file["content"].decode("utf-8")
            return {"content": content}

    def _groq_call(self, model: str, prompt: str) -> Any:
        return self.groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.models[model],
            temperature=0.7,
            max_tokens=4000
        ).choices[0].message.content

    class WebScraperAgent:
        def __init__(self, parent):
            self.parent = parent

        def handle_pagination(self, html: str, base_url: str) -> List[str]:
            """Handle dynamic pagination and return all page URLs"""
            soup = BeautifulSoup(html, 'html.parser')
            page_urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'page' in href:
                    full_url = urlparse(base_url)._replace(path=href).geturl()
                    page_urls.append(full_url)
            return page_urls

        def capture_screenshot(self, url: str, output_path: str) -> None:
            """Capture a screenshot of the webpage"""
            from selenium import webdriver
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            driver.save_screenshot(output_path)
            driver.quit()

        def find_online_sources(self, query, max_results=5):
            """Find relevant sources with multiple fallback methods"""
            try:
                from googlesearch import search
                return list(search(query, num=max_results, stop=max_results, pause=2))
            except ImportError:
                self.logger.warning('Using alternative search method')
                return self._fallback_search(query, max_results)

        def _fallback_search(self, query, max_results):
            # Implement alternative search method
            pass

    def run(self):
        """Run the Streamlit interface with template support"""
        st.title("AI Web Scraper Pro")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # API Keys section
            with st.expander("API Keys", expanded=False):
                self._configure_api_keys()
            
            # Template selection
            st.subheader("Research Template")
            template_names = list(self.streamlit.template_cache.keys())
            selected_template = st.selectbox("Select Template", template_names)
            
            if selected_template:
                st.text_area("Template Preview", self.streamlit.template_cache[selected_template], height=200)
            
            # Industry/Market Selection
            st.subheader("Target Market")
            industry = st.text_input("Industry/Market of Interest")
            
            # Report Parameters
            st.subheader("Report Parameters")
            report_params = {
                "time_horizon": st.slider("Time Horizon (Years)", 1, 10, 5),
                "depth": st.select_slider("Analysis Depth", 
                    options=["Basic", "Standard", "Comprehensive"], 
                    value="Standard"),
                "regions": st.multiselect("Target Regions", 
                    ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"],
                    default=["North America", "Europe", "Asia-Pacific"]),
                "focus_areas": st.multiselect("Focus Areas",
                    ["Market Overview", "Financials", "Competition", "Investment Opportunities", "Trends"],
                    default=["Market Overview", "Competition"])
            }
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Research Query")
            if industry:
                # Generate research brief from template
                template = self.streamlit.template_cache.get(selected_template, "")
                brief = template.replace("[industry/market of interest]", industry)
                brief = brief.replace("[Title of Brief]", f"{industry} Market Analysis")
                
                # Display formatted brief
                st.markdown(brief)
                
                # Process button
                if st.button("Generate Report"):
                    self.streamlit._generate_comprehensive_report(industry, report_params)
        
        with col2:
            st.header("Quick Stats")
            if industry:
                self.streamlit._display_quick_stats(industry)

    def _generate_comprehensive_report(self, industry, params):
        """Generate a comprehensive report based on template and parameters"""
        st.session_state.results = {
            "metadata": {
                "industry": industry,
                "generated_at": datetime.now().isoformat(),
                "parameters": params
            },
            "market_overview": self._analyze_market(industry, params),
            "financials": self._analyze_financials(industry, params),
            "competition": self._analyze_competition(industry, params),
            "investment": self._analyze_investment(industry, params),
            "trends": self._analyze_trends(industry, params)
        }
        
        # Display results in organized sections
        self.streamlit.display_results()

    def _analyze_market(self, industry, params):
        """Analyze market overview section"""
        # Implement market analysis logic
        return {
            "market_size": {"value": 0, "unit": "USD", "year": 2025},
            "cagr": {"value": 0, "period": "2025-2030"},
            "segments": [],
            "regions": {}
        }

    def _analyze_financials(self, industry, params):
        """Analyze financials section"""
        # Implement financial analysis logic
        return {
            "key_metrics": {},
            "valuations": [],
            "financial_ratios": {}
        }

    def _analyze_competition(self, industry, params):
        """Analyze competitive landscape"""
        # Implement competition analysis logic
        return {
            "key_players": [],
            "market_share": {},
            "competitive_advantages": {}
        }

    def _analyze_investment(self, industry, params):
        """Analyze investment opportunities"""
        # Implement investment analysis logic
        return {
            "opportunities": [],
            "risks": [],
            "recommendations": []
        }

    def _analyze_trends(self, industry, params):
        """Analyze industry trends"""
        # Implement trend analysis logic
        return {
            "emerging_trends": [],
            "technology_impact": {},
            "future_outlook": {}
        }

    def _display_quick_stats(self, industry):
        """Display quick statistics about the industry"""
        st.metric("Market Size", "$XXB", delta="↑ 12.5%")
        st.metric("Key Players", "125+", delta="↑ 15")
        st.metric("CAGR", "8.5%", delta="↑ 2.1%")
        
        with st.expander("Top Companies"):
            st.write("1. Company A")
            st.write("2. Company B")
            st.write("3. Company C")

    def display_results(self):
        """Enhanced display of research results"""
        if not st.session_state.results:
            return

        # Report style selector with research brief specific styles
        report_styles = [
            "Executive Summary",
            "Investor Presentation",
            "Technical Deep Dive",
            "Market Entry Analysis",
            "Competitive Intelligence"
        ]
        selected_style = st.selectbox("Report Style", report_styles)
        
        # Export options
        export_formats = ["PDF Report", "Excel Dashboard", "PowerPoint", "Interactive HTML"]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("Research Results")
        with col2:
            export_format = st.selectbox("Export Format", export_formats)
            if st.button("Export Report"):
                self.streamlit._export_research_report(export_format)

        # Create tabs for different sections
        tabs = st.tabs([
            "Executive Overview",
            "Market Analysis",
            "Financial Analysis",
            "Competitive Landscape",
            "Investment Opportunities",
            "Trends & Innovation"
        ])
        
        with tabs[0]:
            self.streamlit._display_executive_overview()
        with tabs[1]:
            self.streamlit._display_market_analysis()
        with tabs[2]:
            self.streamlit._display_financial_analysis()
        with tabs[3]:
            self.streamlit._display_competitive_landscape()
        with tabs[4]:
            self.streamlit._display_investment_opportunities()
        with tabs[5]:
            self.streamlit._display_trends()

    def _display_executive_overview(self):
        """Display executive overview of research results"""
        results = st.session_state.results
        
        # Key metrics dashboard
        st.subheader("Key Metrics")
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Market Size",
                f"${results['market_overview']['market_size']['value']}B",
                delta="↑ 12.5%"
            )
        with cols[1]:
            st.metric(
                "CAGR",
                f"{results['market_overview']['cagr']['value']}%",
                delta="↑ 2.1%"
            )
        with cols[2]:
            st.metric(
                "Companies Analyzed",
                len(results['competition']['key_players']),
                delta="↑ 15"
            )
        with cols[3]:
            st.metric(
                "Investment Score",
                "8.5/10",
                delta="↑ 0.5"
            )
        
        # Executive summary
        st.subheader("Executive Summary")
        st.markdown(self.streamlit._generate_executive_summary())
        
        # Key findings
        st.subheader("Key Findings")
        for finding in self.streamlit._generate_key_findings():
            st.info(finding)

    def _generate_executive_summary(self):
        """Generate an executive summary from the research results"""
        results = st.session_state.results
        industry = results['metadata']['industry']
        
        summary = f"""
        ## {industry} Market Analysis
        
        This comprehensive analysis of the {industry} market reveals significant 
        opportunities for growth and investment. The market is characterized by 
        {len(results['competition']['key_players'])} key players and shows a 
        healthy CAGR of {results['market_overview']['cagr']['value']}%.
        
        ### Key Highlights
        - Market Size: ${results['market_overview']['market_size']['value']}B
        - Growth Trajectory: Positive
        - Market Maturity: Growing
        - Investment Climate: Favorable
        """
        
        return summary

    def _generate_key_findings(self):
        """Generate key findings from the research results"""
        return [
            "Finding 1: Market leadership is concentrated among top 5 players",
            "Finding 2: Emerging technologies are driving market evolution",
            "Finding 3: Regional expansion opportunities in APAC",
            "Finding 4: Regulatory changes creating new opportunities"
        ]

    def _export_research_report(self, format):
        """Export research report in various formats"""
        results = st.session_state.results
        
        if format == "PDF Report":
            self.streamlit._export_pdf_report(results)
        elif format == "Excel Dashboard":
            self.streamlit._export_excel_dashboard(results)
        elif format == "PowerPoint":
            self.streamlit._export_powerpoint(results)
        elif format == "Interactive HTML":
            self.streamlit._export_interactive_html(results)

    def _export_pdf_report(self, results):
        """Generate and export PDF report"""
        buffer = io.BytesIO()
        # Implement PDF generation logic
        return buffer

    def _export_excel_dashboard(self, results):
        """Generate and export Excel dashboard"""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            # Create overview sheet
            overview_df = pd.DataFrame({
                "Metric": ["Market Size", "CAGR", "Players"],
                "Value": [
                    results['market_overview']['market_size']['value'],
                    results['market_overview']['cagr']['value'],
                    len(results['competition']['key_players'])
                ]
            })
            overview_df.to_excel(writer, sheet_name="Overview", index=False)
            
            # Create other sheets
            for section in ["Market", "Financials", "Competition", "Trends"]:
                if section.lower() in results:
                    df = pd.DataFrame(results[section.lower()])
                    df.to_excel(writer, sheet_name=section, index=False)
        
        return buffer

    def _export_powerpoint(self, results):
        """Generate and export PowerPoint presentation"""
        buffer = io.BytesIO()
        # Implement PowerPoint generation logic
        return buffer

    def _export_interactive_html(self, results):
        """Generate and export interactive HTML report"""
        html_content = f"""
        <html>
            <head>
                <title>{results['metadata']['industry']} Market Analysis</title>
                <!-- Add required CSS and JS -->
            </head>
            <body>
                <!-- Add interactive content -->
            </body>
        </html>
        """
        return html_content.encode()

if __name__ == "__main__":
    agent = LLMAgentOrchestrator(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    agent.streamlit.setup_ui()
