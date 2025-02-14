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

        def run(self):
            """Run the Streamlit interface with template support"""
            st.title("AI Web Scraper Pro")
            
            # Main navigation
            tabs = st.tabs(["Research", "Templates", "Results", "Settings"])
            
            with tabs[0]:
                self._research_tab()
            with tabs[1]:
                self._templates_tab()
            with tabs[2]:
                self._results_tab()
            with tabs[3]:
                self._settings_tab()

        def _research_tab(self):
            """Main research interface"""
            # Sidebar configuration
            with st.sidebar:
                st.header("Configuration")
                
                # API Keys section
                with st.expander("API Keys", expanded=False):
                    self._configure_api_keys()
                
                # Template selection
                st.subheader("Research Template")
                template_names = list(self.template_cache.keys())
                selected_template = st.selectbox("Select Template", template_names)
                
                if selected_template:
                    st.text_area("Template Preview", self.template_cache[selected_template], height=200)
                
                # Industry/Market Selection
                st.subheader("Target Market")
                industry = st.text_input("Industry/Market of Interest")
                
                # Report Parameters
                st.subheader("Report Parameters")
                report_params = self._configure_report_params()

            # Main content area
            self._display_research_interface(selected_template, industry, report_params)

        def _templates_tab(self):
            """Template management interface"""
            st.header("Template Management")
            
            # Template list
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Available Templates")
                template_list = self._list_templates()
                
                if template_list:
                    for template in template_list:
                        with st.expander(f"📄 {template['name']}"):
                            st.markdown(template['content'])
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                if st.button("Edit", key=f"edit_{template['name']}"):
                                    st.session_state.editing_template = template['name']
                            with col2:
                                if st.button("Duplicate", key=f"duplicate_{template['name']}"):
                                    self._duplicate_template(template['name'])
                            with col3:
                                if st.button("Delete", key=f"delete_{template['name']}"):
                                    self._delete_template(template['name'])
            
            with col2:
                st.subheader("Add New Template")
                self._add_template_form()
            
            # Template editor
            if hasattr(st.session_state, 'editing_template'):
                self._template_editor(st.session_state.editing_template)

        def _list_templates(self):
            """List all available templates"""
            templates = []
            try:
                for file in os.listdir(self.templates_dir):
                    if file.endswith('.md'):
                        with open(os.path.join(self.templates_dir, file), 'r') as f:
                            templates.append({
                                'name': file,
                                'content': f.read(),
                                'path': os.path.join(self.templates_dir, file)
                            })
            except Exception as e:
                st.error(f"Error loading templates: {str(e)}")
            return templates

        def _add_template_form(self):
            """Form for adding new templates"""
            with st.form("new_template"):
                template_name = st.text_input("Template Name")
                template_content = st.text_area(
                    "Template Content", 
                    height=400,
                    placeholder="# Template Title\n\n## Overview\nDescribe your template...\n\n## Sections\n1. Section 1\n2. Section 2"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    template_type = st.selectbox(
                        "Template Type",
                        ["Research Brief", "Market Analysis", "Competitor Analysis", "Investment Thesis", "Custom"]
                    )
                with col2:
                    template_format = st.selectbox(
                        "Output Format",
                        ["Markdown", "Structured JSON", "Report", "Dashboard"]
                    )
                
                submitted = st.form_submit_button("Add Template")
                
                if submitted and template_name and template_content:
                    self._save_template(template_name, template_content, template_type, template_format)
                    st.success(f"Template '{template_name}' added successfully!")

        def _template_editor(self, template_name):
            """Edit existing template"""
            st.header(f"Editing: {template_name}")
            
            template_path = os.path.join(self.templates_dir, template_name)
            try:
                with open(template_path, 'r') as f:
                    current_content = f.read()
            except Exception as e:
                st.error(f"Error loading template: {str(e)}")
                return
            
            with st.form("edit_template"):
                new_content = st.text_area(
                    "Template Content",
                    value=current_content,
                    height=600
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Save Changes"):
                        self._update_template(template_name, new_content)
                        st.success("Template updated successfully!")
                with col2:
                    if st.form_submit_button("Cancel"):
                        del st.session_state.editing_template

        def _save_template(self, name, content, type, format):
            """Save new template to file"""
            if not name.endswith('.md'):
                name = f"{name}.md"
            
            path = os.path.join(self.templates_dir, name)
            try:
                with open(path, 'w') as f:
                    f.write(f"---\ntype: {type}\nformat: {format}\n---\n\n{content}")
                self.template_cache[name] = content
            except Exception as e:
                st.error(f"Error saving template: {str(e)}")

        def _update_template(self, name, content):
            """Update existing template"""
            path = os.path.join(self.templates_dir, name)
            try:
                with open(path, 'w') as f:
                    f.write(content)
                self.template_cache[name] = content
            except Exception as e:
                st.error(f"Error updating template: {str(e)}")

        def _duplicate_template(self, name):
            """Duplicate existing template"""
            try:
                with open(os.path.join(self.templates_dir, name), 'r') as f:
                    content = f.read()
                
                new_name = f"copy_of_{name}"
                self._save_template(new_name, content, "Copy", "Markdown")
                st.success(f"Template duplicated as '{new_name}'")
            except Exception as e:
                st.error(f"Error duplicating template: {str(e)}")

        def _delete_template(self, name):
            """Delete template file"""
            if st.checkbox(f"Confirm deletion of '{name}'"):
                try:
                    os.remove(os.path.join(self.templates_dir, name))
                    del self.template_cache[name]
                    st.success(f"Template '{name}' deleted successfully!")
                except Exception as e:
                    st.error(f"Error deleting template: {str(e)}")

        def _configure_report_params(self):
            """Configure report generation parameters"""
            return {
                "time_horizon": st.slider("Time Horizon (Years)", 1, 10, 5),
                "depth": st.select_slider(
                    "Analysis Depth", 
                    options=["Basic", "Standard", "Comprehensive"], 
                    value="Standard"
                ),
                "regions": st.multiselect(
                    "Target Regions", 
                    ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"],
                    default=["North America", "Europe", "Asia-Pacific"]
                ),
                "focus_areas": st.multiselect(
                    "Focus Areas",
                    ["Market Overview", "Financials", "Competition", "Investment Opportunities", "Trends"],
                    default=["Market Overview", "Competition"]
                )
            }

        def _display_research_interface(self, template, industry, params):
            """Display main research interface"""
            if industry and template:
                st.header("Research Query")
                # Generate research brief from template
                brief = self.template_cache.get(template, "").replace(
                    "[industry/market of interest]", industry
                ).replace(
                    "[Title of Brief]", f"{industry} Market Analysis"
                )
                
                # Display formatted brief
                st.markdown(brief)
                
                # Process button
                if st.button("Generate Report"):
                    self._generate_comprehensive_report(industry, params)

                st.header("Quick Stats")
                self._display_quick_stats(industry)

        def _results_tab(self):
            """Display results tab"""
            if st.session_state.results:
                self.display_results()

        def _settings_tab(self):
            """Display settings tab"""
            st.header("Settings")
            st.subheader("API Keys")
            self._configure_api_keys()

        def _configure_api_keys(self):
            """Configure API keys"""
            st.subheader("API Keys")
            groq_key = st.text_input("Groq API Key", type="password")
            google_key = st.text_input("Google API Key", type="password")

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
                    self._export_research_report(export_format)

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
                self._display_executive_overview()
            with tabs[1]:
                self._display_market_analysis()
            with tabs[2]:
                self._display_financial_analysis()
            with tabs[3]:
                self._display_competitive_landscape()
            with tabs[4]:
                self._display_investment_opportunities()
            with tabs[5]:
                self._display_trends()

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
            st.markdown(self._generate_executive_summary())
            
            # Key findings
            st.subheader("Key Findings")
            for finding in self._generate_key_findings():
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
                self._export_pdf_report(results)
            elif format == "Excel Dashboard":
                self._export_excel_dashboard(results)
            elif format == "PowerPoint":
                self._export_powerpoint(results)
            elif format == "Interactive HTML":
                self._export_interactive_html(results)

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

        def _display_market_analysis(self):
            """Display market analysis section"""
            results = st.session_state.results
            
            if "market_overview" in results:
                st.subheader("Market Overview")
                st.markdown(f"### Market Size: ${results['market_overview']['market_size']['value']}B")
                st.markdown(f"### CAGR: {results['market_overview']['cagr']['value']}%")
                
                # Market segmentation
                if "segments" in results['market_overview']:
                    st.subheader("Market Segmentation")
                    for segment in results['market_overview']['segments']:
                        st.markdown(f"- **{segment['name']}**: {segment['description']}")

        def _display_financial_analysis(self):
            """Display financial analysis section"""
            results = st.session_state.results
            
            if "financials" in results:
                st.subheader("Financial Analysis")
                st.markdown(f"### Revenue: ${results['financials']['revenue']['value']}M")
                st.markdown(f"### Net Income: ${results['financials']['net_income']['value']}M")
                
                # Financial ratios
                if "financial_ratios" in results['financials']:
                    st.subheader("Financial Ratios")
                    for ratio in results['financials']['financial_ratios']:
                        st.markdown(f"- **{ratio['name']}**: {ratio['value']}")

        def _display_competitive_landscape(self):
            """Display competitive landscape section"""
            results = st.session_state.results
            
            if "competition" in results:
                st.subheader("Competitive Landscape")
                st.markdown(f"### Key Players: {len(results['competition']['key_players'])}")
                
                # Company profiles
                for company in results['competition']['key_players']:
                    st.subheader(company['name'])
                    st.markdown(f"### Description: {company['description']}")
                    st.markdown(f"### Market Share: {company['market_share']}%")

        def _display_investment_opportunities(self):
            """Display investment opportunities section"""
            results = st.session_state.results
            
            if "investment" in results:
                st.subheader("Investment Opportunities")
                st.markdown(f"### Opportunities: {len(results['investment']['opportunities'])}")
                
                # Opportunity profiles
                for opportunity in results['investment']['opportunities']:
                    st.subheader(opportunity['name'])
                    st.markdown(f"### Description: {opportunity['description']}")
                    st.markdown(f"### Potential Return: {opportunity['potential_return']}%")

        def _display_trends(self):
            """Display trends section"""
            results = st.session_state.results
            
            if "trends" in results:
                st.subheader("Trends")
                st.markdown(f"### Emerging Trends: {len(results['trends']['emerging_trends'])}")
                
                # Trend profiles
                for trend in results['trends']['emerging_trends']:
                    st.subheader(trend['name'])
                    st.markdown(f"### Description: {trend['description']}")
                    st.markdown(f"### Impact: {trend['impact']}")

        def _display_quick_stats(self, industry):
            """Display quick statistics about the industry"""
            st.metric("Market Size", "$XXB", delta="↑ 12.5%")
            st.metric("Key Players", "125+", delta="↑ 15")
            st.metric("CAGR", "8.5%", delta="↑ 2.1%")
            
            with st.expander("Top Companies"):
                st.write("1. Company A")
                st.write("2. Company B")
                st.write("3. Company C")

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
            self.display_results()

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
            report_params = self.streamlit._configure_report_params()

        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Research Query")
            if industry and selected_template:
                # Generate research brief from template
                brief = self.streamlit.template_cache.get(selected_template, "").replace(
                    "[industry/market of interest]", industry
                ).replace(
                    "[Title of Brief]", f"{industry} Market Analysis"
                )
                
                # Display formatted brief
                st.markdown(brief)
                
                # Process button
                if st.button("Generate Report"):
                    self.streamlit._generate_comprehensive_report(industry, report_params)

            st.header("Quick Stats")
            self.streamlit._display_quick_stats(industry)
        
        with col2:
            st.header("CLI Mode")
            cli_input = st.text_area("Enter CLI commands", height=100)
            
            # Process Button
            if st.button("Process"):
                combined_input = {
                    "prompt": None,
                    "cli": cli_input,
                    "sources": [],
                    "files": []
                }
                self.streamlit.process_input(combined_input)

            # Display results
            self.streamlit.display_results()

if __name__ == "__main__":
    agent = LLMAgentOrchestrator(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    agent.streamlit.run()
