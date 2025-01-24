# LLM Online Search Assistant 🔍🤖  

**AI-Powered Research Orchestration Framework**  
*Multi-model architecture combining Gemini, Llama3, and Mixtral with Crawl4AI web extraction*  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url)  
[![PyPI](https://img.shields.io/pypi/v/llm-search-tools)](https://pypi.org/project/llm-search-tools/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

<img src="https://crawl4ai.com/assets/demo.gif" width="800" alt="Architecture Diagram">  


## What's New in v2.0 🚀  

### Key Enhancements  
- **Advanced Web Scraping**:  
  - Automatic pagination detection and handling  
  - Rotating user agents to avoid detection  
  - Configurable scroll behavior for dynamic content  

- **Enhanced Data Processing**:  
  - Field-specific extraction with structured JSON output  
  - Improved error handling and recovery  
  - Content-aware chunking for large pages  

- **UI Improvements**:  
  - Expandable result sections for better readability  
  - Interactive field extraction configuration  
  - Real-time progress tracking  

- **Performance Optimizations**:  
  - Asynchronous processing queue  
  - URL deduplication and smart content truncation  
  - Configurable timeout and headless browser settings  

---

## Core Components 🧩  

### Model-Tool Matrix  
| Tool Category         | Primary Model              | Context Window | Key Strengths                          |  
|-----------------------|----------------------------|----------------|----------------------------------------|  
| Web Search            | Gemini Flash               | 128k           | Google Search API integration          |  
| Data Cleaning         | Mixtral 8x7B               | 32k            | Large dataset processing               |  
| Financial Analysis    | Mixtral 8x7B               | 32k            | Pattern recognition in unstructured data|  
| Code Generation       | Gemini Flash               | 128k           | Native code execution tools            |  
| Safety Checks         | Llama Guard 3-8B           | 4k             | Real-time content filtering            |  
| Academic Research     | Llama3-70B                 | 8k             | Technical paper analysis               |  

---

## Installation & Setup ⚙️  

```bash  
pip install llm-search-tools crawl4ai streamlit  
export GROQ_API_KEY="your-key-here"  
export GOOGLE_API_KEY="your-key-here"  
```  

---

## Tool Integration Examples 🛠️  

### 1. Search & Citation Management  
```python  
from llm_search_tools import LLMAgentOrchestrator  

agent = LLMAgentOrchestrator(groq_api_key="your-groq-key", google_api_key="your-google-key")  

# Gemini-powered web search with Google integration  
results = agent.search_agent.web_search("LLM agent architectures 2024")  

# Llama3-70B citation extraction  
citations = agent.search_agent.knowledge_retrieval(results)  
```  

### 2. Data Processing Pipeline  
```python  
# Mixtral 8x7B for large-scale cleaning  
cleaned_data = agent.data_agent.clean_data(raw_dataset)  

# Llama3-70B for anomaly detection  
anomalies = agent.domain_agent.detect_fraud(cleaned_data)  
```  

### 3. Streamlit-Crawl4AI Integration  
```python  
import streamlit as st  
from crawl4ai import AsyncWebCrawler  

def analyze_url(url: str):  
    # Crawl with content-aware chunking  
    crawler = AsyncWebCrawler(chunk_size=30000)  
    content = crawler.crawl(url)  
    
    # Model routing based on content length  
    if len(content) > 32000:  
        return agent.data_agent.clean_data(content)  
    return agent.nlp_agent.analyze_sentiment(content)  

st.title("Web Content Analyzer")  
url = st.text_input("Enter URL")  
if st.button("Analyze"):  
    with st.spinner("Processing..."):  
        analysis = analyze_url(url)  
        st.json(analysis)  
```  

---

## Advanced Tool Configuration ⚡  

### Model-Specific Parameters  
```python  
# Configure Mixtral for financial analysis  
agent.domain_agent.analyze_finance(  
    financial_data,  
    params={  
        "temperature": 0.2,  
        "max_tokens": 2048,  
        "response_format": "json"  
    }  
)  

# Optimize Gemini for code generation  
agent.code_agent.generate_api(  
    "REST API for stock analysis",  
    tools=[genai.Tool.from_google_search()]  
)  
```  

### Multi-Agent Workflow  
```python  
# Task decomposition with Llama3-70B  
steps = agent.domain_agent.analyze_finance("Market trend analysis report")  

# Agent collaboration via Gemma-2-9B  
coordination = agent.data_agent.clean_data(steps)  
```  

---

## Key Integration Patterns 🔄  

### 1. Content-Aware Routing  
```python  
def process_content(content: str):  
    if "financial" in content.lower():  
        return agent.domain_agent.analyze_finance(content)  
    elif len(content) > 10000:  
        return agent.data_agent.clean_data(content)  
    return agent.nlp_agent.rewrite_query(content)  
```  

### 2. Safety-First Processing  
```python  
def safe_processing(content: str):  
    safety_check = agent.nlp_agent.rewrite_query(content, model="LLAMA_GUARD_3_8B")  
    return agent.process(content) if safety_check else None  
```  

### 3. Multi-Modal Analysis  
```python  
# Gemini Flash for image analysis  
screenshot_analysis = agent.nlp_agent.rewrite_query("dashboard.png")  

# Combine with text analysis  
combined_insights = agent.nlp_agent.analyze_sentiment(  
    f"{text_data}\n\nImage analysis: {screenshot_analysis}"  
)  
```  

---

## Deployment 🚀  

```dockerfile  
FROM python:3.10-slim  
RUN pip install llm-search-tools crawl4ai streamlit  
COPY . /app  
CMD ["streamlit", "run", "/app/app.py"]  
```  

---

## Contribution & Support 🤝  

**Guidelines**:  
- Use `Llama3-70B` for core logic changes.  
- Apply `Gemini Flash` for documentation generation.  
- Validate with `Llama Guard` for content safety.  

**Resources**:  
- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)  
- [Groq API Reference](https://console.groq.com/docs)  
- [Gemini Tool Integration Guide](https://ai.google.dev/docs)  

---

**License**: MIT  
**Maintainer**: AI Tools Team  
*Powered by cutting-edge LLM technologies and web extraction tools*  

---
