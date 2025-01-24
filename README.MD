
---

# LLM Online Search Assistant 🔍🤖

A powerful toolkit for AI-powered web research, combining **Crawl4AI** for advanced web crawling and **LLMs** (Gemini, Llama3, Mixtral) for intelligent search orchestration. Includes Streamlit integration for interactive workflows.

**[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url)** [![PyPI](https://img.shields.io/pypi/v/crawl4ai)](https://pypi.org/project/crawl4ai/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="https://crawl4ai.com/assets/demo.gif" width="800" alt="Demo">

---

## Features ✨
1. **AI-Powered Search Orchestration**
   - Automatic query refinement using LLMs ()
   - Multi-engine search (Google/Bing/DuckDuckGo) with privacy controls 
   - Context-aware citation management & reference tracking 

2. **Advanced Crawling with Crawl4AI**
   - Dynamic JS rendering & lazy-load handling 
   - Structured data extraction (JSON/HTML/Markdown) 
   - Browser automation with proxy/session support 

3. **LLM Integration**
   - Local (Llama.cpp/Ollama) and cloud (Gemini) model support 
   - RAG pipeline for factual responses 
   - Multi-agent collaboration patterns 

4. **Streamlit UI**
   - Interactive search dashboard
   - Real-time crawling visualization
   - Export results to Markdown/JSON

---

## Installation 🛠️
```bash
# Core dependencies
pip install crawl4ai streamlit google-generativeai

# For Playwright browsers
python -m playwright install chromium
```

---

## Streamlit Integration Example 🎨

Create `app.py`:
```python
import streamlit as st
from crawl4ai import AsyncWebCrawler
import asyncio

st.title("🕸️ AI Web Research Assistant")

# Sidebar controls
with st.sidebar:
    search_query = st.text_input("Research topic", "Latest AI advancements")
    depth = st.slider("Search depth", 1-5, 3) 

async def run_crawl(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            js_code=["window.scrollTo(0, document.body.scrollHeight);"],
            extract_media=True
        )
        return result

if st.button("Start Research"):
    with st.spinner("Crawling..."):
        result = asyncio.run(run_crawl(f"https://google.com/search?q={search_query}"))
        
        st.subheader("Key Findings")
        st.markdown(result.fit_markdown)
        
        st.subheader("Raw Data")
        with st.expander("Show technical details"):
            st.json(result.model_dump())
```

Run with:
```bash
streamlit run app.py
```

---

## Advanced Use Cases 🔬

### Financial Data Extraction
```python
from crawl4ai import JsonCssExtractionStrategy

schema = {
    "baseSelector": ".stock-table tr",
    "fields": [
        {"name": "symbol", "selector": "td:nth-child(1)"},
        {"name": "price", "selector": "td:nth-child(2)"}
    ]
}

crawler = AsyncWebCrawler(extraction_strategy=JsonCssExtractionStrategy(schema))
result = await crawler.arun("https://markets.example.com")
```

### Academic Research Pipeline
```python
# Multi-step research process 
1. LLM generates search queries
2. Crawl4AI collects papers + citations
3. RAG system summarizes findings
4. Auto-generate literature review
```

---

## Deployment 🚀

**Docker Setup** ():
```dockerfile
FROM python:3.10
RUN pip install crawl4ai streamlit
COPY . /app
CMD ["streamlit", "run", "/app/app.py"]
```

---

## References 📚
1. Web-LLM-Assistant Architecture 
2. Crawl4AI Documentation 
3. RAG Search Implementation 
4. Streamlit UI Patterns 

---

**License**: MIT  
**Contribute**: PRs welcome! See our [contribution guidelines](https://github.com/unclecode/crawl4ai/blob/main/CONTRIBUTING.md).

*Built with ❤️ using [Crawl4AI](https://github.com/unclecode/crawl4ai) and [Streamlit](https://streamlit.io)*

---

This README combines key elements from multiple sources while maintaining technical accuracy. The Streamlit example demonstrates practical integration with Crawl4AI's async capabilities, and the structure follows modern OSS project conventions.
