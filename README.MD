# AI Web Scraper Pro v2.3

An intelligent web scraping solution with multi-LLM orchestration, available as a public Streamlit app.

## 🔑 Getting Started

1. **API Keys Required**
   - In the left sidebar menu, locate the "API Keys" section
   - Enter your Groq API key (get it from [groq.com](https://groq.com))
   - Enter your Google API key (get it from [console.cloud.google.com](https://console.cloud.google.com))
   - Keys are stored temporarily in your session only

2. **Custom Sources Integration**
   In the sidebar, you can add multiple types of sources:
   - 🌐 Websites (enter URLs)
   - 📄 PDF files (upload)
   - 📝 Text files (upload)
   - 📋 Markdown files (upload)

## 💻 Dual Input Interface

The app features a dual-input interface that allows you to use both GUI and CLI simultaneously:

### Natural Language Query (Left Panel)
- Enter free-form research queries
- Example: "Find AI startups in healthcare with recent funding"
- The app will intelligently process your query and search across all sources

### CLI Commands (Right Panel)
- Enter specific commands for targeted searches
- Example: `search --domain healthcare --type startup --timeframe 2024`
- Supports advanced filtering and data extraction

### Custom Source Integration
Both inputs will consider your added sources:
- Websites are scraped for relevant information
- PDFs are parsed and analyzed
- Text and Markdown files are included in the search scope

## 🎯 Features

### Intelligent Processing
- Simultaneous processing of GUI and CLI inputs
- Cross-reference between custom sources and web results
- Smart content extraction from multiple file types

### Source Management
- Add/remove sources dynamically
- Mix different source types
- Preview added sources in sidebar
- Clear all sources with one click

### Data Integration
- Combines results from all sources
- Structured output format
- Cross-reference between sources
- Highlight matching content

## 📊 Output Format

The app provides structured output combining all sources:

```json
{
    "structured_data": {
        "companies": [...],
        "technologies": [...],
        "trends": [...]
    },
    "sources_processed": [
        {"type": "website", "url": "...", "data": {...}},
        {"type": "pdf", "name": "report.pdf", "data": {...}},
        {"type": "markdown", "name": "notes.md", "data": {...}}
    ],
    "analysis": {
        "themes": [...],
        "relationships": [...],
        "timeline": [...]
    }
}
```

## 💡 Tips for Best Results

1. **Combine Inputs Effectively**
   - Use natural language for broad queries
   - Use CLI for specific filters
   - Add relevant custom sources

2. **Source Integration**
   - Add sources before running queries
   - Mix different source types
   - Use clear file names

3. **Query Optimization**
   - Be specific in natural language queries
   - Use proper CLI command syntax
   - Reference source types in queries

## ❓ Troubleshooting

1. **Source Issues**
   - Ensure URLs are accessible
   - Check file formats are supported
   - Verify file size limits

2. **Processing Problems**
   - Try simpler queries first
   - Check source compatibility
   - Review error messages

3. **Output Issues**
   - Reduce number of sources for faster processing
   - Use more specific queries
   - Check source format compatibility

## 🔒 Privacy Note

- API keys are stored in session only
- Uploaded files are processed temporarily
- No data is permanently stored
- Clear your browser cache to remove all data

## 🙏 Acknowledgments

This project builds upon and integrates several powerful technologies and libraries:

- **crawl4ai**: Advanced web crawling and data extraction framework providing sophisticated features like:
  - Asynchronous web crawling
  - LLM-powered extraction
  - Semantic content analysis
  - Multiple chunking strategies
  - HTML to text conversion
  - Caching and retry policies
  - Browser automation
  - SSL certificate handling

- **Groq**: State-of-the-art language models
- **Google Generative AI**: Advanced AI capabilities
- **Streamlit**: Interactive web interface
- **Plotly**: Data visualization
- **Pandas**: Data manipulation and analysis

### Special Thanks To
- crawl4ai team
- Groq team
- Streamlit community
- Open-source contributors

## Research & Reporting Features

### Custom Research Templates
The application supports custom research templates located in the `/Prompts` directory. These templates define the structure and scope of market research reports.

#### Example Template Structure
```markdown
PROMPT: Write a comprehensive report about the [industry/market of interest] and identify key players in different regions.

Research Brief: [Title of Brief]

### Objective
Compile a comprehensive, data-driven report on the state of [industry/market of interest]...

### Methodology
- Data collection methods
- Source validation
- Analysis techniques

### Scope
1. Market Overview
2. Financial Analysis
3. Competitive Landscape
4. Investment Opportunities
5. Trends & Innovation
```

### Report Generation

#### 1. Configure Research Parameters
```python
report_params = {
    "time_horizon": 5,  # Years
    "depth": "Comprehensive",  # Basic, Standard, Comprehensive
    "regions": ["North America", "Europe", "Asia-Pacific"],
    "focus_areas": ["Market Overview", "Competition"]
}
```

#### 2. Example Report Output
```markdown
# AI Software Market Analysis
Generated: 2025-02-14

## Executive Summary
Market Size: $125B (2025)
CAGR: 28.5% (2025-2030)
Key Players: 150+
Investment Score: 8.5/10

## Market Overview
- Total Addressable Market (TAM)
- Regional Distribution
- Growth Drivers
- Market Segments

## Competitive Landscape
- Market Share Analysis
- Key Player Profiles
- Competitive Advantages
- Market Positioning

## Investment Opportunities
- Growth Areas
- Risk Assessment
- ROI Analysis
- Entry Barriers
```

### Export Options

#### 1. PDF Report
Professional PDF reports include:
- Executive Summary
- Data Visualizations
- Detailed Analysis
- Source Citations

#### 2. Excel Dashboard
Interactive Excel workbook with:
- Overview Dashboard
- Market Data
- Financial Metrics
- Competitive Analysis
- Trend Tracking

#### 3. PowerPoint Presentation
Investor-ready slides covering:
- Market Highlights
- Growth Opportunities
- Key Players
- Investment Thesis

#### 4. Interactive HTML
Web-based report featuring:
- Interactive Charts
- Filterable Tables
- Dynamic Visualizations
- Export Capabilities

### Example Usage

1. **Select Template & Industry**
   ```python
   template = "Research Brief.txt"
   industry = "AI Software"
   ```

2. **Configure Analysis**
   ```python
   params = {
       "time_horizon": 5,
       "depth": "Comprehensive",
       "regions": ["North America", "Europe", "APAC"],
       "focus_areas": ["Market", "Competition", "Investment"]
   }
   ```

3. **Generate Report**
   ```python
   report = generate_report(industry, template, params)
   ```

4. **Export Results**
   ```python
   # Export as PDF
   report.export("pdf", "ai_software_analysis.pdf")
   
   # Export as Excel Dashboard
   report.export("excel", "ai_software_metrics.xlsx")
   ```

### Visualization Types

1. **Market Analysis**
   - Market Size Evolution
   - Regional Distribution
   - Segment Breakdown
   - Growth Trends

2. **Financial Metrics**
   - Revenue Analysis
   - Profitability Metrics
   - Valuation Multiples
   - Investment Returns

3. **Competitive Analysis**
   - Market Share
   - Competitor Positioning
   - Strength Assessment
   - Strategic Grouping

4. **Technology Trends**
   - Innovation Timeline
   - Adoption Curves
   - Tech Stack Analysis
   - Future Roadmap

### Best Practices

1. **Template Design**
   - Use clear section headers
   - Include specific metrics
   - Define scope clearly
   - Add methodology details

2. **Report Generation**
   - Validate data sources
   - Cross-reference metrics
   - Include confidence levels
   - Cite sources properly

3. **Data Visualization**
   - Choose appropriate charts
   - Maintain consistency
   - Use clear labels
   - Include legends

4. **Export Formats**
   - Optimize for readability
   - Include navigation aids
   - Add executive summaries
   - Provide source data

### Customization

#### 1. Adding New Templates
Create new template files in the `/Prompts` directory:
```
/Prompts
  ├── Research Brief.txt
  ├── Competitor Analysis.txt
  ├── Investment Thesis.txt
  └── Technology Assessment.txt
```

#### 2. Custom Report Styles
Available styles include:
- Executive Summary
- Investor Presentation
- Technical Deep Dive
- Market Entry Analysis
- Competitive Intelligence

#### 3. Custom Visualizations
Add new visualization types:
```python
@visualization_type
def custom_chart(data, params):
    """Create custom visualization"""
    # Implementation
```

#### 4. Export Templates
Customize export formats:
```python
@export_format
def custom_export(report, path):
    """Custom export format"""
    # Implementation
