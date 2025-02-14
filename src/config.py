USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    # Add more user agents...
]

TIMEOUT_SETTINGS = {
    "page_load": 30,
    "script": 10,
    "retry": 15
}

HEADLESS_OPTIONS = [
    "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1920,1080",
    "--disable-search-engine-choice-screen", "--disable-blink-features=AutomationControlled",
    "--no-sandbox"
]

STRATEGIES = {
    "cosmic": "Full JS rendering with smart detection bypass",
    "stealth": "Headless browsing with anti-detection",
    "lightweight": "Basic HTML fetching",
    "custom": "User-defined strategy"
}