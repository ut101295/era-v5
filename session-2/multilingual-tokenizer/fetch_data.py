"""
Fetch India's Wikipedia page content in English, Hindi, Telugu, and Marathi.
Uses the MediaWiki API directly — no extra dependencies beyond `requests`.
"""

import os
import re
import requests

# Wikipedia language codes and their page titles for "India"
LANGUAGES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "mr": "भारत",
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "mr": "Marathi",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch_wikipedia_page(lang: str, title: str) -> str:
    """Fetch the full plain-text extract of a Wikipedia page."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    
    # Wikipedia API has a limit on extract size, so we need to handle continuation
    full_text = ""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exlimit": 1,
    }
    
    response = requests.get(url, params=params, headers={
        "User-Agent": "BPETokenizerAssignment/1.0 (educational project)"
    })
    response.raise_for_status()
    data = response.json()
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            raise ValueError(f"Page '{title}' not found in {lang}.wikipedia.org")
        full_text = page_data.get("extract", "")
    
    return full_text


def clean_text(text: str) -> str:
    """Clean Wikipedia text by removing noise while preserving multilingual content."""
    # Remove reference markers like [1], [2], [citation needed]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove section headers that are just "==" markers
    text = re.sub(r'={2,}.*?={2,}', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    
    return text


def fetch_all():
    """Fetch and save Wikipedia pages for all configured languages."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for lang, title in LANGUAGES.items():
        print(f"Fetching {LANGUAGE_NAMES[lang]} ({lang}) Wikipedia page: '{title}'...")
        
        try:
            raw_text = fetch_wikipedia_page(lang, title)
            cleaned = clean_text(raw_text)
            
            filepath = os.path.join(DATA_DIR, f"{lang}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned)
            
            # Stats
            word_count = len(cleaned.split())
            char_count = len(cleaned)
            print(f"  ✓ Saved to {filepath}")
            print(f"    Characters: {char_count:,} | Words: {word_count:,}")
            
        except Exception as e:
            print(f"  ✗ Error fetching {lang}: {e}")
            raise
    
    print("\nAll pages fetched successfully!")


if __name__ == "__main__":
    fetch_all()
