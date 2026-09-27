"""
preprocessor.py — Text normalisation pipeline for business names and addresses.

This module implements all the cleaning and normalisation steps identified during EDA:
- Unicode normalisation
- Legal suffix standardisation
- Address abbreviation expansion
- Punctuation cleanup
- Core name extraction (stripping suffixes)
- Postal code extraction
"""
import re
import unicodedata
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import LEGAL_SUFFIX_MAP, LEGAL_SUFFIX_TOKENS, ADDR_ABBREVIATIONS

# Pre-compile regex patterns for performance
RE_MULTI_SPACE    = re.compile(r"\s+")
RE_PUNCTUATION    = re.compile(r"[^\w\s&/,.\-#']", re.UNICODE)  
RE_LEADING_NOISE  = re.compile(r"^[\-<>\[\]\(\)\{\}#*]+\s*")
RE_TRAILING_NOISE = re.compile(r"\s*[\-<>\[\]\(\)\{\}#*]+$")
RE_NULL_LITERAL   = re.compile(r"\bnull\b", re.IGNORECASE)
RE_NUMBERS        = re.compile(r"\b\d+\b")
RE_POSTAL_US      = re.compile(r"\b(\d{5})(?:-\d{4})?\b")         # US ZIP: 12345 or 12345-6789
RE_POSTAL_INDIA   = re.compile(r"\b(\d{6})\b")                    # India PIN: 6 digits
RE_POSTAL_FRANCE  = re.compile(r"\b(\d{5})\b")                    # France: 5 digits


def normalise_unicode(text: str) -> str:
    """Apply NFKD normalisation and strip combining marks (accents)."""
    # Keep a copy with accents for matching, but also provide accent-stripped version
    normalised = unicodedata.normalize("NFKD", text)
    return normalised


def strip_accents(text: str) -> str:
    """Remove all accent/diacritic marks, keeping base characters."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_business_name(name: str) -> str:
    """
    Full normalisation pipeline for a business name.
    
    Steps:
    1. Unicode NFKD normalisation
    2. Lowercase
    3. Strip leading/trailing noise characters (<<, --, etc.)
    4. Replace & with 'and'
    5. Normalise legal suffixes
    6. Collapse whitespace
    """
    if not name or name.strip() == "":
        return ""
    
    # Unicode normalise
    text = normalise_unicode(name)
    
    # Lowercase
    text = text.lower()
    
    # Strip leading/trailing noise (<<, --, etc.)
    text = RE_LEADING_NOISE.sub("", text)
    text = RE_TRAILING_NOISE.sub("", text)
    
    # Replace & with 'and'
    text = text.replace("&", " and ")
    
    # Remove brackets around words: (Inc) → Inc, [Consultancy] → Consultancy
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)
    
    # Normalise legal suffixes using token replacement
    tokens = text.split()
    normalised_tokens = []
    for token in tokens:
        # Strip trailing periods for matching
        token_clean = token.rstrip(".")
        if token_clean in LEGAL_SUFFIX_MAP:
            replacement = LEGAL_SUFFIX_MAP[token_clean]
            if replacement:  # Some map to "" (e.g. m/s)
                normalised_tokens.append(replacement)
        elif token + "." in LEGAL_SUFFIX_MAP:
            replacement = LEGAL_SUFFIX_MAP[token + "."]
            if replacement:
                normalised_tokens.append(replacement)
        else:
            normalised_tokens.append(token)
    
    text = " ".join(normalised_tokens)
    
    # Collapse whitespace
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    
    return text


def extract_core_name(normalised_name: str) -> str:
    """
    Extract the 'core' business name by removing legal suffixes.
    e.g., "prime money llc" → "prime money"
         "davis family private limited" → "davis family"
    """
    if not normalised_name:
        return ""
    
    tokens = normalised_name.split()
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIX_TOKENS]
    
    return " ".join(core_tokens).strip()


def clean_business_address(address: str) -> str:
    """
    Full normalisation pipeline for a business address.
    
    Steps:
    1. Unicode normalise
    2. Lowercase
    3. Replace null literals with empty string
    4. Remove ## prefix noise
    5. Expand abbreviations
    6. Collapse whitespace
    """
    if not address or address.strip() == "" or address.strip().lower() == "nan":
        return ""
    
    text = normalise_unicode(address)
    text = text.lower()
    
    # Remove null literals
    text = RE_NULL_LITERAL.sub("", text)
    
    # Remove ## prefix noise
    text = text.replace("##", " ")
    
    # Replace & with 'and'
    text = text.replace("&", " and ")
    
    # Remove brackets
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)
    
    # Expand abbreviations — handle tokens with trailing punctuation (commas, periods)
    tokens = text.split()
    expanded_tokens = []
    for token in tokens:
        # Strip trailing comma and/or period in any combination
        token_clean = token.rstrip(",. ")
        trailing = token[len(token_clean):]  # preserve trailing comma if present
        if token_clean in ADDR_ABBREVIATIONS:
            expanded_tokens.append(ADDR_ABBREVIATIONS[token_clean] + trailing)
        else:
            expanded_tokens.append(token)
    
    text = " ".join(expanded_tokens)
    
    # Clean up empty comma runs from null removal: ", ," → ","
    text = re.sub(r",\s*,", ",", text)
    # Remove leading/trailing commas
    text = text.strip(", ")
    
    # Collapse whitespace
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    
    return text


def extract_postal_code(address: str, country: str) -> str:
    """Extract postal/ZIP/PIN code from address based on country."""
    if not address:
        return ""
    
    if country == "US":
        match = RE_POSTAL_US.search(address)
        return match.group(1) if match else ""
    elif country == "India":
        match = RE_POSTAL_INDIA.search(address)
        return match.group(1) if match else ""
    elif country == "France":
        match = RE_POSTAL_FRANCE.search(address)
        return match.group(1) if match else ""
    else:
        # Try US-style 5 digits first, then 6 digits
        match = RE_POSTAL_US.search(address) or RE_POSTAL_INDIA.search(address)
        return match.group(1) if match else ""


def extract_name_tokens(normalised_name: str) -> set:
    """Extract set of meaningful word tokens from a normalised name."""
    if not normalised_name:
        return set()
    return set(normalised_name.split())


def extract_addr_tokens(normalised_address: str) -> set:
    """Extract set of word tokens from a normalised address."""
    if not normalised_address:
        return set()
    # Remove commas and split
    text = normalised_address.replace(",", " ")
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return set(text.split())


def extract_numeric_tokens(text: str) -> set:
    """Extract all numeric tokens from text (house numbers, PINs, etc.)."""
    if not text:
        return set()
    return set(RE_NUMBERS.findall(text))


def preprocess_dataframe(df: pd.DataFrame, desc: str = "records") -> pd.DataFrame:
    """
    Apply full preprocessing to a source DataFrame.
    
    Adds columns:
    - name_clean: normalised business name
    - name_core: core name without legal suffixes
    - name_accent_stripped: name with accents removed (for blocking)
    - addr_clean: normalised business address
    - postal_code: extracted postal/ZIP/PIN code
    - name_tokens: set of name word tokens (stored as frozenset)
    - addr_tokens: set of address word tokens (stored as frozenset)
    """
    print(f"  Preprocessing {len(df):,} {desc}...")
    
    # Vectorised string operations for speed
    tqdm.pandas(desc=f"  Cleaning names")
    df["name_clean"] = df["business_name"].progress_apply(clean_business_name)
    
    tqdm.pandas(desc=f"  Extracting core names")
    df["name_core"] = df["name_clean"].progress_apply(extract_core_name)
    
    tqdm.pandas(desc=f"  Stripping accents")
    df["name_accent_stripped"] = df["name_clean"].progress_apply(strip_accents)
    
    tqdm.pandas(desc=f"  Cleaning addresses")
    df["addr_clean"] = df["business_address"].progress_apply(clean_business_address)
    
    # Postal code extraction (vectorised by country)
    df["postal_code"] = df.apply(
        lambda r: extract_postal_code(r["addr_clean"], r["country"]), axis=1
    )
    
    # Compute combined text for TF-IDF blocking
    df["combined_text"] = df["name_clean"] + " " + df["addr_clean"]
    
    print(f"  ✓ Preprocessing complete for {len(df):,} {desc}")
    return df
