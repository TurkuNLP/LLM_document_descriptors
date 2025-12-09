
import hashlib

import unicodedata
import re
from typing import Tuple

# This module provides input processing utilities for descriptor disambiguation.
# Functions from here should be used accross scripts to ensure consistent processing.

def normalize_descriptor(s: str) -> str:
    """Canonical normalization of descriptor strings for merging
    All normalizations should be done using this function to ensure consistency
    What this does:
    - Unicode NFKC normalization
    - Replace common typographic quotes/primes with ASCII equivalents
    - Collapse literal backslash-escaped quotes into quotes
    - Normalize spaces (NBSP to space, remove ZWSP)
    - Replace underscores and multiple spaces with single space
    - Lowercase and trim
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    # Replace some common typographic quotes/primes with ASCII
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u2032": "'",
        "\u201C": '"', "\u201D": '"', "\u2033": '"',
    }
    for u, r in replacements.items():
        s = s.replace(u.encode("utf-8").decode("unicode_escape"), r)
    # Collapse literal backslash-escaped quotes into quotes
    s = s.replace('\\"', '"').replace("\\'", "'")
    # Normalize spaces and remove invisibles
    s = s.replace("\u00A0", " ").replace("\u200B", "")  # NBSP, ZWSP
    s = " ".join(s.split())
    # Replace underscores and spaces with single space, lowercase, and trim
    re.sub(r'[_\s]+', ' ', (s or ''))
    s = s.casefold()
    return s.strip()


def split_pair(text: str) -> Tuple[str, str]:
    """Splits a string into a descriptor and explanation.
    If malformed, returns a tuple of empty strings.
    Descriptor should typically be normalized afterwards.
    Explainer should be stripped of leading/trailing whitespace afterwards.
    """
    try:
        d, e = text.split(";", 1)
        return d, e
    except ValueError:
        return "", ""


def generate_stable_id(descriptor: str, explainer: str, *, length: int = 12) -> str:
    """Deterministic, stable ID for a descriptor–explainer pair.
    Uses SHA-1 over the normalized descriptor + stripped explainer string.
    Truncated to `length` hex characters (default 12).
    Note: normalization is done outside this function and should typically be done beforehand.
    """
    key = f"{descriptor}\u241f{explainer}"  # use an unlikely separator
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return digest[:length]

