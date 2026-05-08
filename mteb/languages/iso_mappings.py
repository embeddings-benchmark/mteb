"""This module provides mappings from ISO language codes to their corresponding language names, scripts, and language families.

Language codes (ISO 639-3) obtained from: https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab
Script codes (ISO 15924) obtained from: https://unicode.org/iso15924/iso15924.txt
ISO 639-1 to ISO 639-3 mapping derived from: https://github.com/datasets/language-codes/blob/main/data/language-codes-full.csv
Default script mapping derived from existing MTEB task data.
"""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Language mappings
path_to_lang_codes = Path(__file__).parent / "iso_639_3_to_language.json"
path_to_lang_scripts = Path(__file__).parent / "iso_15924_to_script.json"
path_to_lang_fam = Path(__file__).parent / "language_family.json"
_PATH_ISO1_TO_ISO3 = Path(__file__).parent / "iso_639_1_to_3.json"
_PATH_DEFAULT_SCRIPTS = Path(__file__).parent / "iso_639_3_to_default_script.json"


with path_to_lang_codes.open("r") as f:
    ISO_TO_LANGUAGE = json.load(f)

with path_to_lang_scripts.open("r") as f:
    ISO_TO_SCRIPT = json.load(f)

with path_to_lang_fam.open("r") as f:
    ISO_TO_FAM = json.load(f)

ISO_TO_FAM_LEVEL0 = {k: v["level0"] for k, v in ISO_TO_FAM.items()}


@cache
def _get_iso1_to_iso3() -> dict[str, str]:
    """Lazy-load ISO 639-1 to ISO 639-3 mapping."""
    with _PATH_ISO1_TO_ISO3.open("r") as f:
        return json.load(f)


@cache
def _get_iso3_to_default_script() -> dict[str, str]:
    """Lazy-load ISO 639-3 to default script mapping."""
    with _PATH_DEFAULT_SCRIPTS.open("r") as f:
        return json.load(f)


# Special HF language values that are not ISO codes
_HF_SPECIAL_VALUES = {"multilingual", "code", "mixed", "other", "unknown"}


def _hf_lang_to_iso_lang_script(hf_lang: str) -> str | None:
    """Convert a HuggingFace language code to MTEB's ISOLanguageScript format.

    Handles ISO 639-1 (2-letter), ISO 639-2/3 (3-letter) codes, and
    codes already in "xxx-Xxxx" format.

    Args:
        hf_lang: A language code from a HuggingFace model/dataset card.

    Returns:
        An ISOLanguageScript string (e.g., "eng-Latn") or None if the code
        cannot be converted.
    """
    hf_lang = hf_lang.strip().lower()

    if hf_lang in _HF_SPECIAL_VALUES:
        return None

    # Already in "xxx-Xxxx" format
    if "-" in hf_lang:
        parts = hf_lang.split("-")
        if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 4:
            return f"{parts[0]}-{parts[1].capitalize()}"
        return None

    # Resolve to ISO 639-3 code
    iso3: str | None = None
    if len(hf_lang) == 2:
        iso3 = _get_iso1_to_iso3().get(hf_lang)
    elif len(hf_lang) == 3 and hf_lang in ISO_TO_LANGUAGE:
        iso3 = hf_lang

    if iso3 is None:
        logger.debug(f"Could not resolve language code: {hf_lang}")
        return None

    script = _get_iso3_to_default_script().get(iso3)
    if script is None:
        logger.debug(f"No default script for language code: {iso3}")
        return None

    return f"{iso3}-{script}"


def _hf_langs_to_iso_lang_scripts(
    hf_langs: str | list[str] | None,
) -> list[str] | None:
    """Convert HuggingFace language codes to a list of MTEB ISOLanguageScript strings.

    Args:
        hf_langs: Language code(s) from a HuggingFace model card. Can be a single
            string, a list of strings, or None.

    Returns:
        A sorted list of unique ISOLanguageScript strings, or None if no valid
        codes could be converted.
    """
    if hf_langs is None:
        return None

    if isinstance(hf_langs, str):
        hf_langs = [hf_langs]

    results = set()
    for lang in hf_langs:
        converted = _hf_lang_to_iso_lang_script(lang)
        if converted is not None:
            results.add(converted)

    return sorted(results) if results else None
