"""This module provides mappings from ISO language codes to their corresponding language names, scripts, and language families.

Language codes (ISO 639-3) obtained from: https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab
Script codes (ISO 15924) obtained from: https://unicode.org/iso15924/iso15924.txt
"""

import json
from pathlib import Path

# Language mappings
path_to_lang_codes = Path(__file__).parent / "iso_639_3_to_language.json"
path_to_lang_scripts = Path(__file__).parent / "iso_15924_to_script.json"
path_to_lang_fam = Path(__file__).parent / "language_family.json"


with path_to_lang_codes.open("r") as f:
    ISO_TO_LANGUAGE = json.load(f)

with path_to_lang_scripts.open("r") as f:
    ISO_TO_SCRIPT = json.load(f)

with path_to_lang_fam.open("r") as f:
    ISO_TO_FAM = json.load(f)

ISO_TO_FAM_LEVEL0 = {k: v["level0"] for k, v in ISO_TO_FAM.items()}
