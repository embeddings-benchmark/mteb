"""Language codes (ISO 639-3) obtained from: https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab
Script codes (ISO 15924) obtained from: https://unicode.org/iso15924/iso15924.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

path_to_lang_codes = Path(__file__).parent / "iso_639_3_to_language.json"
path_to_lang_scripts = Path(__file__).parent / "iso_15924_to_script.json"


with path_to_lang_codes.open("r") as f:
    ISO_TO_LANGUAGE = json.load(f)

with path_to_lang_scripts.open("r") as f:
    ISO_TO_SCRIPT = json.load(f)


@dataclass
class LangScriptFilter:
    language_scripts: set[str]
    scripts: set[str]
    languages: set[str]

    @classmethod
    def from_languages_and_scripts(
        cls, languages: list[str] | None, scripts: list[str] | None
    ) -> LangScriptFilter:
        lang_script_codes = set()
        # normalize to 3 letter language codes
        normalized_langs = set()
        filter_lang = languages is not None

        if filter_lang:
            for lang in languages:
                lang_script = lang.split("-")

                is_lang_script_code = len(lang_script) == 2
                if is_lang_script_code:
                    normalized_langs.add(lang_script[0])
                    lang_script_codes.add(lang)
                else:
                    normalized_langs.add(lang)

        filter_scripts = scripts is not None
        script_codes: set[str] = set(scripts) if filter_scripts else set()
        return cls(
            language_scripts=lang_script_codes,
            scripts=script_codes,
            languages=normalized_langs,
        )

    def is_valid_language(self, language: str) -> bool:
        passed_lang_script_filter = False
        passed_language_fitler = False

        if not self.language_scripts or language in self.language_scripts:
            passed_lang_script_filter = True
        if not self.languages or language in self.languages:
            passed_language_fitler = True

        return passed_lang_script_filter and passed_language_fitler

    def is_valid_script(self, script: str) -> bool:
        if not self.scripts or script in self.scripts:
            return True
        return False
