"""Language codes (ISO 639-3) obtained from: https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab
Script codes (ISO 15924) obtained from: https://unicode.org/iso15924/iso15924.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Language typings
ISO_LANGUAGE_SCRIPT = str  # a 3-letter ISO 639-3 language code followed by a 4-letter ISO 15924 script code (e.g. "eng-Latn")
ISO_LANGUAGE = str  # a 3-letter ISO 639-3 language code
ISO_SCRIPT = str  # a 4-letter ISO 15924 script code


# Language mappings
path_to_lang_codes = Path(__file__).parent / "iso_639_3_to_language.json"
path_to_lang_scripts = Path(__file__).parent / "iso_15924_to_script.json"


with path_to_lang_codes.open("r") as f:
    ISO_TO_LANGUAGE = json.load(f)

with path_to_lang_scripts.open("r") as f:
    ISO_TO_SCRIPT = json.load(f)


@dataclass
class LanguageScripts:
    language_scripts: set[str]
    scripts: set[str]
    languages: set[str]

    @classmethod
    def from_languages_and_scripts(
        cls, languages: list[str] | None = None, scripts: list[str] | None = None
    ) -> LanguageScripts:
        lang_script_codes = set()
        script_codes: set[str] = set(scripts) if (scripts is not None) else set()
        # normalize to 3 letter language codes
        normalized_langs = set()

        if languages is not None:
            for lang in languages:
                lang_script = lang.split("-")

                if len(lang_script) == 2:
                    normalized_langs.add(lang_script[0])
                    lang_script_codes.add(lang)
                    script_codes.add(lang_script[1])
                else:
                    normalized_langs.add(lang)

        return cls(
            language_scripts=lang_script_codes,
            scripts=script_codes,
            languages=normalized_langs,
        )

    def contains_language(self, language: str) -> bool:
        if not self.language_scripts and not self.languages:
            return True

        langscript = language.split("-")
        is_langscript = len(langscript) == 2

        if is_langscript:
            _lang = langscript[0]
            if self.language_scripts and language in self.language_scripts:
                return True
        else:
            _lang = language

        if _lang in self.languages:
            return True
        return False

    def contains_script(self, script: str) -> bool:
        return script in self.scripts
