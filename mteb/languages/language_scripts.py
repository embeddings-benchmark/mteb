from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class LanguageScripts:
    """A utility class to represent a set of languages and scripts.

    It can be used to check if a language or script is contained within the set.
    It supports both ISO 639-3 language codes and ISO 15924 script codes.

    Attributes:
        language_scripts: A set of language-script combinations (e.g. "eng-Latn").
        scripts: A set of script codes (e.g. "Latn").
        languages: A set of language codes (e.g. "eng").
    """

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
        """Whether the set contains a specific language."""
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

    def contains_languages(self, languages: Iterable[str]) -> bool:
        """Whether is contains all of the languages"""
        for l in languages:
            if not self.contains_language(l):
                return False
        return True

    def contains_script(self, script: str) -> bool:
        """Whether the set contains a specific script."""
        return script in self.scripts

    def contains_scripts(self, scripts: Iterable[str]) -> bool:
        """Whether is contains all of the scripts"""
        for s in scripts:
            if not self.contains_script(s):
                return False
        return True
