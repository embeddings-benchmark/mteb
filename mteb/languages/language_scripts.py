from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mteb.languages.check_language_code import check_language_code

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self


@dataclass
class LanguageScripts:
    """A utility class to represent a set of languages and scripts.

    It can be used to check if a language or script is contained within the set.
    It supports both ISO 639-3 language codes and ISO 15924 script codes.

    Attributes:
        language_scripts: A set of language-script combinations (e.g. "eng-Latn").
        scripts: A set of script codes (e.g. "Latn").
        languages: A set of language codes (e.g. "eng").
        bare_languages: The language codes that were given without a script.
    """

    language_scripts: set[str]
    scripts: set[str]
    languages: set[str]
    bare_languages: set[str]

    @classmethod
    def from_languages_and_scripts(
        cls,
        languages: Sequence[str] | None = None,
        scripts: Sequence[str] | None = None,
    ) -> Self:
        """Create a LanguageScripts object from lists of languages and scripts.

        Args:
            languages: A list of language codes (e.g. ["eng", "fra-Latn"]).
            scripts: A list of script codes (e.g. ["Latn", "Cyrl"]).

        Returns:
            A LanguageScripts object representing the provided languages and scripts.
        """
        lang_script_codes = set()
        script_codes: set[str] = set()
        # normalize to 3 letter language codes
        normalized_langs = set()
        bare_langs = set()

        if scripts is not None:
            for script in scripts:
                check_language_code(script)
                script_codes.add(script)

        if languages is not None:
            for lang in languages:
                lang_script = lang.split("-")

                if len(lang_script) == 2:
                    normalized_langs.add(lang_script[0])
                    lang_script_codes.add(lang)
                    check_language_code(lang)
                    script_codes.add(lang_script[1])
                else:
                    check_language_code(lang)
                    normalized_langs.add(lang)
                    bare_langs.add(lang)

        return cls(
            language_scripts=lang_script_codes,
            scripts=script_codes,
            languages=normalized_langs,
            bare_languages=bare_langs,
        )

    @staticmethod
    def check_filters(
        languages: Sequence[str] | None = None,
        scripts: Sequence[str] | None = None,
    ) -> None:
        """Check that a script filter is not combined with a language code that includes a script.

        Args:
            languages: A list of language codes (e.g. ["eng", "fra-Latn"]).
            scripts: A list of script codes (e.g. ["Latn", "Cyrl"]).

        Raises:
            ValueError: If `scripts` is given and a language code in `languages` includes a script.
        """
        if not scripts or not languages:
            return
        with_script = [lang for lang in languages if "-" in lang]
        if with_script:
            raise ValueError(
                f"Language codes with a script {with_script} cannot be combined with the script filter {list(scripts)}, "
                "use the language code without its script (e.g. languages=['zho'], script=['Hant']) or drop `script`."
            )

    def contains_language(self, language: str) -> bool:
        """Whether the set contains a specific language.

        Args:
            language: The language code to check for. Can be in the form "eng" or "eng-Latn".

        Returns:
            True if the language is contained in the set, False otherwise.
        """
        if not self.language_scripts and not self.languages:
            return True

        langscript = language.split("-")
        is_langscript = len(langscript) == 2

        if is_langscript:
            _lang = langscript[0]
            if self.language_scripts and language in self.language_scripts:
                return True
            # a language given with a script does not match its other scripts
            return _lang in self.bare_languages

        return language in self.languages

    def contains_languages(self, languages: Iterable[str]) -> bool:
        """Whether is containing all the languages

        Args:
            languages: An iterable of language codes to check for.

        Returns:
            True if all languages are contained in the set, False otherwise.
        """
        return all(self.contains_language(l) for l in languages)

    def contains_script(self, script: str) -> bool:
        """Whether the set contains a specific script.

        Args:
            script: The script code to check for.

        Returns:
            True if the script is contained in the set, False otherwise.
        """
        return script in self.scripts

    def contains_langscript_script(self, langscript: str) -> bool:
        """Whether the set contains the script of a language-script code.

        Args:
            langscript: A code in the form "eng-Latn". Only the script part is checked.

        Returns:
            True if the script of the code is contained in the set, False otherwise.
        """
        return self.contains_script(langscript.rsplit("-", maxsplit=1)[-1])

    def contains_scripts(self, scripts: Iterable[str]) -> bool:
        """Whether is containing all the scripts

        Args:
            scripts: An iterable of script codes to check for.

        Returns:
            True if all scripts are contained in the set, False otherwise.
        """
        return all(self.contains_script(s) for s in scripts)
