from .iso_mappings import (
    ISO_TO_LANGUAGE,
    ISO_TO_SCRIPT,
    path_to_lang_codes,
    path_to_lang_scripts,
)
from .programming_languages import PROGRAMMING_LANGS


def check_language_code(code: str) -> None:
    """This method checks that the language code (e.g. "eng-Latn") is valid.

    Args:
        code: The language code to check.

    Raises:
        ValueError: If the language code is not valid.
    """
    lang, script = code.split("-")
    if script == "Code":
        if lang in PROGRAMMING_LANGS:
            return  # override for code
        else:
            raise ValueError(
                f"Programming language {lang} is not a valid programming language."
            )
    if lang not in ISO_TO_LANGUAGE:
        raise ValueError(
            f"Invalid language code: {lang}, you can find valid ISO 639-3 codes in {path_to_lang_codes}"
        )
    if script not in ISO_TO_SCRIPT:
        raise ValueError(
            f"Invalid script code: {script}, you can find valid ISO 15924 codes in {path_to_lang_scripts}"
        )
