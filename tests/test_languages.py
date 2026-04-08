"""Contains test cases for the mteb.languages submodule"""

import pytest
from attr import dataclass

from mteb.languages import (
    LanguageScripts,
    _hf_lang_to_iso_lang_script,
    _hf_langs_to_iso_lang_scripts,
)


@dataclass
class LangScriptTestCase:
    args: dict
    contains_language: list[str]
    not_contains_language: list[str]
    contains_script: list[str]
    not_contains_script: list[str]


test_cases = [
    LangScriptTestCase(
        args={"languages": ["fra"], "scripts": None},
        contains_language=["fra", "fra-Latn"],
        not_contains_language=["eng"],
        contains_script=[],
        not_contains_script=["Latn"],
    ),
    LangScriptTestCase(
        args={"languages": ["fra", "eng"], "scripts": ["Latn"]},
        contains_language=["fra", "fra-Latn", "eng", "eng-Latn"],
        not_contains_language=["deu"],
        contains_script=["Latn"],
        not_contains_script=["Cyrl"],
    ),
    LangScriptTestCase(
        args={"languages": ["fra-Latn"]},
        contains_language=["fra", "fra-Latn"],
        not_contains_language=["eng", "eng-Latn"],
        contains_script=["Latn"],
        not_contains_script=["Cyrl"],
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_langscripts(test_case: LangScriptTestCase):
    langscripts = LanguageScripts.from_languages_and_scripts(**test_case.args)

    for lang in test_case.contains_language:
        assert langscripts.contains_language(lang)

    for lang in test_case.not_contains_language:
        assert not langscripts.contains_language(lang)

    for script in test_case.contains_script:
        assert langscripts.contains_script(script)

    for script in test_case.not_contains_script:
        assert not langscripts.contains_script(script)


class TestHfLangToIsoLangScript:
    """Tests for converting HuggingFace language codes to MTEB ISOLanguageScript format."""

    @pytest.mark.parametrize(
        "hf_lang, expected",
        [
            ("en", "eng-Latn"),
            ("fr", "fra-Latn"),
            ("de", "deu-Latn"),
            ("zh", "zho-Hans"),
            ("ja", "jpn-Jpan"),
            ("ko", "kor-Hang"),
            ("ar", "ara-Arab"),
            ("hi", "hin-Deva"),
            ("ru", "rus-Cyrl"),
            ("no", "nor-Latn"),
            ("nb", "nob-Latn"),
            ("nn", "nno-Latn"),
        ],
    )
    def test_iso_639_1_codes(self, hf_lang, expected):
        assert _hf_lang_to_iso_lang_script(hf_lang) == expected

    @pytest.mark.parametrize(
        "hf_lang, expected",
        [
            ("eng", "eng-Latn"),
            ("fra", "fra-Latn"),
            ("zho", "zho-Hans"),
            ("ara", "ara-Arab"),
        ],
    )
    def test_iso_639_3_codes(self, hf_lang, expected):
        assert _hf_lang_to_iso_lang_script(hf_lang) == expected

    @pytest.mark.parametrize(
        "hf_lang",
        ["multilingual", "code", "mixed", "other", "unknown"],
    )
    def test_special_values_return_none(self, hf_lang):
        assert _hf_lang_to_iso_lang_script(hf_lang) is None

    def test_already_formatted(self):
        assert _hf_lang_to_iso_lang_script("eng-Latn") == "eng-Latn"

    def test_unknown_code_returns_none(self):
        assert _hf_lang_to_iso_lang_script("xx") is None
        assert _hf_lang_to_iso_lang_script("zzz") is None
        assert _hf_lang_to_iso_lang_script("invalid") is None

    def test_whitespace_handling(self):
        assert _hf_lang_to_iso_lang_script("  en  ") == "eng-Latn"


class TestHfLangsToIsoLangScripts:
    """Tests for converting lists of HuggingFace language codes."""

    def test_none_input(self):
        assert _hf_langs_to_iso_lang_scripts(None) is None

    def test_single_string(self):
        assert _hf_langs_to_iso_lang_scripts("en") == ["eng-Latn"]

    def test_list_input(self):
        result = _hf_langs_to_iso_lang_scripts(["en", "fr", "de"])
        assert result == ["deu-Latn", "eng-Latn", "fra-Latn"]

    def test_filters_special_values(self):
        result = _hf_langs_to_iso_lang_scripts(["en", "multilingual", "fr"])
        assert result == ["eng-Latn", "fra-Latn"]

    def test_all_invalid_returns_none(self):
        assert _hf_langs_to_iso_lang_scripts(["multilingual", "unknown"]) is None

    def test_deduplication(self):
        result = _hf_langs_to_iso_lang_scripts(["en", "en", "en"])
        assert result == ["eng-Latn"]

    def test_sorted_output(self):
        result = _hf_langs_to_iso_lang_scripts(["zh", "en", "ar"])
        assert result == ["ara-Arab", "eng-Latn", "zho-Hans"]
