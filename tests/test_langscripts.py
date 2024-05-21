from __future__ import annotations

import pytest
from attr import dataclass

from mteb.languages import LanguageScripts


@dataclass
class LangScriptTestCase:
    args: dict
    contains_language: list[str]
    not_contains_language: list[str]
    contains_script: list[str]
    not_contains_script: list[str]


test_cases = [
    LangScriptTestCase(
        args=dict(languages=["fra"], scripts=None),
        contains_language=["fra", "fra-Latn"],
        not_contains_language=["eng"],
        contains_script=[],
        not_contains_script=["Latn"],
    ),
    LangScriptTestCase(
        args=dict(languages=["fra", "eng"], scripts=["Latn"]),
        contains_language=["fra", "fra-Latn", "eng", "eng-Latn"],
        not_contains_language=["deu"],
        contains_script=["Latn"],
        not_contains_script=["Cyrl"],
    ),
    LangScriptTestCase(
        args=dict(languages=["fra-Latn"]),
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
