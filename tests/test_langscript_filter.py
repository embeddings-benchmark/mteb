

from mteb.abstasks.languages import LangScriptFilter

def test_langscript_filter():

    ls_filter = LangScriptFilter.from_languages_and_scripts(
        languages=["fra"],
        scripts=None
    )

    assert ls_filter.is_valid_language("fra")
    assert ls_filter.is_valid_language("fra-Latn")
    assert not ls_filter.is_valid_language("eng")

