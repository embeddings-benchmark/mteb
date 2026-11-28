"""Tests for the citation readers shared by the citation checks."""

from __future__ import annotations

import pytest

from scripts.tools.citation_utils import (
    Entry,
    clean_arxiv_id,
    clean_doi,
    find_arxiv_id,
    first_surname,
    fold,
    parse_file,
    parse_string,
    similarity,
    strip_tex,
    surnames_match,
    title_tokens,
)

pytestmark = pytest.mark.core


def test_parse_string_reads_nested_values():
    bibtex = """@article{cited2024,
        title = {The {LLM} Era: A {Survey}},
        author = {Doe, Jane and Roe, John},
        year = {2024},
    }"""
    assert parse_string(bibtex) == [
        Entry(
            key="cited2024",
            entry_type="article",
            fields={
                "title": "The {LLM} Era: A {Survey}",
                "author": "Doe, Jane and Roe, John",
                "year": "2024",
            },
        )
    ]


def test_parse_string_reads_quoted_and_bare_values():
    bibtex = """@misc{quoted,
        title = "A quoted, {protected} title",
        doi = 10.1234/example,
        note = {a note with an escaped \\} brace}
    }"""
    (entry,) = parse_string(bibtex)
    assert entry.fields["title"] == "A quoted, {protected} title"
    assert entry.fields["doi"] == "10.1234/example"
    assert entry.fields["note"] == "a note with an escaped \\} brace"


def test_parse_string_skips_non_entries_and_reads_parentheses():
    bibtex = """@comment{note, this = is not an entry}
    @string{open = "{"}
    @inproceedings(parenthesised,
        title = {A title},
    )"""
    assert [entry.key for entry in parse_string(bibtex)] == ["parenthesised"]


def test_parse_file(tmp_path):
    path = tmp_path / "refs.bib"
    path.write_text("@article{a, title = {A title}}", encoding="utf-8")
    assert [entry.key for entry in parse_file(path)] == ["a"]


def test_entry_fields_and_year():
    (entry,) = parse_string("@article{a, title = {T}, year = {2023 in press}}")
    assert entry.title == "T"
    assert not entry.doi
    assert entry.year == 2023
    (other,) = parse_string("@misc{b, title = {T}}")
    assert other.year is None


def test_entry_arxiv_id_from_own_field_or_free_text():
    (explicit,) = parse_string("@misc{a, eprint = {2401.12345v2}}")
    assert clean_arxiv_id(explicit.arxiv_id) == "2401.12345"

    (in_journal,) = parse_string(
        "@article{b, journal = {arXiv preprint arXiv:2506.08872}}"
    )
    assert in_journal.arxiv_id == "2506.08872"

    (in_note,) = parse_string("@misc{c, note = {See https://arxiv.org/abs/2301.00123}}")
    assert in_note.arxiv_id == "2301.00123"

    (unrelated,) = parse_string("@article{d, journal = {Nature}}")
    assert not unrelated.arxiv_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1234/Abc", "10.1234/Abc"),
        ("doi: 10.1234/Abc", "10.1234/Abc"),
        ("https://doi.org/10.1234/Abc", "10.1234/Abc"),
        ("http://dx.doi.org/10.1234/Abc.", "10.1234/Abc"),
        ("doi.org/10.1234/Abc,", "10.1234/Abc"),
        ("https://example.org/10.1234/Abc", ""),
        ("10.123/Abc", ""),
        ("not a doi", ""),
        ("", ""),
    ],
)
def test_clean_doi(raw, expected):
    assert clean_doi(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2401.12345v3", "2401.12345"),
        ("arXiv:2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v2", "2401.12345"),
        ("math.GT/0309136v1", "math.GT/0309136"),
        ("not-an-id", ""),
        ("", ""),
    ],
)
def test_clean_arxiv_id(raw, expected):
    assert clean_arxiv_id(raw) == expected


def test_find_arxiv_id_in_free_text():
    assert find_arxiv_id("arXiv preprint arXiv:2506.08872") == "2506.08872"
    assert find_arxiv_id("doi: 10.48550/arXiv.2401.12345") == "2401.12345"
    assert find_arxiv_id("https://arxiv.org/pdf/2301.00123v2") == "2301.00123"
    assert not find_arxiv_id("Published in Nature")


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        ("Ferguson, Sharon", "ferguson"),
        ("Sharon Ferguson and John Doe", "ferguson"),
        (r"Sharon {\emph{Ferguson}} and John Doe", "ferguson"),
        ("van der Berg, Jane", "van der berg"),
        ("", ""),
    ],
)
def test_first_surname(authors, expected):
    assert first_surname(authors) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("ferguson", "ferguson", True),
        ("ferguson", "", True),
        ("van der berg", "berg", True),
        ("al rashid", "al-rashid", True),
        ("smith", "jones", False),
    ],
)
def test_surnames_match(a, b, expected):
    assert surnames_match(a, b) is expected


def test_strip_tex_and_fold():
    assert strip_tex(r"\emph{Important} $x + y$").strip() == "Important"
    assert fold("Café {Ö}") == "cafe o"
    assert title_tokens("A Survey of {Large Language} Models") == [
        "survey",
        "large",
        "language",
        "models",
    ]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("The {LLM} Era", "the llm era", 1.0),
        ("Attention Is All You Need:\nA Study", "attention is all you need", 0.95),
        ("A Study of Things", "A Study of Other Things Entirely", None),
        ("Cat", "Dog", 0.0),
        ("", "A title", 0.0),
    ],
)
def test_similarity(a, b, expected):
    score = similarity(a, b)
    if expected is None:
        # Shared words in a different order must not count as a match.
        assert score < 0.9
    else:
        assert score == expected
