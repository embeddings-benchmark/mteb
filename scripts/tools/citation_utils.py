"""Shared BibTeX reading and bibliographic normalisation for the citation checks.

``check_citations.py`` and ``check_preprints.py`` both read citations from
BibTeX and compare titles, authors, DOIs and arXiv identifiers. Keeping those
readers in one module means the two checks cannot drift apart, and lets them be
tested without importing either network-facing script.

The comparison rules mirror refaudit 0.4.9 (MIT, Copyright (c) 2026 SHARE Lab,
University of Waterloo): cached verdicts in ``citation_cache.json`` were
produced with those rules, so changing them here would invalidate every cached
result. The index lookups, checker and cache are still supplied by refaudit;
this module is the first step towards replacing that dependency (mteb#5849).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# --- BibTeX reading --------------------------------------------------------

_ENTRY_START = re.compile(
    r"@(?P<type>\w+)\s*[{(]\s*(?P<key>[^,\s{}()]+)\s*,", re.MULTILINE
)
_FIELD_START = re.compile(r"(\w+)\s*=\s*")
_NON_ENTRIES = frozenset({"comment", "preamble", "string"})


@dataclass(frozen=True)
class Entry:
    """One bibliography entry, with its fields kept as the source wrote them."""

    key: str
    entry_type: str
    fields: dict[str, str] = field(default_factory=dict)

    def get(self, name: str, default: str = "") -> str:
        return self.fields.get(name.lower(), default)

    @property
    def title(self) -> str:
        return self.get("title")

    @property
    def doi(self) -> str:
        return self.get("doi")

    @property
    def arxiv_id(self) -> str:
        """The arXiv id, from its own field or from free text.

        ``eprint`` is the proper field, but Google Scholar exports leave the id
        in ``journal`` or ``note`` instead. Those entries are still findable, so
        the other fields are searched before giving up.
        """
        explicit = self.get("eprint") or self.get("archiveprefix_id")
        if clean_arxiv_id(explicit):
            return explicit
        for name in ("journal", "note", "howpublished", "booktitle", "url", "doi"):
            if found := find_arxiv_id(self.get(name)):
                return found
        return explicit

    @property
    def year(self) -> int | None:
        match = re.search(r"\d{4}", self.get("year"))
        return int(match.group()) if match else None


def _matching_close(source: str, start: int) -> int:
    """Index of the delimiter closing the one at ``start``, or ``len(source)``."""
    opener = source[start]
    closer = "}" if opener == "{" else ")"
    depth = 0
    i = start
    while i < len(source):
        if source[i] == "\\":  # an escaped delimiter is not structural
            i += 2
            continue
        if source[i] == opener:
            depth += 1
        elif source[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(source)


def _read_value(source: str, start: int) -> tuple[str, int]:
    """Read one field value, returning it and the position just after it."""
    i = start
    while i < len(source) and source[i].isspace():
        i += 1
    if i >= len(source):
        return "", i
    if source[i] in "{(":
        end = _matching_close(source, i)
        return source[i + 1 : end], min(end + 1, len(source))
    if source[i] == '"':
        content_start = i + 1
        i = content_start
        while i < len(source):
            if source[i] == "\\":
                i += 2
                continue
            if source[i] == '"':
                return source[content_start:i], i + 1
            i += 1
        return source[content_start:i], i
    # A bare value runs to the next comma, closing brace or newline.
    end = i
    while end < len(source) and source[end] not in ",}\n":
        end += 1
    return source[i:end].strip(), end


def parse_string(source: str) -> list[Entry]:
    """Read the BibTeX entries in ``source``, in order.

    Only entry type, cite key and raw field strings are kept. Macros and
    crossrefs are not expanded: a checker must report what the citation
    actually says rather than silently guessing what a macro might mean.
    """
    entries: list[Entry] = []
    for match in _ENTRY_START.finditer(source):
        entry_type = match.group("type").lower()
        if entry_type in _NON_ENTRIES:
            continue
        opening = next(
            (i for i in range(match.start(), match.end()) if source[i] in "{("),
            None,
        )
        if opening is None:
            continue
        closing = _matching_close(source, opening)
        body = source[opening + 1 : closing]

        fields: dict[str, str] = {}
        position = 0
        while field_match := _FIELD_START.search(body, position):
            value, position = _read_value(body, field_match.end())
            fields[field_match.group(1).lower()] = " ".join(value.split())
        entries.append(Entry(match.group("key").strip(), entry_type, fields))
    return entries


def parse_file(path: str | Path) -> list[Entry]:
    """Read the BibTeX entries in a file."""
    return parse_string(Path(path).read_text(encoding="utf-8", errors="replace"))


# --- Normalisation ---------------------------------------------------------

_TEX_MATH = re.compile(r"\$[^$]*\$")
_TEX_COMMAND = re.compile(r"\\[a-zA-Z]+\s*")
_BRACES = re.compile(r"[{}]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_AUTHOR_SEPARATOR = re.compile(r"\s+and\s+")

# Words that do not distinguish one title from another.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "with",
        "without",
        "via",
        "using",
        "toward",
        "towards",
    }
)


def strip_tex(value: str) -> str:
    """Drop TeX math and commands, and unwrap brace protection."""
    text = _TEX_MATH.sub(" ", value or "")
    text = _TEX_COMMAND.sub(" ", text)
    return _BRACES.sub("", text)


def fold(value: str) -> str:
    """Lowercase, de-accent, strip punctuation and collapse whitespace."""
    text = unicodedata.normalize("NFKD", strip_tex(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return _WHITESPACE.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def title_tokens(value: str) -> list[str]:
    """The content words of a title, without stopwords."""
    return [token for token in fold(value).split() if token not in _STOPWORDS]


def similarity(a: str, b: str) -> float:
    """How alike two titles are, on a scale from 0 to 1.

    A sequence ratio catches typos and word-order changes. A second signal
    allows the shorter title to be a leading run of the longer one, which is
    the common case of a dropped subtitle. Plain containment is deliberately
    not enough: two different papers can share nearly all their words.
    """
    left, right = fold(a), fold(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    score = difflib.SequenceMatcher(None, left, right).ratio()
    tokens_a, tokens_b = title_tokens(a), title_tokens(b)
    if not tokens_a or not tokens_b:
        return score

    shorter, longer = (
        (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    )
    # Require a few content words before a prefix match can override the ratio.
    if len(shorter) >= 3 and longer[: len(shorter)] == shorter:
        return max(score, 0.95)
    return score


def first_surname(author_field: str) -> str:
    """Surname of the first author, in either BibTeX author convention."""
    if not author_field:
        return ""
    first = _AUTHOR_SEPARATOR.split(strip_tex(author_field).strip())[0].strip()
    if not first:
        return ""
    if "," in first:  # "Ferguson, Sharon"
        return fold(first.split(",", 1)[0])
    parts = fold(first).split()  # "Sharon Ferguson"
    return parts[-1] if parts else ""


def surnames_match(a: str, b: str) -> bool:
    """Tolerant comparison of two surnames, or True if either is unknown."""
    if not a or not b or a == b:
        return True
    if a.rsplit(maxsplit=1)[-1] == b.rsplit(maxsplit=1)[-1]:  # van der Berg / Berg
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


_DOI_LABEL = re.compile(r"^\s*(?:doi\s*:\s*)?", re.IGNORECASE)
_DOI_URL = re.compile(r"^(?:https?://)?(?:www\.)?(?:dx\.)?doi\.org/", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9<>\[\]+]+", re.IGNORECASE)


def clean_doi(raw: str) -> str:
    """The bare DOI in ``raw``, or "" if it is not a usable DOI.

    Validating before the value is put in a request path keeps a malformed
    field from steering the lookup somewhere unintended. URLs are accepted too:
    dropping them would downgrade an identifier lookup to a weaker title one.
    """
    text = strip_tex(raw or "").strip()
    text = _DOI_LABEL.sub("", text)
    text = _DOI_URL.sub("", text).strip().rstrip(".,;")
    return text if _DOI_PATTERN.fullmatch(text) else ""


_ARXIV_LABEL = re.compile(r"^arxiv\s*:\s*", re.IGNORECASE)
_ARXIV_URL = re.compile(r"^https?://arxiv\.org/abs/", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})", re.IGNORECASE
)


def clean_arxiv_id(raw: str) -> str:
    """The bare arXiv id in ``raw``, without a version suffix, or ""."""
    text = strip_tex(raw or "").strip()
    text = _ARXIV_LABEL.sub("", text)
    text = _ARXIV_URL.sub("", text)
    text = _ARXIV_VERSION.sub("", text.strip())
    return text if _ARXIV_PATTERN.fullmatch(text) else ""


# An arXiv id in free text; the most explicit form is tried first.
_ARXIV_IN_TEXT = (
    re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s,;}]+)", re.IGNORECASE),
    re.compile(r"10\.48550/arxiv\.([^\s,;}]+)", re.IGNORECASE),
    re.compile(r"arxiv[:\s]+([^\s,;}]+)", re.IGNORECASE),
)


def find_arxiv_id(text: str) -> str:
    """Find an arXiv id inside a free-text field, or return "".

    Some exports write ``journal={arXiv preprint arXiv:2506.08872}`` instead
    of using ``eprint``. Reading only ``eprint`` would report findable preprints
    as unverifiable.
    """
    if not text:
        return ""
    cleaned = strip_tex(text)
    for pattern in _ARXIV_IN_TEXT:
        for match in pattern.finditer(cleaned):
            if found := clean_arxiv_id(match.group(1)):
                return found
    return ""
