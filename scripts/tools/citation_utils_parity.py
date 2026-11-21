"""Compare the local citation helpers with refaudit on this repository's citations.

A development check, not a CI gate: it imports refaudit, the dependency that
``citation_utils.py`` exists to replace. Run it from the repository root with
``make citation-utils-parity``. It exits non-zero on any difference.
"""

from __future__ import annotations

import ast
from pathlib import Path

from refaudit import parse_string as refaudit_parse
from refaudit.normalize import (
    clean_arxiv_id as refaudit_clean_arxiv_id,
)
from refaudit.normalize import (
    clean_doi as refaudit_clean_doi,
)
from refaudit.normalize import (
    find_arxiv_id as refaudit_find_arxiv_id,
)
from refaudit.normalize import (
    first_surname as refaudit_first_surname,
)
from refaudit.normalize import (
    fold as refaudit_fold,
)
from refaudit.normalize import (
    similarity as refaudit_similarity,
)
from refaudit.normalize import (
    strip_tex as refaudit_strip_tex,
)
from refaudit.normalize import (
    surnames_match as refaudit_surnames_match,
)
from refaudit.normalize import (
    title_tokens as refaudit_title_tokens,
)

from scripts.tools import citation_utils as local

REPO = Path(__file__).parents[2]
FIELD_NAMES = (
    "title",
    "author",
    "doi",
    "eprint",
    "journal",
    "note",
    "booktitle",
    "url",
    "year",
)
MAX_REPORTED = 10


def citation_sources() -> list[tuple[Path, str]]:
    """BibTeX-bearing strings in mteb, read without importing the library."""
    sources: list[tuple[Path, str]] = []
    for path in sorted((REPO / "mteb").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if "@" in value and "{" in value and "title" in value:
                    sources.append((path, value))
    return sources


def entry_row(
    entry: object,
) -> tuple[str, str, dict[str, str], str, str, str, int | None]:
    return (
        entry.key,
        entry.entry_type,
        entry.fields,
        entry.title,
        entry.doi,
        entry.arxiv_id,
        entry.year,
    )


def compare(sources: list[tuple[Path, str]]) -> list[str]:
    """Differences between the two implementations, capped for readability."""
    differences: list[str] = []
    values: list[str] = []
    for path, source in sources:
        upstream = refaudit_parse(source)
        local_rows = [entry_row(entry) for entry in local.parse_string(source)]
        if local_rows != [entry_row(entry) for entry in upstream]:
            differences.append(f"{path.relative_to(REPO)}: parse_string differs")
        for entry in upstream:
            values.extend(entry.get(name) for name in FIELD_NAMES)

    normalizers = (
        ("strip_tex", local.strip_tex, refaudit_strip_tex),
        ("fold", local.fold, refaudit_fold),
        ("title_tokens", local.title_tokens, refaudit_title_tokens),
        ("clean_doi", local.clean_doi, refaudit_clean_doi),
        ("clean_arxiv_id", local.clean_arxiv_id, refaudit_clean_arxiv_id),
        ("find_arxiv_id", local.find_arxiv_id, refaudit_find_arxiv_id),
        ("first_surname", local.first_surname, refaudit_first_surname),
    )
    for name, local_fn, upstream_fn in normalizers:
        for value in values:
            if local_fn(value) != upstream_fn(value):
                differences.append(f"{name}({value!r})")
                break

    for a, b in zip(values[::3], values[1::3], strict=False):
        if local.similarity(a, b) != refaudit_similarity(a, b):
            differences.append(f"similarity({a!r}, {b!r})")
            break
    surnames = [refaudit_first_surname(value) for value in values]
    for a, b in zip(surnames[::3], surnames[1::3], strict=False):
        if local.surnames_match(a, b) != refaudit_surnames_match(a, b):
            differences.append(f"surnames_match({a!r}, {b!r})")
            break
    return differences


def main() -> int:
    sources = citation_sources()
    differences = compare(sources)
    print(f"compared {len(sources)} BibTeX strings from mteb/")
    if differences:
        for difference in differences[:MAX_REPORTED]:
            print(f"  DIFF {difference}")
        if len(differences) > MAX_REPORTED:
            print(f"  ... and {len(differences) - MAX_REPORTED} more")
        return 1
    print("citation helpers match refaudit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
