"""Rewrite BibTeX citation keys to the Better BibTeX (Zotero) format.

Key format (Better BibTeX formula):
    auth.transliterate.clean.lower + "_" + veryshorttitle.transliterate.clean.lower + "_" + year

Only the key in the `@type{key,` header is changed; the rest of the source is left as is.
Dry run by default, pass `--write` to modify files.

Usage:
    python scripts/update_citation_keys.py [--write] [paths ...]
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import bibtexparser
from pylatexenc.latex2text import LatexNodes2Text

ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,")
NON_ENTRY_TYPES = {"comment", "string", "preamble"}

# Better BibTeX default skipWords
SKIP_WORDS = {
    "a",
    "ab",
    "aboard",
    "about",
    "above",
    "across",
    "after",
    "against",
    "al",
    "along",
    "amid",
    "among",
    "an",
    "and",
    "anti",
    "around",
    "as",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "but",
    "by",
    "d",
    "da",
    "das",
    "de",
    "del",
    "dell",
    "dello",
    "dei",
    "degli",
    "della",
    "delle",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dos",
    "du",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "el",
    "en",
    "et",
    "except",
    "for",
    "from",
    "gli",
    "i",
    "il",
    "in",
    "inside",
    "into",
    "is",
    "l",
    "la",
    "las",
    "le",
    "les",
    "like",
    "lo",
    "los",
    "near",
    "nor",
    "of",
    "off",
    "on",
    "onto",
    "or",
    "over",
    "past",
    "per",
    "plus",
    "round",
    "save",
    "since",
    "so",
    "some",
    "sur",
    "than",
    "the",
    "through",
    "to",
    "toward",
    "towards",
    "un",
    "una",
    "unas",
    "under",
    "underneath",
    "une",
    "unlike",
    "uno",
    "unos",
    "until",
    "up",
    "upon",
    "versus",
    "via",
    "von",
    "while",
    "with",
    "within",
    "without",
    "yet",
    "zu",
    "zum",
}
NAME_PARTICLES = {
    "van",
    "von",
    "der",
    "den",
    "de",
    "del",
    "della",
    "di",
    "da",
    "du",
    "la",
    "le",
    "ten",
    "ter",
    "bin",
    "al",
}
TITLE_SPLIT = re.compile(r"[.:!?]")


def latex_to_text(s: str) -> str:
    return LatexNodes2Text().latex_to_text(s).replace("{", "").replace("}", "")


def transliterate(s: str) -> str:
    s = s.replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("Ł", "L")
    s = s.replace("ß", "ss").replace("æ", "ae").replace("Æ", "AE").replace("đ", "d")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def clean(s: str) -> str:
    """Keep only characters that are safe in a citation key."""
    return re.sub(r"[^A-Za-z0-9\-]", "", s)


def first_author_family(authors: str) -> str:
    first = re.split(r"\s+and\s+", authors.strip(), maxsplit=1)[0].strip()
    if first.lower() == "others":
        return ""
    if first.startswith("{") and first.endswith("}"):  # corporate author
        return latex_to_text(first)
    if "," in first:
        return latex_to_text(first.split(",", 1)[0])
    tokens = latex_to_text(first).split()
    if not tokens:
        return ""
    family = [tokens[-1]]
    for tok in reversed(tokens[:-1]):  # prepend particles: "van der Berg"
        if tok.lower() in NAME_PARTICLES:
            family.insert(0, tok)
        else:
            break
    return " ".join(family)


def very_short_title(title: str) -> str:
    main = TITLE_SPLIT.split(latex_to_text(title).strip(), maxsplit=1)[0]
    words = [re.sub(r"[^\w\-]", "", w) for w in main.split()]
    words = [w for w in words if w]
    for w in words:
        if w.lower() not in SKIP_WORDS:
            return w
    return words[0] if words else ""


def make_key(fields: dict[str, str]) -> str | None:
    authors = fields.get("author") or fields.get("editor") or ""
    title = fields.get("title", "")
    year = re.search(r"\d{4}", fields.get("year", "") or fields.get("date", ""))
    if not (authors and title and year):
        return None
    auth = clean(transliterate(first_author_family(authors))).lower()
    short = clean(transliterate(very_short_title(title))).lower()
    if not auth or not short:
        return None
    return f"{auth}_{short}_{year.group()}"


def find_entries(text: str):
    """Yield (old_key, key_span, entry_text) for each BibTeX entry in the text."""
    for m in ENTRY_START.finditer(text):
        if m.group(1).lower() in NON_ENTRY_TYPES:
            continue
        depth, end = 0, None
        for i in range(text.index("{", m.start()), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        yield m.group(2), m.span(2), text[m.start() : end]


def parse_fields(entry_text: str) -> dict[str, str] | None:
    # the python source may contain escaped quotes/backslashes; undo the common ones
    lib = bibtexparser.parse_string(entry_text)
    if not lib.entries or lib.failed_blocks:
        return None
    return {k.lower(): f.value for k, f in lib.entries[0].fields_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=[Path("mteb")], type=Path)
    parser.add_argument("--write", action="store_true", help="modify files in place")
    args = parser.parse_args()

    files = sorted(
        f
        for p in args.paths
        for f in ([p] if p.is_file() else p.rglob("*"))
        if f.suffix in {".py", ".bib", ".md"} and f.is_file()
    )

    # pass 1: (old key, title) -> proposed key
    proposed: dict[tuple[str, str], str] = {}
    skipped: dict[str, str] = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        for old, _, entry in find_entries(text):
            fields = parse_fields(entry)
            ident = (old, _norm_title(fields))
            if ident in proposed:
                continue
            new = make_key(fields) if fields else None
            if new is None:
                skipped[old] = str(f)
            else:
                proposed[ident] = new

    # Same generated key + same title is the same paper (merge); same key with
    # a different title is a different paper, so add a/b/... postfix like Better BibTeX.
    by_new = defaultdict(set)
    for (_old, title), new in proposed.items():
        by_new[new].add(title)
    mapping: dict[tuple[str, str], str] = {}
    for (old, title), new in proposed.items():
        titles = sorted(by_new[new])
        i = titles.index(title)
        mapping[(old, title)] = new if i == 0 else f"{new}{_letters(i)}"

    # pass 2: rewrite (right to left so spans stay valid)
    changed_files = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        edits = []
        for old, span, entry in find_entries(text):
            new = mapping.get((old, _norm_title(parse_fields(entry))))
            if new and new != old:
                edits.append((span, new))
        if not edits:
            continue
        changed_files += 1
        for (start, end), new in sorted(edits, reverse=True):
            text = text[:start] + new + text[end:]
        if args.write:
            f.write_text(text, encoding="utf-8")

    renamed = {o: n for (o, _), n in mapping.items() if o != n}
    for old, new in sorted(renamed.items()):
        print(f"{old} -> {new}")
    print(
        f"\n{len(renamed)} keys renamed in {changed_files} files "
        f"({'written' if args.write else 'dry run, use --write to apply'}); "
        f"{len(skipped)} keys skipped (missing author/title/year):"
    )
    for old, f in sorted(skipped.items()):
        print(f"  skipped {old} ({f})")


def _norm_title(fields: dict[str, str] | None) -> str:
    title = latex_to_text((fields or {}).get("title", ""))
    return re.sub(r"\W+", " ", title).strip().lower()


def _letters(i: int) -> str:
    """1 -> a, 2 -> b, ..., 26 -> z, 27 -> aa"""
    s = ""
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


if __name__ == "__main__":
    main()
