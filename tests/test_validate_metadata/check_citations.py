"""Check that the works cited by mteb exist and are cited correctly.

Every task, benchmark and model may cite using a BibTeX. A
citation that does not resolve to the work it claims - to prevent
fabricated citations from AI usage, draft, or similarly incorrect citations
this CI test checks that the cited work exists and is cited correctly.
See original issue: https://github.com/embeddings-benchmark/mteb/issues/5471.

The entries are checked with [refaudit](https://pypi.org/project/refaudit/),
which looks each one up by DOI, arXiv id or title in Crossref, DataCite, arXiv,
DBLP and OpenAlex.

Run it with `make citation-check`, or directly:

    python tests/test_validate_metadata/check_citations.py [--recheck-days 3650]

Results are cached in `citation_cache.json`, which is checked into the repository:
an entry checked less than `--recheck-days` days ago is not looked up again, so a
run normally only checks the citations that were added since. The script updates
that file, so commit it together with your citation.

Citations that cannot be verified but are correct - a workshop paper with no DOI,
a dataset card, a thesis - are listed in `citation_allowlist.json` with the reason
they are accepted. The list is meant to shrink: when you fix a citation, remove
its key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from refaudit import Cache, Checker, Entry, Verdict, default_resolvers, parse_string

import mteb

CACHE_PATH = Path(__file__).parent / "citation_cache.json"
ALLOWLIST_PATH = Path(__file__).parent / "citation_allowlist.json"

CONTACT_EMAIL = os.environ.get("MTEB_CONTACT_EMAIL", "kenneth.enevoldsen@cas.au.dk")
"""Crossref, DataCite and OpenAlex give identified callers a more reliable request pool."""

FLUSH_EVERY = 25
"""A full run takes a while; keep what it has found so far so it can be resumed."""


def citations() -> dict[str, tuple[Entry, set[str]]]:
    """Every distinct BibTeX entry in the library, with the names citing it.

    A task, benchmark or model may cite several works in one string; each entry in
    it is checked on its own.
    """
    cited_works: list[tuple[str, str | None]] = [
        (task.metadata.name, task.metadata.bibtex_citation)
        for task in mteb.get_tasks(
            exclude_superseded=False, exclude_aggregate=False, exclude_beta=False
        )
    ]
    cited_works += [
        (benchmark.name, benchmark.citation) for benchmark in mteb.get_benchmarks()
    ]
    cited_works += [(model.name, model.citation) for model in mteb.get_model_metas()]

    entries: dict[str, tuple[Entry, set[str]]] = {}
    for name, bibtex in cited_works:
        for entry in parse_string(bibtex or ""):
            _, cited_by = entries.setdefault(entry.key, (entry, set()))
            cited_by.add(name)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recheck-days",
        type=float,
        default=30.0,
        help="look a citation up again once its cached result is this old (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="citations to check at a time (default: 4)",
    )
    args = parser.parse_args()

    entries = citations()
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    to_check = [entry for key, (entry, _) in entries.items() if key not in allowlist]

    cache = Cache(CACHE_PATH, ttl_days=args.recheck_days)
    checker = Checker(default_resolvers(CONTACT_EMAIL), cache=cache)
    print(
        f"checking {len(to_check)} citations "
        f"({len(entries) - len(to_check)} allowlisted), this can take a while",
        flush=True,
    )

    results = []
    for result in checker.check_all(to_check, workers=args.workers):
        results.append(result)
        if len(results) % FLUSH_EVERY == 0:
            cache.flush()
            print(f"  {len(results)}/{len(to_check)}", flush=True)
    cache.flush()

    print(
        "\n"
        + ", ".join(
            f"{verdict.value}: {n}"
            for verdict, n in Counter(r.verdict for r in results).most_common()
        )
    )

    # SKIPPED is refaudit abstaining on an entry with nothing to look up; we report it
    findings = sorted(
        (r for r in results if r.verdict.is_finding or r.verdict is Verdict.SKIPPED),
        key=lambda r: (r.verdict.value, r.key),
    )
    for result in findings:
        cited_by = ", ".join(sorted(entries[result.key][1])[:3])
        print(
            f"\n{result.key} [{result.verdict.value}] cited by {cited_by}"
            f"\n  cited: {result.entry_title}"
            f"\n  found: {result.found_title or '-'} ({result.source})"
            f"\n  {result.note}"
        )

    if not findings:
        print("\nevery citation checks out")
        return 0
    print(
        f"\n{len(findings)} citations could not be verified. Please check them against the "
        "original publication - a dataset usually states how it wants to be cited on its "
        "landing page or in its README. If a work is cited correctly but is simply not "
        f"indexed by any of the sources, add its key to {ALLOWLIST_PATH.name} with the reason."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
