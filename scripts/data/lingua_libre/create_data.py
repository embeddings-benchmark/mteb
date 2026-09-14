#!/usr/bin/env python3
"""Build the Lingua Libre spoken-word retrieval tasks for MTEB.

Source is Lingua Libre, a Wikimedia project where volunteers record individual words, with
the recordings hosted on Wikimedia Commons. Commons is free by site policy and every file
sampled here is CC-BY-SA 4.0 or CC0, so the published set is CC-BY-SA 4.0.

Commons files it under `Category:Lingua Libre pronunciation-<iso 639-3>`, one category per
language, which is what makes a wide multilingual build tractable: the language code comes
straight from the category name rather than having to be inferred. There are over 340 such
categories. Languages are chosen by a fixed rule, described in `select_languages`.

The word and the speaker both sit in the filename, `LL-Q150 (fra)-Speaker-word.wav`, so the
label needs no annotation. Three filters apply, because the categories hold more than
single words:

- entries with no letters are dropped, which removes recordings of bare punctuation such
  as `!` or `$`;
- entries containing whitespace are dropped, which removes the read sentences that some
  contributors upload alongside words;
- a word is kept once, since the same word read by two speakers would otherwise be
  relevant to only one of its recordings.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/lingua_libre/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/lingua_libre/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from math import gcd
from pathlib import Path

import pyarrow.parquet as pq
import requests
import soundfile as sf
from datasets import Audio, Dataset
from huggingface_hub import HfApi
from scipy.signal import resample_poly

_TARGET = "vnahata/LinguaLibre-word-retrieval"
_LICENSE = "cc-by-sa-4.0"
_UA = "mteb-dataset-builder/1.0 (https://github.com/embeddings-benchmark/mteb)"
_COMMONS = "https://commons.wikimedia.org/w/api.php"

_N_LANGUAGES = 40
_MIN_FILES = 300  # probed per category, so a language has enough words to rank
_PER_LANGUAGE = 150
_MIN_WORD_CHARS = 2
_MAX_WORD_CHARS = 30
_TARGET_RATE = 16000

_FILENAME = re.compile(r"^LL-Q\d+\s*\(([a-z]{2,3})\)-(.+?)-(.+)$")


def _get(params: dict) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(
                _COMMONS,
                params={**params, "format": "json"},
                headers={"User-Agent": _UA},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def _language_categories() -> list[str]:
    out, cont = [], {}
    while True:
        d = _get(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": "Category:Lingua Libre pronunciation",
                "cmtype": "subcat",
                "cmlimit": 500,
                **cont,
            }
        )
        for m in d.get("query", {}).get("categorymembers", []):
            name = m["title"].replace("Category:", "")
            if re.fullmatch(r"Lingua Libre pronunciation-[a-z]{3}", name):
                out.append(name)
        if "continue" not in d:
            return sorted(out)
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def _category_files(category: str, pages: int = 1) -> list[str]:
    """Listed newest first. Alphabetical order puts thousands of digits and symbols
    before the first real word, so sorting by upload time is what makes a page usable."""
    out, cont = [], {}
    for _ in range(pages):
        d = _get(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "file",
                "cmlimit": 500,
                "cmsort": "timestamp",
                "cmdir": "desc",
                **cont,
            }
        )
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        if "continue" not in d:
            break
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        time.sleep(0.2)
    return out


def _covered_languages() -> set[str]:
    """ISO codes already reachable through some mteb audio task."""
    import mteb

    covered = set()
    for task in mteb.get_tasks():
        if "audio" not in (task.metadata.modalities or []):
            continue
        langs = task.metadata.eval_langs
        values = (
            [v for vs in langs.values() for v in vs]
            if isinstance(langs, dict)
            else list(langs)
        )
        covered.update(v.split("-")[0] for v in values)
    return covered


def select_languages() -> list[str]:
    """Languages absent from every existing mteb audio task, with enough recordings.

    Half of Lingua Libre's larger categories are languages the benchmark already reaches,
    so taking them in plain ISO order spends most of the slots on coverage that exists.
    """
    covered = _covered_languages()
    chosen = []
    for category in _language_categories():
        code = category.rsplit("-", 1)[1]
        if code in covered:
            continue
        files = _category_files(category)
        if len(files) >= _MIN_FILES:
            chosen.append(code)
            print(f"    {code}: {len(files)}+ files", flush=True)
        if len(chosen) >= _N_LANGUAGES:
            break
        time.sleep(0.2)
    return chosen


def _words(files: list[str]) -> dict[str, str]:
    """Filename -> word, keeping one recording per word."""
    out, seen = {}, set()
    for f in files:
        stem = re.sub(
            r"\.(wav|ogg|oga|mp3|flac)$", "", f.replace("File:", ""), flags=re.I
        )
        m = _FILENAME.match(stem)
        if not m:
            continue
        word = m.group(3).strip()
        if not (_MIN_WORD_CHARS <= len(word) <= _MAX_WORD_CHARS):
            continue
        # A leading hyphen marks an affix such as "-able" rather than a word, whitespace
        # marks a read sentence, and an entry with no letters is bare punctuation.
        if not word[0].isalpha() or any(ch.isspace() for ch in word):
            continue
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        out[f] = word
    return out


def _to_opus(raw: bytes) -> bytes | None:
    data, rate = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if len(data) < rate * 0.2:
        return None
    if rate != _TARGET_RATE:
        factor = gcd(int(rate), _TARGET_RATE)
        data = resample_poly(data, _TARGET_RATE // factor, int(rate) // factor)
    buf = io.BytesIO()
    sf.write(buf, data, _TARGET_RATE, format="OGG", subtype="OPUS")
    return buf.getvalue()


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    codes_path = work / "languages.json"
    if codes_path.exists():
        codes = json.loads(codes_path.read_text(encoding="utf-8"))
    else:
        codes = select_languages()
        codes_path.write_text(json.dumps(codes), encoding="utf-8")
    print(f"selected {len(codes)} languages", flush=True)

    counts: dict[str, int] = {}
    for code in codes:
        out_path = work / f"{code}.parquet"
        if out_path.exists():
            counts[code] = pq.read_metadata(out_path).num_rows
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        files = _category_files(f"Lingua Libre pronunciation-{code}", pages=2)
        mapping = dict(list(_words(files).items())[:_PER_LANGUAGE])

        # one imageinfo request per file would be thousands of round trips
        urls: dict[str, str] = {}
        titles = list(mapping)
        for i in range(0, len(titles), 40):
            d = _get(
                {
                    "action": "query",
                    "titles": "|".join(titles[i : i + 40]),
                    "prop": "imageinfo",
                    "iiprop": "url",
                }
            )
            for page in (d.get("query") or {}).get("pages", {}).values():
                url = ((page.get("imageinfo") or [{}])[0]).get("url")
                if url:
                    urls[page["title"]] = url
            time.sleep(0.2)

        rows = []
        for f, word in mapping.items():
            url = urls.get(f)
            if not url:
                continue
            # Commons throttles sustained downloads, and a throttled response decodes
            # as garbage rather than raising, so the status is checked and the loop paced.
            try:
                r = requests.get(url, headers={"User-Agent": _UA}, timeout=90)
                if r.status_code != 200:
                    time.sleep(2)
                    continue
                audio = _to_opus(r.content)
            except Exception:
                continue
            finally:
                time.sleep(0.15)
            if audio is None:
                continue
            rows.append(
                {
                    "id": f"{code}-{len(rows)}",
                    "audio": {"bytes": audio, "path": None},
                    "text": word,
                }
            )

        if len(rows) < 40:
            print(f"  {code}: skipped, only {len(rows)} words", flush=True)
            continue
        Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=_TARGET_RATE)
        ).to_parquet(str(out_path))
        counts[code] = len(rows)
        print(f"  {code}: {len(rows)} words", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - automatic-speech-recognition",
        "language:",
        *[f"  - {c}" for c in codes],
        "tags:",
        "  - mteb",
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        lines += [
            f"  - config_name: {c}",
            "    data_files:",
            "      - split: test",
            f"        path: {c}.parquet",
        ]
    lines += [
        "---",
        "",
        "# Lingua Libre spoken-word retrieval (MTEB)",
        "",
        "Single words read aloud by volunteers, paired with the written word, across many",
        "languages.",
        "",
        "Recordings come from Lingua Libre, a Wikimedia project, hosted on Wikimedia",
        f"Commons, which is free by site policy. Published as {_LICENSE}. Audio is 16 kHz",
        "Opus. Bare punctuation and read sentences are excluded, and each word is kept once.",
        "",
        "Built by `scripts/data/lingua_libre/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = sorted(p.stem for p in work.glob("*.parquet"))
    for code in codes:
        api.upload_file(
            path_or_fileobj=str(work / f"{code}.parquet"),
            path_in_repo=f"{code}.parquet",
            repo_id=_TARGET,
            repo_type="dataset",
        )
        print(f"  pushed {code}", flush=True)
    api.upload_file(
        path_or_fileobj=io.BytesIO(_card(codes).encode()),
        path_in_repo="README.md",
        repo_id=_TARGET,
        repo_type="dataset",
    )
    print(f"pushed {_TARGET}: {len(codes)} languages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("lingualibre_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: Wikimedia Commons, Lingua Libre (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
