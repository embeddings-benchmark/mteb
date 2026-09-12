#!/usr/bin/env python3
"""Build the Afri-MCQA multiple choice visual QA task for MTEB.

Source is [Afri-MCQA](https://arxiv.org/abs/2601.05699), culturally grounded multiple
choice questions about photographs, in 16 African languages. #5356 added the a2i/i2a
retrieval directions over the same source; this is the QA formulation asked for in that
thread, where the question and its photograph together select the correct answer.

The answer key is the constraint that shapes the build. The per-language `*_test`
configs ship the four options already shuffled into `native_option_1..4` with no label,
so they cannot be scored. The `*_dev` configs keep `correct_native` separate from
`wrong_native_o1..o3`, so `dev` is the only split where the correct answer is known and
is what this task evaluates on.

Each row keeps its own options, which is what lets the task be scored the way mteb
scores its other multiple choice sets: the candidates become `top_ranked` so a question
is ranked against its own four options rather than a pooled corpus, and the metric is
accuracy. Because the source lists the correct answer first in every row, the options
are shuffled by a hash of the question, which keeps the order fixed across rebuilds
without leaving the answer at index 0 every time.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/afri_mcqa_multiple_choice/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/afri_mcqa_multiple_choice/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Image, load_dataset
from huggingface_hub import HfApi

_SOURCE_REPO = "Atnafu/Afri-MCQA"
_SOURCE_REV = "8b8c53df57b0c2cf9d9798c53515ba1dd14df669"
_TARGET = "vnahata/AfriMCQA-multiple-choice"
_LICENSE = "cc-by-nc-4.0"

_LANGS = {
    "Akan_Twi": "twi",
    "Amharic": "amh",
    "Chichewa": "nya",
    "Hausa": "hau",
    "Igbo": "ibo",
    "Kikuyu": "kik",
    "Kinyarwanda": "kin",
    "Lingala": "lin",
    "Luganda": "lug",
    "Oromo": "orm",
    "Sesotho": "sot",
    "Setswana": "tsn",
    "Somali": "som",
    "Tigrinya": "tir",
    "Yoruba": "yor",
    "Zulu": "zul",
}

_QUESTION = "native_question"
_CORRECT = "correct_native"
_WRONG = ["wrong_native_o1", "wrong_native_o2", "wrong_native_o3"]


# A few cells survive the source's export as the string a null becomes rather than as
# an empty cell, so they have to be caught by value and not just by emptiness.
_NULLISH = {"", "nan", "none", "null", "n/a", "na"}


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return "" if text.casefold() in _NULLISH else text


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}

    for name, code in _LANGS.items():
        q_path = work / f"{code}.parquet"
        if q_path.exists():
            counts[code] = {"questions": pq.read_metadata(q_path).num_rows}
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        ds = load_dataset(
            _SOURCE_REPO, f"{name}_dev", revision=_SOURCE_REV, split="dev"
        )
        text_cols = [c for c in [_QUESTION, _CORRECT, *_WRONG] if c in ds.column_names]
        text = ds.select_columns(text_cols).to_dict()

        # Read the photographs undecoded: decoding every image only to check that it is
        # present would cost far more memory than the build needs.
        raw = ds.select_columns(["image"]).cast_column("image", Image(decode=False))
        blobs = [rec["image"].get("bytes") if rec["image"] else None for rec in raw]

        rows, seen_q = [], set()
        for i in range(len(blobs)):
            question = _clean(text.get(_QUESTION, [None] * len(blobs))[i])
            correct = _clean(text.get(_CORRECT, [None] * len(blobs))[i])
            wrong = [_clean(text.get(w, [None] * len(blobs))[i]) for w in _WRONG]
            wrong = [w for w in wrong if w]

            # A question with no photograph, no text, or no answer cannot be scored, one
            # whose correct answer is also listed as its own distractor is ambiguous, and
            # one left with no distractor is not a choice at all.
            if not (blobs[i] and question and correct):
                continue
            if correct in wrong or not wrong:
                continue
            # An identical question asked twice about the same language would be
            # relevant to whichever copy came first, so the repeat is dropped.
            if question in seen_q:
                continue
            seen_q.add(question)

            # The source always lists the answer first, so leaving source order would
            # put the correct option at index 0 for every single question. Shuffled by a
            # hash of the question so the order is fixed across rebuilds.
            options = [correct, *wrong]
            digest = hashlib.md5(question.encode("utf-8")).digest()
            options.sort(key=lambda o: hashlib.md5(digest + o.encode("utf-8")).digest())

            rows.append(
                {
                    "id": f"{code}-q-{len(rows)}",
                    "text": question,
                    "image": {"bytes": blobs[i], "path": None},
                    "candidates": options,
                    "answer": correct,
                }
            )

        if len(rows) < 20:
            print(f"  {code}: skipped, only {len(rows)} questions", flush=True)
            continue

        Dataset.from_list(rows).cast_column("image", Image()).to_parquet(str(q_path))
        n_opts = sum(len(r["candidates"]) for r in rows)
        counts[code] = {"questions": len(rows), "options": n_opts}
        print(f"  {code}: {counts[code]}", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - visual-question-answering",
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
            "      - split: dev",
            f"        path: {c}.parquet",
        ]
    lines += [
        "---",
        "",
        "# Afri-MCQA multiple choice visual QA (MTEB)",
        "",
        "Culturally grounded multiple choice questions about photographs, in 16 African",
        "languages. Each row is a question, the photograph it asks about, its answer",
        "options, and which of them is correct. Options are shuffled by a hash of the",
        "question, because the source always lists the answer first.",
        "",
        f"Built from `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}, using",
        "the official `dev` split, the only one where the correct answer is labelled.",
        "",
        "Built by `scripts/data/afri_mcqa_multiple_choice/create_data.py` in the MTEB repo.",
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
    parser.add_argument("--work-dir", type=Path, default=Path("afri_qa_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
