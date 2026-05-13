# MVEB data-preparation scripts

Reference-style scripts documenting how each `mteb/<dataset>` video / audio-visual artifact was produced for the MVEB contribution to MTEB. **Not runnable end-to-end** — sources (YouTube, Kaggle, HF mirrors) routinely move or expire. The canonical artifact for each dataset is the published HuggingFace dataset listed in the script's `TARGET_REPO` constant.

## Conventions

Each subdirectory is named for the dataset and contains a `create_data.py`. The script's module docstring carries:

- **Source**: original dataset citation
- **Mirror used**: the upstream HF / Kaggle / S3 URL the original pipeline pulled from
- **MVEB-specific processing**: numbered list of the deltas MVEB applied (filtering, sampling, subsetting, audio extraction, schema)
- **Final size**: row counts per split

The code body is the deterministic transformation reduced to a small `main()` + helpers, with `Dataset.from_generator` / `Dataset.from_dict` and `push_to_hub(TARGET_REPO)` at the end.

## Reconstructed entries

A few entries are flagged "REFERENCE ONLY — RECONSTRUCTED" in their docstring: the published HF artifact is authoritative, the original processing pipeline was not preserved, and the body is a best-effort reconstruction from the schema observed on the Hub. These should be treated as documentation, not a faithful pipeline replay.
