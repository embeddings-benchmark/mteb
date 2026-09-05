#!/usr/bin/env python3
"""Build the COCO Modality Equivalence dataset for MTEB.

Constructs a dataset with six configs (one per retrieval direction) over a
shared pool of MSCOCO images where every item is simultaneously available as:
  - an image
  - a text caption (from MBEIR / mscoco_task0)
  - a human-spoken caption (from SpokenCOCO, whybe-choi/SpokenCOCOA2IRetrieval)
  - a TTS-synthesized caption (from SpeechCoco, dukesun99/SpeechCoco-A2I)

Because the candidate pool is identical across all six configs, any retrieval
score difference is attributable to modality difficulty rather than content.

Source schemas (all BEIR format with corpus/queries/qrels configs):
  SpokenCOCO A2I -- corpus: {id, image}  queries: {id, audio}
  SpeechCoco  A2I -- corpus: {id, image}  queries: {id, audio}
  MSCOCO T2I     -- corpus: {id, text, image}  (split name is "corpus")

Examples:
  uv run python scripts/data/coco_modality_equivalence/create_data.py --stage build
  uv run python scripts/data/coco_modality_equivalence/create_data.py --stage all --repo mteb/coco-modality-equivalence
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, Features, Image, Value, load_dataset

_SPOKEN_COCO = "whybe-choi/SpokenCOCOA2IRetrieval"
_SPEECH_COCO = "dukesun99/SpeechCoco-A2I"
_MSCOCO = "mteb/mbeir_mscoco_task0"
_SPLIT = "test"
_DEFAULT_REPO = "mteb/coco-modality-equivalence"

# ── source loaders ────────────────────────────────────────────────────────────


def _load_spoken_coco() -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Return (image_id -> image_bytes, image_id -> human_audio_bytes)."""
    print("Loading SpokenCOCO corpus …")
    corpus = load_dataset(_SPOKEN_COCO, "corpus", split=_SPLIT)
    images: dict[str, bytes] = {}
    for row in corpus:
        img = row["image"]
        raw = img.get("bytes") if isinstance(img, dict) else None
        if raw and row["id"]:
            images[str(row["id"])] = raw

    print(f"  {len(images)} corpus images")

    print("Loading SpokenCOCO qrels …")
    qrels = load_dataset(_SPOKEN_COCO, "qrels", split=_SPLIT)
    # query-id -> corpus-id (image)
    q2img: dict[str, str] = {str(r["query-id"]): str(r["corpus-id"]) for r in qrels}

    print("Loading SpokenCOCO queries (audio) …")
    queries = load_dataset(_SPOKEN_COCO, "queries", split=_SPLIT)
    # image-id -> first audio bytes
    human_by_image: dict[str, bytes] = {}
    for row in queries:
        qid = str(row["id"])
        img_id = q2img.get(qid)
        if img_id is None or img_id in human_by_image:
            continue
        aud = row["audio"]
        raw = aud.get("bytes") if isinstance(aud, dict) else None
        if raw:
            human_by_image[img_id] = raw

    print(f"  {len(human_by_image)} images with human audio")
    return images, human_by_image


def _load_speech_coco() -> dict[str, bytes]:
    """Return image_id -> TTS audio bytes from SpeechCoco."""
    print("Loading SpeechCoco qrels …")
    qrels = load_dataset(_SPEECH_COCO, "qrels", split=_SPLIT)
    q2img = {str(r["query-id"]): str(r["corpus-id"]) for r in qrels}

    print("Loading SpeechCoco queries (TTS audio) …")
    queries = load_dataset(_SPEECH_COCO, "queries", split=_SPLIT)
    tts_by_image: dict[str, bytes] = {}
    for row in queries:
        qid = str(row["id"])
        img_id = q2img.get(qid)
        if img_id is None or img_id in tts_by_image:
            continue
        aud = row["audio"]
        raw = aud.get("bytes") if isinstance(aud, dict) else None
        if raw:
            tts_by_image[img_id] = raw

    print(f"  {len(tts_by_image)} images with TTS audio")
    return tts_by_image


def _load_mscoco_text() -> dict[str, str]:
    """Return image_id -> text caption from MSCOCO MBEIR corpus."""
    print("Loading MSCOCO corpus (text + image) …")
    # mbeir uses split name "corpus", not "test"
    corpus = load_dataset(_MSCOCO, "corpus", split="corpus")
    text_by_image: dict[str, str] = {}
    for row in corpus:
        iid = str(row["id"])
        if iid not in text_by_image and row.get("text"):
            text_by_image[iid] = str(row["text"])
    print(f"  {len(text_by_image)} images with text captions")
    return text_by_image


# ── build ─────────────────────────────────────────────────────────────────────


def _build_beir_config(
    corpus_rows: list[dict],
    query_rows: list[dict],
    qrels_rows: list[dict],
    corpus_features: Features,
    query_features: Features,
) -> DatasetDict:
    qrels_features = Features(
        {
            "query-id": Value("string"),
            "corpus-id": Value("string"),
            "score": Value("int32"),
        }
    )
    return DatasetDict(
        {
            "corpus": Dataset.from_list(corpus_rows, features=corpus_features),
            "queries": Dataset.from_list(query_rows, features=query_features),
            "qrels": Dataset.from_list(qrels_rows, features=qrels_features),
        }
    )


def stage_build(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)

    images, human_by_image = _load_spoken_coco()
    tts_by_image = _load_speech_coco()
    text_by_image = _load_mscoco_text()

    shared_ids = sorted(
        set(images) & set(human_by_image) & set(tts_by_image) & set(text_by_image)
    )
    print(f"\nShared image IDs across all sources: {len(shared_ids)}")
    if not shared_ids:
        raise RuntimeError(
            "No shared IDs found. Corpus IDs may differ across datasets — "
            "inspect IDs from each source and adapt _load_* functions."
        )

    # Pre-build per-image index for query IDs
    # We use image_id as both the corpus-id for images and the query-id for
    # image queries; human/TTS audio get synthetic IDs "h_{img_id}" / "s_{img_id}"
    img_feats = Features({"id": Value("string"), "image": Image()})
    txt_feats = Features({"id": Value("string"), "text": Value("string")})
    aud_feats = Features({"id": Value("string"), "audio": Audio(sampling_rate=16_000)})

    img_corpus, txt_corpus, aud_h_corpus, aud_s_corpus = [], [], [], []
    img_queries = []
    qrels_t2i, qrels_a2i_h, qrels_a2i_s = [], [], []
    qrels_i2t, qrels_i2a_h, qrels_i2a_s = [], [], []

    for img_id in shared_ids:
        h_id = f"h_{img_id}"
        s_id = f"s_{img_id}"

        img_corpus.append({"id": img_id, "image": images[img_id]})
        txt_corpus.append({"id": img_id, "text": text_by_image[img_id]})
        aud_h_corpus.append({"id": h_id, "audio": human_by_image[img_id]})
        aud_s_corpus.append({"id": s_id, "audio": tts_by_image[img_id]})
        img_queries.append({"id": img_id, "image": images[img_id]})

        qrels_t2i.append({"query-id": img_id, "corpus-id": img_id, "score": 1})
        qrels_a2i_h.append({"query-id": h_id, "corpus-id": img_id, "score": 1})
        qrels_a2i_s.append({"query-id": s_id, "corpus-id": img_id, "score": 1})
        qrels_i2t.append({"query-id": img_id, "corpus-id": img_id, "score": 1})
        qrels_i2a_h.append({"query-id": img_id, "corpus-id": h_id, "score": 1})
        qrels_i2a_s.append({"query-id": img_id, "corpus-id": s_id, "score": 1})

    configs = {
        "t2i": _build_beir_config(
            img_corpus, txt_corpus, qrels_t2i, img_feats, txt_feats
        ),
        "a2i_h": _build_beir_config(
            img_corpus,
            [
                {"id": r["id"].replace("h_", ""), "audio": r["audio"]}
                for r in aud_h_corpus
            ],
            qrels_a2i_h,
            img_feats,
            aud_feats,
        ),
        "a2i_s": _build_beir_config(
            img_corpus,
            [
                {"id": r["id"].replace("s_", ""), "audio": r["audio"]}
                for r in aud_s_corpus
            ],
            qrels_a2i_s,
            img_feats,
            aud_feats,
        ),
        "i2t": _build_beir_config(
            txt_corpus, img_queries, qrels_i2t, txt_feats, img_feats
        ),
        "i2a_h": _build_beir_config(
            aud_h_corpus, img_queries, qrels_i2a_h, aud_feats, img_feats
        ),
        "i2a_s": _build_beir_config(
            aud_s_corpus, img_queries, qrels_i2a_s, aud_feats, img_feats
        ),
    }

    for name, dd in configs.items():
        out = work / name
        dd.save_to_disk(str(out))
        print(f"Saved config '{name}' -> {out}")

    print(f"\nBuild complete: {len(shared_ids)} items, 6 configs")
    print("Image IDs sample:", shared_ids[:5])


# ── push ──────────────────────────────────────────────────────────────────────


def stage_push(work: Path, repo: str) -> None:
    from datasets import load_from_disk

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

    for cfg_name in ("t2i", "a2i_h", "a2i_s", "i2t", "i2a_h", "i2a_s"):
        cfg_path = work / cfg_name
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found — run --stage build first")
        dd = load_from_disk(str(cfg_path))
        for split_name, ds in dd.items():
            ds.push_to_hub(repo, config_name=cfg_name, split=split_name)
        print(f"Pushed config '{cfg_name}'")

    info = api.dataset_info(repo)
    print(f"\nPushed to {repo}\nRevision: {info.sha}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["build", "push", "all"],
        default="build",
    )
    parser.add_argument("--work", default="work/coco_modality_equivalence")
    parser.add_argument("--repo", default=_DEFAULT_REPO)
    args = parser.parse_args()

    work = Path(args.work)
    if args.stage in ("build", "all"):
        stage_build(work)
    if args.stage in ("push", "all"):
        stage_push(work, args.repo)


if __name__ == "__main__":
    main()
