#!/usr/bin/env python3
"""Package 11hu83/vggsound into MTEB a2v / v2a retrieval Hub datasets.

Source has ~15.4k YouTube clips with paired video + audio (~62 GB if loaded
via parquet). This script reads metadata.csv, deterministically shuffles with
seed 42, keeps 2048 ids, downloads only those media files, then pushes
separate A2V and V2A repos in the standard corpus / queries / qrels layout.

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/vggsound_retrieval/create_data.py \\
      --a2v-repo-id {repo_id}/VGGSound-A2V \\
      --v2a-repo-id {repo_id}/VGGSound-V2A \\
      --push
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, Video
from huggingface_hub import create_repo, get_token, hf_hub_download
from tqdm import tqdm

_SOURCE = "11hu83/vggsound"
_SOURCE_REVISION = "dc7815aa65132c48f070686e0004180f03b5abcf"
_N_SAMPLES = 2048
_SEED = 42
_DOWNLOAD_WORKERS = 8

_VIDEO_ID_RE = re.compile(r"^video/(.+)/video\.mp4$")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def _video_id_from_file_name(file_name: str) -> str | None:
    m = _VIDEO_ID_RE.match(file_name)
    return m.group(1) if m else None


def _download_pair(video_id: str, token: str | None) -> dict | None:
    try:
        video_path = hf_hub_download(
            _SOURCE,
            f"video/{video_id}/video.mp4",
            repo_type="dataset",
            revision=_SOURCE_REVISION,
            token=token,
        )
        audio_path = hf_hub_download(
            _SOURCE,
            f"audio/{video_id}/audio.wav",
            repo_type="dataset",
            revision=_SOURCE_REVISION,
            token=token,
        )
    except Exception as e:  # noqa: BLE001
        print(f"skip {video_id}: {e}")
        return None
    if Path(video_path).stat().st_size == 0 or Path(audio_path).stat().st_size == 0:
        print(f"skip {video_id}: empty media")
        return None
    return {"id": video_id, "video": video_path, "audio": audio_path}


def _id_order(token: str | None) -> list[str]:
    meta_path = hf_hub_download(
        _SOURCE,
        "metadata.csv",
        repo_type="dataset",
        revision=_SOURCE_REVISION,
        token=token,
    )
    df = pd.read_csv(meta_path)
    all_ids: list[str] = []
    for name in df["file_name"]:
        vid = _video_id_from_file_name(str(name))
        if vid is not None:
            all_ids.append(vid)
    return pd.Series(all_ids).sample(frac=1.0, random_state=_SEED).tolist()


def _ensure_n_pairs(
    n_samples: int,
    workers: int = _DOWNLOAD_WORKERS,
    token: str | None = None,
) -> list[dict]:
    """Deterministically permute ids (seed 42) and download until n_samples."""
    order = _id_order(token)
    pairs_by_id: dict[str, dict] = {}
    cursor = 0
    batch_size = max(workers * 2, min(n_samples, 256))

    with tqdm(total=n_samples, desc="download media") as pbar:
        while len(pairs_by_id) < n_samples and cursor < len(order):
            need = n_samples - len(pairs_by_id)
            batch = order[cursor : cursor + max(need, batch_size)]
            cursor += len(batch)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_download_pair, vid, token): vid for vid in batch
                }
                for fut in as_completed(futures):
                    if len(pairs_by_id) >= n_samples:
                        break
                    result = fut.result()
                    if result is None or result["id"] in pairs_by_id:
                        continue
                    pairs_by_id[result["id"]] = result
                    pbar.update(1)

    # Preserve deterministic order from the seeded permutation
    pairs: list[dict] = []
    for video_id in order:
        if video_id in pairs_by_id:
            pairs.append(pairs_by_id[video_id])
        if len(pairs) >= n_samples:
            break

    if len(pairs) < n_samples:
        raise SystemExit(f"Only collected {len(pairs)} pairs; expected {n_samples}")
    return pairs


def _push_a2v(pairs: list[dict], repo_id: str, token: str) -> None:
    corpus = Dataset.from_list(
        [
            {"id": f"d-{p['id']}", "video": p["video"], "modality": "video"}
            for p in pairs
        ]
    ).cast_column("video", Video())
    queries = Dataset.from_list(
        [
            {"id": f"q-{p['id']}", "audio": p["audio"], "modality": "audio"}
            for p in pairs
        ]
    ).cast_column("audio", Audio())
    qrels = Dataset.from_list(
        [
            {"query-id": f"q-{p['id']}", "corpus-id": f"d-{p['id']}", "score": 1}
            for p in pairs
        ]
    )
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    DatasetDict({"test": corpus}).push_to_hub(repo_id, "corpus", token=token)
    DatasetDict({"test": queries}).push_to_hub(repo_id, "queries", token=token)
    DatasetDict({"test": qrels}).push_to_hub(repo_id, "qrels", token=token)


def _push_v2a(pairs: list[dict], repo_id: str, token: str) -> None:
    corpus = Dataset.from_list(
        [
            {"id": f"d-{p['id']}", "audio": p["audio"], "modality": "audio"}
            for p in pairs
        ]
    ).cast_column("audio", Audio())
    queries = Dataset.from_list(
        [
            {"id": f"q-{p['id']}", "video": p["video"], "modality": "video"}
            for p in pairs
        ]
    ).cast_column("video", Video())
    qrels = Dataset.from_list(
        [
            {"query-id": f"q-{p['id']}", "corpus-id": f"d-{p['id']}", "score": 1}
            for p in pairs
        ]
    )
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    DatasetDict({"test": corpus}).push_to_hub(repo_id, "corpus", token=token)
    DatasetDict({"test": queries}).push_to_hub(repo_id, "queries", token=token)
    DatasetDict({"test": qrels}).push_to_hub(repo_id, "qrels", token=token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2v-repo-id", default="Wissam42/VGGSound-A2V")
    parser.add_argument("--v2a-repo-id", default="Wissam42/VGGSound-V2A")
    parser.add_argument("--n-samples", type=int, default=_N_SAMPLES)
    parser.add_argument("--workers", type=int, default=_DOWNLOAD_WORKERS)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/data/vggsound_retrieval/pairs_manifest.json"),
        help="Write/read downloaded pair paths for resume",
    )
    parser.add_argument(
        "--from-manifest",
        action="store_true",
        help="Reuse an existing pairs manifest instead of downloading",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--directions",
        default="both",
        choices=("a2v", "v2a", "both"),
        help="Which Hub datasets to build",
    )
    args = parser.parse_args()
    token = _hf_token()

    if args.from_manifest:
        if not args.manifest.is_file():
            raise SystemExit(f"Manifest not found: {args.manifest}")
        pairs = json.loads(args.manifest.read_text())
        if len(pairs) < args.n_samples:
            raise SystemExit(f"Manifest has {len(pairs)} pairs; need {args.n_samples}")
        pairs = pairs[: args.n_samples]
    else:
        pairs = _ensure_n_pairs(args.n_samples, workers=args.workers, token=token)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(pairs, indent=2))
        print(f"wrote manifest {args.manifest}")

    print(f"sampled={len(pairs)} source={_SOURCE}@{_SOURCE_REVISION}")

    if not args.push:
        print("Dry run (pass --push to upload). Example pair:")
        print(f"  id={pairs[0]['id']}")
        print(f"  video={pairs[0]['video']}")
        print(f"  audio={pairs[0]['audio']}")
        return

    if not token:
        raise SystemExit("Set HF_TOKEN to push")

    if args.directions in ("a2v", "both"):
        print(f"Pushing A2V → {args.a2v_repo_id}")
        _push_a2v(pairs, args.a2v_repo_id, token)
    if args.directions in ("v2a", "both"):
        print(f"Pushing V2A → {args.v2a_repo_id}")
        _push_v2a(pairs, args.v2a_repo_id, token)
    print("done")


if __name__ == "__main__":
    main()
