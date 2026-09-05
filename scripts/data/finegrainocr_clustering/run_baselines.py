#!/usr/bin/env python3
"""Run reproducible multimodal and unimodal FineGrainOCR clustering baselines.

Image and text embeddings are computed once per model and cached. The script
scores image-only, text-only, and the exact additive fusion used by MTEB's CLIP
and SigLIP wrappers. It also reports an L2-normalized fusion-weight sweep to
show whether both modalities contribute useful cluster signal.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

MODELS = {
    "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "google/siglip-base-patch16-256": "b078df89e446d623010d890864d4207fe6399f61",
}
DEFAULT_SEED = 42
DEFAULT_REPEATS = 10
DEFAULT_CLUSTER_SIZE = 16_384
DEFAULT_KMEANS_BATCH_SIZE = 512


def _tensor(output: Any) -> torch.Tensor:
    if isinstance(output, BaseModelOutputWithPooling):
        return output.pooler_output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected feature output: {type(output)}")
    return output


def _encode(
    dataset: Any,
    model_name: str,
    revision: str,
    *,
    image_batch_size: int,
    text_batch_size: int,
    local_files_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    processor = AutoProcessor.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    ).eval()

    started = time.perf_counter()
    image_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(dataset), image_batch_size), desc="Image encoding"
        ):
            images = [
                image.convert("RGB")
                for image in dataset[start : start + image_batch_size]["image"]
            ]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            features = _tensor(model.get_image_features(**inputs))
            image_batches.append(features.float().cpu().numpy())
    image_seconds = time.perf_counter() - started

    started = time.perf_counter()
    text_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(dataset), text_batch_size), desc="Text encoding"
        ):
            texts = dataset[start : start + text_batch_size]["text"]
            inputs = processor(
                text=texts,
                return_tensors="pt",
                padding="max_length"
                if model_name.startswith("google/siglip")
                else True,
                truncation=True,
            )
            features = _tensor(model.get_text_features(**inputs))
            text_batches.append(features.float().cpu().numpy())
    text_seconds = time.perf_counter() - started

    return (
        np.concatenate(image_batches),
        np.concatenate(text_batches),
        {"image": image_seconds, "text": text_seconds},
    )


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, np.finfo(embeddings.dtype).eps)


def _score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int,
    cluster_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    v_measures: list[float] = []
    ami_scores: list[float] = []
    n_clusters = len(np.unique(labels))
    for _ in tqdm(range(repeats), desc="Clustering repeats", leave=False):
        indices = np.asarray(rng.choices(range(len(embeddings)), k=cluster_size))
        sampled_embeddings = embeddings[indices]
        sampled_labels = labels[indices]
        predictions = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=DEFAULT_KMEANS_BATCH_SIZE,
            init="k-means++",
            n_init=1,
            random_state=seed,
        ).fit_predict(sampled_embeddings)
        v_measures.append(float(v_measure_score(sampled_labels, predictions)))
        ami_scores.append(
            float(adjusted_mutual_info_score(sampled_labels, predictions))
        )
    return {
        "v_measure": float(np.mean(v_measures)),
        "v_measure_std": float(np.std(v_measures)),
        "v_measures": v_measures,
        "ami": float(np.mean(ami_scores)),
        "ami_std": float(np.std(ami_scores)),
        "ami_scores": ami_scores,
    }


def _experiment_embeddings(
    image: np.ndarray, text: np.ndarray, *, seed: int
) -> dict[str, np.ndarray]:
    normalized_image = _l2_normalize(image)
    normalized_text = _l2_normalize(text)
    experiments = {
        "random_gaussian_control": np.random.default_rng(seed)
        .standard_normal(image.shape)
        .astype(np.float32),
        "image_only_mteb": image,
        "text_only_mteb": text,
        "joint_mteb_add": image + text,
    }
    for image_weight in (0.25, 0.5, 0.75):
        fused = image_weight * normalized_image + (1 - image_weight) * normalized_text
        experiments[f"joint_normalized_image_weight_{image_weight:.2f}"] = (
            _l2_normalize(fused)
        )
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODELS),
        help="May be passed more than once; defaults to both cached baselines.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    loaded = load_from_disk(args.dataset)
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(loaded)}")
    dataset = loaded["test"]
    labels = np.asarray(dataset["label"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.model or list(MODELS):
        revision = MODELS[model_name]
        slug = model_name.replace("/", "--")
        embedding_path = args.output_dir / f"{slug}-embeddings.npz"
        if embedding_path.exists():
            cached = np.load(embedding_path)
            image_embeddings = cached["image"]
            text_embeddings = cached["text"]
            encoding_seconds = {"image": 0.0, "text": 0.0, "cached": True}
        else:
            image_embeddings, text_embeddings, encoding_seconds = _encode(
                dataset,
                model_name,
                revision,
                image_batch_size=args.image_batch_size,
                text_batch_size=args.text_batch_size,
                local_files_only=not args.allow_download,
            )
            np.savez_compressed(
                embedding_path,
                image=image_embeddings,
                text=text_embeddings,
            )
            encoding_seconds["cached"] = False

        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                f"Modality shape mismatch: {image_embeddings.shape} != "
                f"{text_embeddings.shape}"
            )
        results: dict[str, Any] = {
            "model": model_name,
            "revision": revision,
            "rows": len(labels),
            "classes": len(np.unique(labels)),
            "embedding_dimension": image_embeddings.shape[1],
            "encoding_seconds": encoding_seconds,
            "protocol": {
                "seed": args.seed,
                "repeats": args.repeats,
                "bootstrap_cluster_size": args.cluster_size,
                "kmeans_batch_size": DEFAULT_KMEANS_BATCH_SIZE,
                "kmeans_n_init": 1,
            },
            "embedding_norms": {
                "image_mean": float(np.mean(np.linalg.norm(image_embeddings, axis=1))),
                "text_mean": float(np.mean(np.linalg.norm(text_embeddings, axis=1))),
            },
            "experiments": {},
        }
        for name, embeddings in _experiment_embeddings(
            image_embeddings, text_embeddings, seed=args.seed
        ).items():
            print(f"model={model_name} experiment={name}")
            results["experiments"][name] = _score(
                embeddings,
                labels,
                repeats=args.repeats,
                cluster_size=args.cluster_size,
                seed=args.seed,
            )

        result_path = args.output_dir / f"{slug}.json"
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"results={result_path}")


if __name__ == "__main__":
    main()
