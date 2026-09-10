"""Run native MTEB ToolRetrieval evaluations on Modal.

The evaluation itself uses ``mteb.get_model`` and ``mteb.evaluate``; this
script only provides a GPU execution environment and copies MTEB's generated
artifacts back into the local ``results/`` directory.

Run from the repository root::

    modal run scripts/run_toolret_mteb_modal.py

To evaluate one model instead::

    modal run scripts/run_toolret_mteb_modal.py --model-name intfloat/e5-base-v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal


MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-large-en-v1.5",
)
REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("mteb-toolretrieval-native-results")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "mteb==2.20.11",
        "torch==2.8.0",
        "transformers==5.16.1",
        "sentence-transformers==6.0.1",
        "datasets==5.0.1",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/mteb",
        ignore=[".git", ".venv", "results", "toolret_cache", "__pycache__"],
    )
)


@app.function(image=image, gpu="L4", timeout=2 * 60 * 60)
def evaluate_model(model_name: str) -> dict[str, str]:
    """Evaluate one model using MTEB's public APIs and return its raw artifacts."""
    sys.path.insert(0, "/root/mteb")

    import mteb
    from mteb.models import CachedEmbeddingWrapper

    cache_root = Path("/tmp/mteb-cache")
    model = mteb.get_model(model_name, device="cuda")
    tasks = mteb.get_tasks(tasks=["ToolRetrieval", "ToolRetrievalInstruction"])
    cached_model = CachedEmbeddingWrapper(model, cache_root / "embedding_cache")
    try:
        mteb.evaluate(
            cached_model,
            tasks,
            cache=mteb.ResultCache(cache_root),
            overwrite_strategy="always",
            encode_kwargs={"batch_size": 32},
            show_progress_bar=False,
        )
    finally:
        cached_model.close()

    artifacts = cache_root / "results"
    return {
        str(path.relative_to(cache_root)): path.read_text()
        for path in artifacts.rglob("*")
        if path.is_file()
    }


@app.local_entrypoint()
def main(model_name: str | None = None) -> None:
    """Write the raw MTEB artifacts from each remote run into ``results/``."""
    runs: list[dict[str, str]]
    if model_name:
        runs = [evaluate_model.remote(model_name)]
    else:
        runs = list(evaluate_model.map(MODELS))

    for artifacts in runs:
        for relative_path, content in artifacts.items():
            destination = Path(relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
            print(destination)
