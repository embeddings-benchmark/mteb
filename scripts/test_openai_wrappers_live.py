"""Test OpenAIAPI*Wrapper against real vLLM servers and real (small) MTEB tasks.

Companion to scripts/serve_vllm_models.sh. Start the matching server first,
then run the same scenario here:

    scripts/serve_vllm_models.sh text-embed              # terminal 1
    python scripts/test_openai_wrappers_live.py text-embed  # terminal 2

Scenarios
---------
  text-embed              OpenAIAPIEncodeWrapper,     text-only  -> NanoSciFactRetrieval
  multimodal-embed        OpenAIAPIEncodeWrapper,     image+text -> VisRAGRetArxivQA
  text-rerank             OpenAIAPIRerankWrapper,     text-only  -> AskUbuntuDupQuestions
  multimodal-rerank       OpenAIAPIRerankWrapper,     image+text -> manual smoke test
                          (no image-reranking MTEB task exists yet, so this
                          hand-builds a small text-query-vs-image-document
                          example instead, calling .predict() directly)
  text-token-embed        OpenAIAPITokenEmbedWrapper, text-only  -> NanoSciFactRetrieval
  multimodal-token-embed  OpenAIAPITokenEmbedWrapper, image+text -> VisRAGRetArxivQA

Each scenario matches one case in serve_vllm_models.sh (same model, same
default port).

Usage
-----
  python scripts/test_openai_wrappers_live.py text-embed
  python scripts/test_openai_wrappers_live.py multimodal-embed --endpoint-url http://localhost:8000
  python scripts/test_openai_wrappers_live.py text-rerank --num-proc 4
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

import mteb
from mteb.models.openai_wrappers import (
    OpenAIAPIEncodeWrapper,
    OpenAIAPIRerankWrapper,
    OpenAIAPITokenEmbedWrapper,
)

# Same public demo asset vLLM's own examples use (e.g. vision_rerank_api_online.py).
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


class _FakeTaskMetadata:
    """Minimal stand-in for TaskMetadata, enough for the wrappers' predict()/encode()."""

    def __init__(self, name: str) -> None:
        self.name = name


@dataclass
class Scenario:
    description: str
    wrapper: Literal["encode", "rerank", "token_embed"]
    model_name: str
    default_port: int
    modalities: list[str]
    task_names: list[str] | None
    """MTEB tasks to run via mteb.evaluate(); None means "manual smoke test"."""
    wrapper_kwargs: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {
    "text-embed": Scenario(
        description="Text-only embeddings via /v1/embeddings",
        wrapper="encode",
        model_name="BAAI/bge-small-en-v1.5",
        default_port=8000,
        modalities=["text"],
        task_names=["NanoSciFactRetrieval"],
        # bge-small has no chat template, so `messages` (the wrapper's
        # default text path) gets rejected with a 400; use `input` instead.
        wrapper_kwargs={"use_chat_template": False},
    ),
    "multimodal-embed": Scenario(
        description="Image+text embeddings via vLLM's Chat Embeddings API",
        wrapper="encode",
        model_name="Qwen/Qwen3-VL-Embedding-2B",
        default_port=8000,
        modalities=["text", "image"],
        task_names=["Vidore3HrRetrieval"],
    ),
    "text-rerank": Scenario(
        description="Text-only reranking via /v1/rerank",
        wrapper="rerank",
        model_name="BAAI/bge-reranker-v2-m3",
        default_port=8001,
        modalities=["text"],
        task_names=["AskUbuntuDupQuestions"],
    ),
    "multimodal-rerank": Scenario(
        description="Image+text reranking via /v1/rerank (manual smoke test)",
        wrapper="rerank",
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        default_port=8001,
        modalities=["text", "image"],
        task_names=None,
    ),
    "text-token-embed": Scenario(
        description="Text-only ColBERT-style multi-vector retrieval via /pooling",
        wrapper="token_embed",
        model_name="TomoroAI/tomoro-colqwen3-embed-4b",
        default_port=8002,
        modalities=["text"],
        task_names=["NanoSciFactRetrieval"],
        # bge-m3 has no chat template, so `messages` (the wrapper's default
        # text path) gets rejected with a 400; use the plain `input` field.
        wrapper_kwargs={"use_chat_template": False},
    ),
    "multimodal-token-embed": Scenario(
        description="Image+text ColBERT-style multi-vector retrieval via /pooling",
        wrapper="token_embed",
        model_name="TomoroAI/tomoro-colqwen3-embed-4b",
        default_port=8002,
        modalities=["text", "image"],
        task_names=["Vidore3HrRetrieval"],
    ),
}


def _build_model(
    scenario: Scenario, endpoint_url: str
) -> OpenAIAPIEncodeWrapper | OpenAIAPIRerankWrapper | OpenAIAPITokenEmbedWrapper:
    common_kwargs: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "model_name": scenario.model_name,
        "modalities": scenario.modalities,
        **scenario.wrapper_kwargs,
    }
    if scenario.wrapper == "encode":
        return OpenAIAPIEncodeWrapper(**common_kwargs)
    if scenario.wrapper == "rerank":
        return OpenAIAPIRerankWrapper(**common_kwargs)
    return OpenAIAPITokenEmbedWrapper(**common_kwargs)


def _run_mteb_eval(model: Any, task_names: list[str], num_proc: int | None) -> None:
    tasks = mteb.get_tasks(tasks=task_names)
    result = mteb.evaluate(model, tasks, num_proc=num_proc)
    print()
    print("Results:")
    for task_result in result.task_results:
        print(f"  {task_result.task_name}: main_score={task_result.main_score:.4f}")


def _fetch_image(url: str):  # -> PIL.Image.Image
    from PIL import Image

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _manual_rerank_smoke_test(model: OpenAIAPIRerankWrapper) -> None:
    """Hand-built text-query-vs-image-document example.

    There's currently no image-reranking task in MTEB to run via
    mteb.evaluate(), so this calls .predict() directly instead, mirroring
    vLLM's own vision_rerank_api_online.py example.
    """
    print("Fetching test image...")
    image = _fetch_image(IMAGE_URL)

    class _Batch(dict):
        pass

    query_text = "What is shown in this image?"
    doc_texts = ["A red panda climbing a tree.", "A city skyline at night."]

    inputs1 = [_Batch(text=[query_text, query_text])]
    inputs2 = [_Batch(text=doc_texts, image=[image, image])]

    scores = model.predict(
        inputs1,
        inputs2,
        task_metadata=_FakeTaskMetadata("manual-multimodal-rerank-smoke-test"),
        hf_split="test",
        hf_subset="default",
        show_progress_bar=False,
    )

    print()
    print("Results (query vs. each image+text document):")
    for doc_text, score in zip(doc_texts, scores):
        print(f"  score={score:.4f}  document text='{doc_text}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scenario", choices=sorted(SCENARIOS), help="Scenario to run")
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Server URL. Defaults to http://localhost:<scenario's default port>.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for MTEB data loading (passed to mteb.evaluate).",
    )
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    endpoint_url = args.endpoint_url or f"http://localhost:{scenario.default_port}"

    print(f"Scenario: {args.scenario} - {scenario.description}")
    print(f"Model:    {scenario.model_name}")
    print(f"Endpoint: {endpoint_url}")

    try:
        model = _build_model(scenario, endpoint_url)
    except ConnectionError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        print(
            f"Start the server first: scripts/serve_vllm_models.sh {args.scenario}",
            file=sys.stderr,
        )
        sys.exit(1)

    if scenario.task_names is None:
        assert isinstance(model, OpenAIAPIRerankWrapper)
        _manual_rerank_smoke_test(model)
    else:
        _run_mteb_eval(model, scenario.task_names, args.num_proc)


if __name__ == "__main__":
    main()
