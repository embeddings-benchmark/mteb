"""LightOn-rerank models: pointwise (PW) cross-encoders and generative listwise (LW) rerankers.

- PW models ship with a sentence-transformers CrossEncoder integration and score
  each (query, document) pair independently via logit("Yes") - logit("No").
- LW models rank exactly ``WINDOW_SIZE`` documents per forward pass by generating a
  permutation string (e.g. ``[2] > [1] > [4] > [3]``). Larger candidate pools are
  reranked with a sliding window (window 4, stride 2) moving from the bottom of the
  list to the top, so the best candidates bubble up to the front.
Reference: https://huggingface.co/collections/lightonai/lighton-rerank
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class LightOnListwiseRerankerWrapper:
    """Generative listwise reranker with sliding-window inference.

    ``predict`` receives the flattened (query, document) pairs produced by
    ``SearchCrossEncoderWrapper`` (grouped per query, in candidate order), reranks
    each query's candidate list with the sliding window, and returns one score per
    pair (``n_docs - final_rank``) so that sorting by score reproduces the
    listwise ranking.

    Windows for all queries that sit at the same window position are batched into a
    single ``generate()`` call (queries are independent, so this is exact), following
    the LightOn reference implementation.
    """

    prompt_template = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    # The 4B backbone enables thinking by default; an empty think block keeps the
    # generation budget for the ranking permutation. The 0.8B/2B prompts omit it.
    think_block = "<think>\n\n</think>\n\n"
    # Windows per generate() call; halved on OOM at runtime. Values match the
    # reference evaluation (an image window holds 4 full-page images).
    text_window_batch_size = 16
    image_window_batch_size = 8

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        *,
        use_think_block: bool = False,
        window_size: int = 4,
        stride: int = 2,
        max_new_tokens: int = 30,
        max_length: int = 2048,
        **kwargs: Any,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        kwargs.setdefault("dtype", torch.bfloat16)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_name, revision=revision, **kwargs
            )
            .to(self.device)
            .eval()
        )
        # Batched generation needs left padding (completions are sliced at the
        # prompt length); the pinned revisions set padding_side in tokenizer_config.
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)

        self.use_think_block = use_think_block
        self.window_size = window_size
        self.stride = stride
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        # A malformed generation silently falls back to the input order, which drags
        # scores toward the first-stage ranking with no error raised. Track fallbacks
        # so silent regressions stay visible.
        self.parse_stats = {"calls": 0, "fallbacks": 0}

    def _parse_permutation(self, text: str, window_len: int) -> list[int]:
        self.parse_stats["calls"] += 1
        strict = re.compile(r"\s*>\s*".join([r"\[(\d+)\]"] * window_len))
        match = strict.search(text)
        if match:
            order = [int(x) - 1 for x in match.groups()]
            if sorted(order) == list(range(window_len)):
                return order
        self.parse_stats["fallbacks"] += 1
        return list(range(window_len))

    def _build_text_prompt(self, query: str, docs: list[str]) -> str:
        body = "\n".join(f"[{i + 1}]: {d}" for i, d in enumerate(docs))
        user = f"Query: {query}\n\nRank these passages from most to least relevant:\n{body}\n\nRanking:"
        prompt = self.prompt_template.format(user=user)
        if self.use_think_block:
            prompt += self.think_block
        return prompt

    def _build_multimodal_window(
        self, query: str, docs: list[str | Image.Image]
    ) -> tuple[str, list[Image.Image]]:
        """Render a window containing at least one image via the processor's chat template."""
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Query: {query}\n\nRank these documents from most to least relevant:\n",
            }
        ]
        images = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                content.append({"type": "text", "text": f"[{i + 1}]: {doc}\n"})
            else:
                content.extend(
                    [
                        {"type": "text", "text": f"[{i + 1}]: "},
                        {"type": "image", "image": doc},
                        {"type": "text", "text": "\n"},
                    ]
                )
                images.append(doc)
        content.append({"type": "text", "text": "\nRanking:"})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.use_think_block and not text.endswith(self.think_block):
            text += self.think_block
        return text, images

    @torch.inference_mode()
    def _generate(
        self, texts: list[str], images: list[Image.Image] | None
    ) -> list[str]:
        processor_kwargs: dict[str, Any] = dict(
            text=texts, return_tensors="pt", padding=True
        )
        if images:
            processor_kwargs["images"] = images
        else:
            processor_kwargs.update(truncation=True, max_length=self.max_length)
        inputs = self.processor(**processor_kwargs).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return [
            self.processor.decode(out[i][input_len:], skip_special_tokens=True)
            for i in range(len(texts))
        ]

    def _rank_windows_batched(
        self,
        windows: list[tuple[str, list[str | Image.Image]]],
        batch_size: int | None = None,
    ) -> list[list[int]]:
        """Rank a batch of (query, window_docs) windows, halving the batch on OOM.

        One batch unit is one window (window_size documents per generate() call);
        when batch_size is None the per-modality default is used.
        """
        results: list[list[int] | None] = [None] * len(windows)
        has_images = any(
            not isinstance(doc, str) for _, docs in windows for doc in docs
        )
        if batch_size is None:
            batch_size = (
                self.image_window_batch_size
                if has_images
                else self.text_window_batch_size
            )
        i = 0
        while i < len(windows):
            sub = windows[i : i + batch_size]
            if has_images:
                texts, flat_images = [], []
                for query, docs in sub:
                    text, images = self._build_multimodal_window(query, docs)
                    texts.append(text)
                    flat_images.extend(images)
            else:
                texts = [self._build_text_prompt(q, docs) for q, docs in sub]
                flat_images = None
            try:
                decoded = self._generate(texts, flat_images)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "OOM during listwise window generation; retrying with batch size %d",
                    batch_size,
                )
                continue
            for j, output in enumerate(decoded):
                results[i + j] = self._parse_permutation(output, len(sub[j][1]))
            i += len(sub)
        return results

    def _sliding_window_rank(
        self,
        queries: list[str],
        doc_lists: list[list[str | Image.Image]],
        show_progress_bar: bool = True,
        batch_size: int | None = None,
    ) -> list[list[int]]:
        """Rerank each query's documents; returns per-query ordering of original indices.

        Bottom-to-top single pass: windows within a query are sequential (each sees
        prior windows' updates); queries at the same window position are batched.
        """
        orders = [list(range(len(docs))) for docs in doc_lists]
        max_n = max((len(docs) for docs in doc_lists), default=0)
        if max_n <= 1:
            return orders

        start = max(0, max_n - self.window_size)
        positions = list(range(start, -1, -self.stride))
        if positions[-1] != 0:
            positions.append(0)

        # Window positions are derived from the longest candidate list; for shorter
        # lists the clamp maps several global positions onto the same (p, end)
        # window. Re-ranking a window on its own output is not idempotent, so each
        # query processes a given window at most once (single-pass protocol).
        last_window: list[tuple[int, int] | None] = [None] * len(orders)

        for pos in tqdm(
            positions,
            desc="Sliding-window positions",
            disable=not show_progress_bar,
        ):
            windows, meta = [], []
            for query_idx, order in enumerate(orders):
                n = len(order)
                end = min(pos + self.window_size, n)
                p = max(0, end - self.window_size)
                if end - p < 2 or last_window[query_idx] == (p, end):
                    continue
                last_window[query_idx] = (p, end)
                window_docs = [doc_lists[query_idx][order[k]] for k in range(p, end)]
                windows.append((queries[query_idx], window_docs))
                meta.append((query_idx, p, end))
            if not windows:
                continue
            permutations = self._rank_windows_batched(windows, batch_size=batch_size)
            for (query_idx, p, end), perm in zip(meta, permutations):
                order = orders[query_idx]
                order[p:end] = [order[p + j] for j in perm]
        return orders

    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Array:
        queries: list[str] = []
        for batch in inputs1:
            if batch.get("image") is not None:
                raise ValueError(
                    "LightOn-rerank models support text queries only; "
                    "got queries with an image modality."
                )
            queries.extend(batch["text"])
        documents: list[str | Image.Image] = []
        for batch in inputs2:
            images = batch.get("image")
            texts = batch.get("text")
            if images is not None and texts is not None:
                raise ValueError(
                    "LightOn-rerank models score each candidate as either a text "
                    "passage or a page image, not both; this task provides documents "
                    "with both modalities. Use an image-only or text-only variant."
                )
            documents.extend(images if images is not None else texts)
        group_queries: list[str] = []
        group_docs: list[list[str | Image.Image]] = []
        for query, document in zip(queries, documents):
            if not group_queries or query != group_queries[-1]:
                group_queries.append(query)
                group_docs.append([])
            group_docs[-1].append(document)

        orders = self._sliding_window_rank(
            group_queries,
            group_docs,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )

        scores: list[float] = []
        for order in orders:
            n = len(order)
            doc_scores = [0.0] * n
            for rank, original_idx in enumerate(order):
                doc_scores[original_idx] = float(n - rank)
            scores.extend(doc_scores)

        if self.parse_stats["fallbacks"]:
            logger.warning(
                "Listwise permutation parse fallbacks: %d/%d windows fell back to input order.",
                self.parse_stats["fallbacks"],
                self.parse_stats["calls"],
            )
        return np.asarray(scores)


LIGHTON_CITATION = r"""@misc{ananya2026lightonrerank,
  title={One Adapter, Both Modalities: Field Notes from Building and Serving a Multimodal Reranker},
  author={Ananya, Ishrat Jahan and Chatelain, Amelie},
  year={2026},
  howpublished={\url{https://huggingface.co/blog/lightonai/lighton-rerank}},
}"""

# MS MARCO and NQ for text; the visual-document training mix overlaps the ViDoRe v1
# train splits (same lineage as the colpali train set).
lighton_rerank_training_data = {
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NQ",
    "NQHardNegatives",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
}

_LIGHTON_COMMON = dict(
    languages=["eng-Latn"],
    open_weights=True,
    release_date="2026-07-08",
    license="apache-2.0",
    max_tokens=16384,
    embed_dim=None,
    similarity_fn_name=None,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=lighton_rerank_training_data,
    modalities=["image", "text"],
    model_type=["cross-encoder"],
    citation=LIGHTON_CITATION,
    superseded_by=None,
    contacts=["coreprinciple6"],
)

_PW_COMMON = dict(
    loader=CrossEncoderWrapper,
    loader_kwargs=dict(
        model_kwargs=dict(dtype=torch.bfloat16),
    ),
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    **_LIGHTON_COMMON,
)

_LW_COMMON = dict(
    loader=LightOnListwiseRerankerWrapper,
    framework=["PyTorch", "Transformers", "safetensors"],
    **_LIGHTON_COMMON,
)

lighton_rerank_pw_0_8b = ModelMeta(
    name="lightonai/LightOn-rerank-PW-0.8B",
    revision="e2459c2554b95969a68cba3aa6f0bd696723934f",
    n_parameters=852_985_920,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=1627,
    reference="https://huggingface.co/lightonai/LightOn-rerank-PW-0.8B",
    adapted_from="Qwen/Qwen3.5-0.8B",
    **_PW_COMMON,
)

lighton_rerank_pw_2b = ModelMeta(
    name="lightonai/LightOn-rerank-PW-2B",
    revision="c5333533d5ebea2142490d5eacd42f0c8363c808",
    n_parameters=2_213_241_664,
    n_embedding_parameters=508_559_360,
    memory_usage_mb=4221,
    reference="https://huggingface.co/lightonai/LightOn-rerank-PW-2B",
    adapted_from="Qwen/Qwen3.5-2B",
    **_PW_COMMON,
)

lighton_rerank_pw_4b = ModelMeta(
    name="lightonai/LightOn-rerank-PW-4B",
    revision="8f0e0c2d506734bf61ff67ab3c4553efebddba39",
    n_parameters=4_539_265_536,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8658,
    reference="https://huggingface.co/lightonai/LightOn-rerank-PW-4B",
    adapted_from="Qwen/Qwen3.5-4B",
    **_PW_COMMON,
)

lighton_rerank_lw_0_8b = ModelMeta(
    name="lightonai/LightOn-rerank-LW-0.8B",
    revision="b6b897c646a9ea980692fb3278ffa04867919b8c",
    n_parameters=852_985_920,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=1627,
    reference="https://huggingface.co/lightonai/LightOn-rerank-LW-0.8B",
    adapted_from="Qwen/Qwen3.5-0.8B",
    **_LW_COMMON,
)

lighton_rerank_lw_2b = ModelMeta(
    name="lightonai/LightOn-rerank-LW-2B",
    revision="4629d733e8381a7d2c8cfba17378ce3ceaf0edd4",
    n_parameters=2_213_241_664,
    n_embedding_parameters=508_559_360,
    memory_usage_mb=4221,
    reference="https://huggingface.co/lightonai/LightOn-rerank-LW-2B",
    adapted_from="Qwen/Qwen3.5-2B",
    **_LW_COMMON,
)

lighton_rerank_lw_4b = ModelMeta(
    name="lightonai/LightOn-rerank-LW-4B",
    revision="92b0661fa69e4820ac3647548a7f8e940b83cd41",
    n_parameters=4_539_265_536,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8658,
    reference="https://huggingface.co/lightonai/LightOn-rerank-LW-4B",
    adapted_from="Qwen/Qwen3.5-4B",
    loader_kwargs=dict(use_think_block=True),
    **_LW_COMMON,
)
