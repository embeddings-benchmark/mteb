from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Literal

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.colpali_models import COLPALI_TRAINING_DATA
from mteb.models.model_implementations.pylate_models import (
    denseon_lateon_supervised_data,
    mdenseon_mlateon_code_data,
    mdenseon_mlateon_organic_data,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.similarity_functions import (
    _token_budget_chunks,
    cos_sim,
    max_sim,
    pairwise_cos_sim,
    pairwise_max_sim,
)
from mteb.types import PromptType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

_EMBED_DIMS: dict[str, int] = {"single_vector": 1024, "multi_vector": 128}


def _query_lengths(queries: Array, device: torch.device) -> torch.Tensor:
    import torch

    return torch.tensor([len(query) for query in queries], device=device)


class NeoMMERetrieverWrapper(AbsEncoder):
    """Encode text and images with the native Transformers NeoMME retriever.

    Args:
        vector_type: "multi_vector" (128-d token vectors scored with MaxSim) or
            "single_vector" (one 1024-d vector scored with cosine).
        embed_dim: Must match vector_type if given. Leave it unset with
            "single_vector": ModelMeta.load_model checks it against the
            registered 128.
        max_text_tokens: Text is cut to this many tokens before the chat template
            is applied, as in the paper. Image pages are not truncated.
        max_batch_tokens: Upper bound on padded tokens per text batch.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        vector_type: Literal["single_vector", "multi_vector"] = "multi_vector",
        embed_dim: int | None = None,
        torch_dtype: str | torch.dtype | None = None,
        max_text_tokens: int = 8192,
        max_batch_tokens: int = 65536,
        **kwargs: Any,
    ) -> None:
        if vector_type not in _EMBED_DIMS:
            raise ValueError(
                f"vector_type must be one of {list(_EMBED_DIMS)}, got {vector_type!r}"
            )
        dimension = _EMBED_DIMS[vector_type]
        if embed_dim is not None and embed_dim != dimension:
            raise ValueError(
                f"{vector_type} gives {dimension}-d vectors, got embed_dim={embed_dim}. "
                "Leave embed_dim unset: vector_type sets the dimension."
            )

        import torch
        from transformers import NeoMMEForRetrieval, NeoMMEProcessor

        self._mteb_model_meta: ModelMeta | None = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vector_type = vector_type
        self.max_text_tokens = max_text_tokens
        self.max_batch_tokens = max_batch_tokens
        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16
                if torch.device(self.device).type == "cuda"
                and torch.cuda.is_bf16_supported()
                else torch.float32
            )
        self.model = (
            NeoMMEForRetrieval.from_pretrained(
                model_name,
                revision=revision,
                dtype=torch_dtype,
                **kwargs,
            )
            .to(self.device)
            .eval()
        )
        self.processor = NeoMMEProcessor.from_pretrained(model_name, revision=revision)

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta | None) -> None:
        if meta is None:
            self._mteb_model_meta = None
            return
        if self.vector_type == "single_vector":
            self._mteb_model_meta = meta.model_copy(
                update={
                    "embed_dim": _EMBED_DIMS["single_vector"],
                    "model_type": ["dense"],
                    "similarity_fn_name": ScoringFunction.COSINE,
                },
                deep=True,
            )
            return
        # MTEB saves results under experiments/ when experiment_kwargs is set.
        # Dropping the default head keeps multi-vector results in the plain
        # revision folder, which the leaderboard reads.
        experiment_kwargs = dict(meta.experiment_kwargs or {})
        experiment_kwargs.pop("vector_type", None)
        self._mteb_model_meta = meta.model_copy(
            update={"experiment_kwargs": experiment_kwargs or None}, deep=True
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode text in length-sorted batches and images in the loader's batches."""
        import torch

        task = "document" if prompt_type == PromptType.document else "query"
        texts: dict[int, str] = {}
        image_parts: list[dict[int, torch.Tensor]] = []
        n_rows = 0
        with torch.inference_mode():
            # Encoding images per loader batch avoids holding every decoded page.
            for batch in inputs:
                batch_texts, batch_images, n_batch_rows = self._split_batch(
                    batch, n_rows
                )
                texts.update(batch_texts)
                if batch_images:
                    # Pages are not cut to max_text_tokens, because cutting a page
                    # would drop part of its image grid.
                    image_parts.append(self._encode_contents(batch_images, "document"))
                n_rows += n_batch_rows
            text_part = self._encode_texts(texts, task)
        embeddings = self._combine([text_part, *image_parts], n_rows)
        if self.vector_type == "multi_vector":
            return embeddings
        return torch.stack(embeddings).numpy()

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Use cosine for dense vectors and MaxSim averaged over query tokens for token vectors."""
        if self.vector_type == "single_vector":
            return cos_sim(embeddings1, embeddings2)
        scores = max_sim(embeddings1, embeddings2)
        return scores / _query_lengths(embeddings1, scores.device)[:, None]

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Score corresponding pairs with the same rule as retrieval."""
        if self.vector_type == "single_vector":
            return pairwise_cos_sim(embeddings1, embeddings2)
        scores = pairwise_max_sim(embeddings1, embeddings2)
        return scores / _query_lengths(embeddings1, scores.device)

    def _split_batch(
        self, batch: BatchedInput, offset: int
    ) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]], int]:
        """Return the text and image contents of a loader batch, keyed by row."""
        n_rows = len(batch.get("text") or batch["image"])
        texts = batch.get("text", [""] * n_rows)
        images = batch.get("image", [None] * n_rows)
        image_token = self.processor.image_token
        text_contents: dict[int, str] = {}
        image_contents: dict[int, list[dict[str, Any]]] = {}
        for index, (text, image) in enumerate(
            zip(texts, images, strict=True), start=offset
        ):
            if image is not None:
                image_contents[index] = [{"type": "image", "image": image}]
            # MTEB's collator rejects None in text columns, so rows with only an
            # image have "" as text.
            if text or image is None:
                # The chat template raises an error when text contains the image
                # token "<img>", which marks where a page image goes. Some
                # CQADupStack posts contain a bare "<img>" HTML tag, so it is
                # escaped to "&lt;img&gt;".
                text_contents[index] = text.replace(
                    image_token, html.escape(image_token)
                )
        return text_contents, image_contents, n_rows

    def _truncate_texts(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Cut each text to max_text_tokens and return the texts and token counts."""
        offsets = self.processor.tokenizer(
            texts, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        truncated, lengths = [], []
        for text, text_offsets in zip(texts, offsets, strict=True):
            if len(text_offsets) <= self.max_text_tokens:
                truncated.append(text)
                lengths.append(len(text_offsets))
                continue
            # Cutting at the start of the first dropped token keeps characters
            # that span several tokens whole.
            truncated.append(text[: text_offsets[self.max_text_tokens][0]])
            lengths.append(self.max_text_tokens)
        return truncated, lengths

    def _encode_texts(
        self, texts: dict[int, str], task: str
    ) -> dict[int, torch.Tensor]:
        """Encode texts in batches of at most max_batch_tokens padded tokens."""
        if not texts:
            return {}
        row_indices = list(texts)
        # Truncating the templated sequence would drop the trailing query
        # expansion tokens, so the text is cut before the template is applied.
        truncated, lengths = self._truncate_texts(list(texts.values()))
        # The task marker and the query expansion tokens come on top of the text.
        overhead = len(
            self.processor.apply_chat_template(
                [[{"role": "user", "content": ""}]],
                task=task,
                tokenize=True,
                return_dict=True,
            )["input_ids"][0]
        )
        embeddings: dict[int, torch.Tensor] = {}
        for positions in _token_budget_chunks(
            [length + overhead for length in lengths], self.max_batch_tokens
        ):
            batch_texts = {
                row_indices[position]: truncated[position] for position in positions
            }
            embeddings.update(self._encode_contents(batch_texts, task))
        return embeddings

    def _combine(
        self, parts: list[dict[int, torch.Tensor]], n_rows: int
    ) -> list[torch.Tensor]:
        """Merge each row's text and image embeddings, in row order.

        Rows with both concatenate their token vectors (multi-vector) or add and
        renormalize their dense vectors (single-vector).
        """
        import torch

        embeddings: dict[int, torch.Tensor] = {}
        for part in parts:
            for index, embedding in part.items():
                existing = embeddings.get(index)
                if existing is None:
                    embeddings[index] = embedding
                elif self.vector_type == "single_vector":
                    embeddings[index] = torch.nn.functional.normalize(
                        existing + embedding, dim=-1
                    )
                else:
                    embeddings[index] = torch.cat((existing, embedding))
        return [embeddings[index] for index in range(n_rows)]

    def _encode_contents(
        self,
        contents: dict[int, str] | dict[int, list[dict[str, Any]]],
        task: str,
    ) -> dict[int, torch.Tensor]:
        """Wrap each content in a user message and encode them as one batch."""
        encoded = self.processor.apply_chat_template(
            [[{"role": "user", "content": content}] for content in contents.values()],
            task=task,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": "longest"},
        ).to(self.device)
        dense = self.vector_type == "single_vector"
        output = self.model(**encoded, output_dense=dense, output_multivector=not dense)
        if dense:
            return dict(
                zip(contents, output.dense_embeddings.float().cpu(), strict=True)
            )
        # One device-to-host copy per batch instead of one per row.
        embeddings = output.embeddings.float().cpu()
        masks = encoded["attention_mask"].bool().cpu()
        return {
            index: embedding[mask]
            for index, embedding, mask in zip(contents, embeddings, masks, strict=True)
        }


neomme_260m_retriever = ModelMeta(
    name="Hcompany/NeoMME-260M-Retriever",
    loader=NeoMMERetrieverWrapper,
    revision="481083153ac22931afbd614e672212bffbcb666e",
    release_date="2026-09-03",
    # English and the 20 FineWeb2-HQ languages in Table 8 of the paper.
    languages=[
        "eng-Latn",
        "deu-Latn",
        "fra-Latn",
        "spa-Latn",
        "rus-Cyrl",
        "cmn-Hani",
        "jpn-Jpan",
        "ita-Latn",
        "por-Latn",
        "pol-Latn",
        "nld-Latn",
        "ind-Latn",
        "tur-Latn",
        "ces-Latn",
        "vie-Latn",
        "swe-Latn",
        "fas-Arab",
        "arb-Arab",
        "ell-Grek",
        "dan-Latn",
        "hun-Latn",
    ],
    n_parameters=263_068_978,
    n_embedding_parameters=67_108_864,
    memory_usage_mb=502,
    max_tokens=16_384,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=False,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Hcompany/NeoMME-260M-Retriever",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    # Text: the LightOn fine-tuning mixture. Pages: ColPali, VisRAG and VDR.
    training_datasets=denseon_lateon_supervised_data
    | mdenseon_mlateon_organic_data
    | mdenseon_mlateon_code_data
    | COLPALI_TRAINING_DATA
    | {
        "VDRMultilingualRetrieval",
        "VisRAG-Ret-Train-Synthetic-data",
        "VisRAG-Ret-Train-In-domain-data",
    },
    adapted_from="Hcompany/NeoMME-260M",
    modalities=["image", "text"],
    model_type=["late-interaction"],
    extra_requirements_groups=["neomme", "image"],
    citation="""@misc{lac2026neomme,
    title={NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference},
    author={Aurélien Lac and Tony Wu},
    year={2026},
    eprint={2609.01657},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
    url={https://arxiv.org/abs/2609.01657},
}""",
)
