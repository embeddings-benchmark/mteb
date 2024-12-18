from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class NomicWrapper(Wrapper):
    """following the hf model card documentation."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        input_type = self.get_prompt_name(self.model_prompts, task_name, prompt_type)

        # default to search_document if input_type and prompt_name are not provided
        if input_type is None:
            input_type = "search_document"

        sentences = [f"{input_type}: {sentence}" for sentence in sentences]

        emb = self.model.encode(sentences, batch_size=batch_size, **kwargs)
        # v1.5 has a non-trainable layer norm to unit normalize the embeddings for binary quantization
        # the outputs are similar to if we just normalized but keeping the same for consistency
        if self.model_name == "nomic-ai/nomic-embed-text-v1.5":
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
            emb = F.normalize(emb, p=2, dim=1)
            if kwargs.get("convert_to_tensor", False):
                emb = emb.cpu().detach().numpy()

        return emb


model_prompts = {
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    PromptType.query.value: "search_query: ",
    PromptType.passage.value: "search_document: ",
}

NOMIC_CITATION = """
@misc{nussbaum2024nomic,
      title={Nomic Embed: Training a Reproducible Long Context Text Embedder}, 
      author={Zach Nussbaum and John X. Morris and Brandon Duderstadt and Andriy Mulyar},
      year={2024},
      eprint={2402.01613},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

nomic_embed_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        revision="b0753ae76394dd36bcfb912a46018088bca48be0",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b0753ae76394dd36bcfb912a46018088bca48be0",
    release_date="2024-02-10",  # first commit
    citation=NOMIC_CITATION,
    n_parameters=137_000_000,
    memory_usage=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
)

nomic_embed_v1 = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1",
        revision="0759316f275aa0cb93a5b830973843ca66babcf5",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0759316f275aa0cb93a5b830973843ca66babcf5",
    release_date="2024-01-31",  # first commit
    n_parameters=None,
    memory_usage=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    citation=NOMIC_CITATION,
    adapted_from=None,
    superseded_by="nomic-ai/nomic-embed-text-v1.5",
)

nomic_embed_v1_ablated = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1-ablated",
        revision="7d948905c5d5d3874fa55a925d68e49dbf411e5f",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1-ablated",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7d948905c5d5d3874fa55a925d68e49dbf411e5f",
    release_date="2024-01-15",  # first commit
    n_parameters=None,
    memory_usage=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1-ablated",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
)


nomic_embed_v1_ablated = ModelMeta(
    loader=partial(  # type: ignore
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1-unsupervised",
        revision="b53d557b15ae63852847c222d336c1609eced93c",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1-unsupervised",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b53d557b15ae63852847c222d336c1609eced93c",
    release_date="2024-01-15",  # first commit
    n_parameters=None,
    memory_usage=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
)
