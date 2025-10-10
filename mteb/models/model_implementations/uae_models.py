import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class UAEWrapper(SentenceTransformerEncoderWrapper):
    """following the hf model card documentation."""

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
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        sentences = [text for batch in inputs for text in batch["text"]]

        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_metadata.name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")
        if prompt_name and prompt_name in self.model.prompts:
            prompt = self.model.prompts[prompt_name]
            sentences = [prompt.format(text=sentence) for sentence in sentences]

        embeddings = self.model.encode(
            sentences,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


uae_large_v1 = ModelMeta(
    loader=UAEWrapper,
    loader_kwargs=dict(
        # https://github.com/SeanLee97/AnglE/blob/b04eae166d8596b47293c75b4664d3ad820d7331/angle_emb/angle.py#L291-L314
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: {text}",
            "Summarization": 'Summarize sentence "{text}" in one word:"',
        },
    ),
    name="WhereIsAI/UAE-Large-V1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
    release_date="2023-12-04",  # initial commit of hf model.
    n_parameters=int(335 * 1e6),
    memory_usage_mb=1278,
    max_tokens=512,
    embed_dim=1024,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/WhereIsAI/UAE-Large-V1",
    use_instructions=True,
    citation="""
    @article{li2023angle,
      title={AnglE-optimized Text Embeddings},
      author={Li, Xianming and Li, Jing},
      journal={arXiv preprint arXiv:2309.12871},
      year={2023}
    }
    """,
    training_datasets={
        # source: https://arxiv.org/pdf/2309.12871
        # not in MTEB
        "MNLI",
        "NLI",
        "SNLI",
    },
    public_training_code=None,
    public_training_data=None,
)
