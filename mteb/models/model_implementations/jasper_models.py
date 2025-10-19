import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

from .nvidia_models import nvidia_training_datasets

logger = logging.getLogger(__name__)


class JasperModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str | Callable[[str], str] | None = None,
        max_seq_length: int = 2048,
        **kwargs: Any,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.instruction_template = instruction_template
        self.model.max_seq_length = max_seq_length

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
        instruction = self.get_task_instruction(task_metadata, prompt_type)

        # to passage prompts won't be applied to passages
        if prompt_type == PromptType.document:
            instruction = None
        inputs = [text for batch in inputs for text in batch["text"]]

        embeddings = self.model.encode(
            inputs,
            normalize_embeddings=True,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


jasper_en_v1 = ModelMeta(
    loader=JasperModel,
    loader_kwargs=dict(
        config_kwargs={"is_text_encoder": True, "vector_dim": 12288},
        model_kwargs={
            "attn_implementation": "sdpa",
            "torch_dtype": torch.bfloat16,
        },
        trust_remote_code=True,
        max_seq_length=2048,
        instruction_template="Instruct: {instruction}\nQuery: ",
    ),
    name="NovaSearch/jasper_en_vision_language_v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d6330ce98f8a0d741e781df845904c9484f00efa",
    release_date="2024-12-11",  # first commit
    n_parameters=1_999_000_000,
    memory_usage_mb=3802,
    max_tokens=131072,
    embed_dim=8960,
    license="apache-2.0",
    reference="https://huggingface.co/infgrad/jasper_en_vision_language_v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets=set(
        # stage 1, 2, 3
        #  "In jasper model the teacher model is nvidia/NV-Embed-v2", source https://huggingface.co/NovaSearch/jasper_en_vision_language_v1
        # fineweb-edu
        # https://huggingface.co/datasets/sentence-transformers/embedding-training-data
        # stage 4
        # BAAI/Infinity-MM
    )
    | nvidia_training_datasets,
    # training logs https://api.wandb.ai/links/dunnzhang0/z8jqoqpb
    # more codes https://huggingface.co/NovaSearch/jasper_en_vision_language_v1/commit/da9b77d56c23d9398fa8f93af449102784f74e1d
    public_training_code="https://github.com/NovaSearch-Team/RAG-Retrieval/blob/c40f4638b705eb77d88305d2056901ed550f9f4b/rag_retrieval/train/embedding/README.md",
    public_training_data="https://huggingface.co/datasets/infgrad/jasper_text_distill_dataset",
    citation="""
@misc{zhang2025jasperstelladistillationsota,
      title={Jasper and Stella: distillation of SOTA embedding models},
      author={Dun Zhang and Jiacheng Li and Ziyang Zeng and Fulong Wang},
      year={2025},
      eprint={2412.19048},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.19048},
}
""",
)
