from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

GME_CITATION = """@misc{zhang2024gme,
      title={GME: Improving Universal Multimodal Retrieval by Multimodal LLMs},
      author={Zhang, Xin and Zhang, Yanzhao and Xie, Wen and Li, Mingxin and Dai, Ziqi and Long, Dingkun and Xie, Pengjun and Zhang, Meishan and Li, Wenjie and Zhang, Min},
      year={2024},
      eprint={2412.16855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={http://arxiv.org/abs/2412.16855}
}"""


class GmeQwen2VL(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        model_path: str | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = model_path or model_name
        self.model: SentenceTransformer = SentenceTransformer(
            model_name, revision=revision, trust_remote_code=trust_remote_code, **kwargs
        )
        self.model.eval()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        instruction = self.get_instruction(task_metadata, prompt_type)

        # Don't use instruction for documents
        if prompt_type == PromptType.document:
            instruction = None
        elif instruction is not None and isinstance(instruction, str):
            # Ensure instruction ends with a period
            if instruction[-1] != ".":
                instruction += "."

        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            has_text = "text" in batch
            has_image = "image" in batch

            # Prepare batch inputs for SentenceTransformer
            batch_inputs = []

            if has_text and has_image:
                # Fused-modal: both text and image
                for text, image in zip(batch["text"], batch["image"]):
                    item = {"text": text, "image": image}
                    if instruction:
                        item["prompt"] = instruction
                    batch_inputs.append(item)
            elif has_text:
                # Text-only
                for text in batch["text"]:
                    item = {"text": text}
                    if instruction:
                        item["prompt"] = instruction
                    batch_inputs.append(item)
            elif has_image:
                # Image-only
                for image in batch["image"]:
                    item = {"image": image}
                    if instruction:
                        item["prompt"] = instruction
                    batch_inputs.append(item)
            else:
                raise ValueError("Batch must contain either 'text' or 'image'")

            # Encode using SentenceTransformer
            embeddings = self.model.encode(
                batch_inputs,
                convert_to_tensor=True,
                show_progress_bar=False,
                **kwargs,
            )

            all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.numpy()


training_data = {
    "MSMARCO",
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQAHardNegatives",
    # TriviaQA (Joshi et al., 2017),
    # SQuAD (Rajpurkar et al., 2016),
    "FEVER",
    # AllNLI for SimCSE (Gao et al., 2021), selecting a total of 1 million entries.
    # ImageNet (Deng et al., 2009)
    # LAION (Schuhmann et al., 2022),
    # mscoco (Lin et al., 2014),
    # Docmatix (Laurenc¸on et al., 2024)
    # synthetic data
    # M-BEIR (Wei et al., 2024)
}


gme_qwen2vl_2b = ModelMeta(
    loader=GmeQwen2VL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    model_type=["dense"],
    languages=["eng-Latn", "cmn-Hans"],
    open_weights=True,
    revision="9cfa6413f704a7c1cf5064d240748e10c876b286",
    release_date="2024-12-24",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    memory_usage_mb=8427,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Sentence Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=GME_CITATION,
)

gme_qwen2vl_7b = ModelMeta(
    loader=GmeQwen2VL,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    model_type=["dense"],
    languages=["eng-Latn", "cmn-Hans"],
    open_weights=True,
    revision="e54cb53a76dba4895a7a2f88fc8021f3679ed4f1",
    release_date="2024-12-24",
    modalities=["image", "text"],
    n_parameters=8_290_000_000,
    memory_usage_mb=31629,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Sentence Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=GME_CITATION,
)
