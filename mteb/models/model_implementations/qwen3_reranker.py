from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from vllm import SamplingParams

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class Qwen3RerankerWrapper:
    """Wrapper for Qwen3 Reranker models."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        self.max_length = self.tokenizer.model_max_length
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

    @staticmethod
    def format_instruction(instruction: str | None, query: str, doc: str) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def process_inputs(self, pairs: list[str]) -> dict:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs: dict) -> list[float]:
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

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
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> Array:
        queries = [text for batch in inputs1 for text in batch["text"]]
        instructions = None
        if "instruction" in inputs1.dataset.features:
            instructions = [text for batch in inputs1 for text in batch["instruction"]]
        passages = [text for batch in inputs2 for text in batch["text"]]

        all_scores = []
        for i in tqdm(
            range(0, len(queries), batch_size),
            disable=not show_progress_bar,
            desc="Computing relevance scores",
        ):
            batch_queries = queries[i : i + batch_size]
            batch_passages = passages[i : i + batch_size]
            batch_instructions = (
                instructions[i : i + batch_size]
                if instructions is not None
                else [None] * len(batch_queries)
            )

            pairs = [
                self.format_instruction(instr, query, doc)
                for instr, query, doc in zip(
                    batch_instructions, batch_queries, batch_passages
                )
            ]

            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
            all_scores.extend(scores)

        return np.array(all_scores)


qwen3_reranker_training_data = {
    # source: https://arxiv.org/pdf/2506.05176
    "MIRACLReranking",
    "DuRetrieval",
    "MrTidyRetrieval",
    "T2Reranking",
    "MSMARCO",
    "NQ",
    "HotpotQA",
    "CodeSearchNet",
    "MultiLongDocRetrieval",
    # "NLI",
    # "simclue",
    # "multi-cpr",
    # + synthetic data
}

QWEN3_CITATION = """@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}"""

qwen3_reranker_0_6b = ModelMeta(
    loader=Qwen3RerankerWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="Qwen/Qwen3-Reranker-0.6B",
    revision="6e9e69830b95c52b5fd889b7690dda3329508de3",
    release_date="2025-05-29",
    languages=None,
    n_parameters=595776512,
    n_embedding_parameters=155309056,
    memory_usage_mb=1136.0,
    max_tokens=40960,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=qwen3_reranker_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=QWEN3_CITATION,
    contacts=None,
)

qwen3_reranker_4b = ModelMeta(
    loader=Qwen3RerankerWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="Qwen/Qwen3-Reranker-4B",
    revision="f16fc5d5d2b9b1d0db8280929242745d79794ef5",
    release_date="2025-06-03",
    languages=None,
    n_parameters=4021784576,
    n_embedding_parameters=388272640,
    memory_usage_mb=7671.0,
    max_tokens=40960,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Reranker-4B",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=qwen3_reranker_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=QWEN3_CITATION,
    contacts=None,
)

qwen3_reranker_8b = ModelMeta(
    loader=Qwen3RerankerWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="Qwen/Qwen3-Reranker-8B",
    revision="5fa94080caafeaa45a15d11f969d7978e087a3db",
    release_date="2025-05-29",
    languages=None,
    n_parameters=8188548096,
    n_embedding_parameters=621236224,
    memory_usage_mb=15618.0,
    max_tokens=40960,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Reranker-8B",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=qwen3_reranker_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=QWEN3_CITATION,
    contacts=None,
)
