from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.model_meta import ModelMeta

from .rerankers_custom import RerankerWrapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import BatchedInput, PromptType

logger = logging.getLogger(__name__)


class QueritWrapper(RerankerWrapper):
    """
    Multi-GPU / multi-process reranker wrapper for mteb.mteb evaluation.
    Supports flattening all query-passage pairs without explicit grouping.
    """

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        from transformers import AutoModel, AutoTokenizer

        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_args
        )
        logger.info(f"Using model {model_name}")

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if "[CLS]" not in self.tokenizer.get_vocab():
            raise ValueError("Tokenizer missing required special token '[CLS]'")
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.pad_token_id = self.tokenizer.pad_token_id or 0

        self.max_length = (
            min(kwargs.get("max_length", 4096), self.tokenizer.model_max_length) - 1
        )  # sometimes it's a v large number/max int
        logger.info(f"Using max_length of {self.max_length}, 1 token for [CLS]")
        self.model.eval()

    def process_inputs(
        self,
        pairs: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Encode a batch of (query, document) pairs:
        - Concatenate prompt + Query + Content
        - Append [CLS] at the end
        - Left-pad to max_length
        - Generate custom attention mask based on block types
        """
        # Construct input texts
        enc = self.tokenizer(
            pairs,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids_list: list[list[int]] = []
        attn_mask_list: list[torch.Tensor] = []

        for ids in enc["input_ids"]:
            # Append [CLS] token
            ids = ids + [self.cls_token_id]
            block_types = [1] * (len(ids) - 1) + [2]  # content + CLS

            # Pad or truncate
            if len(ids) < self.max_length:
                pad_len = self.max_length - len(ids)
                ids = [self.pad_token_id] * pad_len + ids
                block_types = [0] * pad_len + block_types
            else:
                ids = ids[-self.max_length :]
                block_types = block_types[-self.max_length :]

            attn = self.compute_mask_content_cls(block_types)
            input_ids_list.append(ids)
            attn_mask_list.append(attn)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        attention_mask = torch.stack(attn_mask_list, dim=0).to(self.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

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
        **kwargs: Any,
    ) -> list[float]:
        """
        Predict relevance scores for query-passage pairs.
        Supports both single-process and multi-process/multi-GPU modes.
        """
        # Flatten all pairs from mteb.mteb DataLoaders
        queries = [text for batch in inputs1 for text in batch["text"]]
        passages = [text for batch in inputs2 for text in batch["text"]]

        instructions = None
        if "instruction" in inputs2.dataset.features:
            instructions = [text for batch in inputs1 for text in batch["instruction"]]

        num_pairs = len(queries)
        if num_pairs == 0:
            return []
        final_scores: list[float] = []

        batch_size = kwargs.get("batch_size", self.batch_size)
        with tqdm(total=num_pairs, desc="Scoring", ncols=100) as pbar:
            for start in range(0, num_pairs, batch_size):
                end = min(start + batch_size, num_pairs)
                batch_q = queries[start:end]
                batch_d = passages[start:end]

                batch_instructions = (
                    instructions[start:end]
                    if instructions is not None
                    else [None] * len(batch_q)
                )
                pairs = [
                    self.format_instruction(instr, query, doc)
                    for instr, query, doc in zip(batch_instructions, batch_q, batch_d)
                ]
                enc = self.process_inputs(pairs)
                out = self.model(**enc)
                scores = out["score"].squeeze(-1).detach().float().cpu().tolist()

                if not isinstance(scores, list):
                    scores = [scores]

                final_scores.extend(scores)
                pbar.update(len(scores))

        return final_scores

    @staticmethod
    def format_instruction(instruction: str | None, query: str, doc: str) -> str:
        if instruction is None:
            output = f"Judge whether the Content meets the requirements based on the Query. Query: {query}; Content: {doc}"
        else:
            output = f"{instruction} Query: {query}; Content: {doc}"
        return output

    @staticmethod
    def compute_mask_content_cls(block_types: list[int]) -> torch.Tensor:
        """
        Create custom attention mask based on token block types:
        - 0: padding   → ignored
        - 1: content   → causal attention to previous content only
        - 2: [CLS]     → causal attention to all non-padding tokens

        Args:
            block_types: List of token types for one sequence

        Returns:
            [1, seq_len, seq_len] boolean attention mask (True = allowed to attend)
        """
        pos = torch.tensor(block_types, dtype=torch.long)
        n = pos.shape[0]
        if n == 0:
            return torch.empty((0, 0), dtype=torch.bool, device=pos.device)

        row_types = pos.view(n, 1)
        col_types = pos.view(1, n)

        row_idx = torch.arange(n, device=pos.device).view(n, 1)
        col_idx = torch.arange(n, device=pos.device).view(1, n)
        causal_mask = col_idx <= row_idx

        # Content tokens only attend to previous content
        mask_content = (row_types == 1) & (col_types == 1) & causal_mask

        # [CLS] attends to all non-pad tokens (causal)
        mask_cls = (row_types == 2) & (col_types != 0) & causal_mask

        type_mask = mask_content | mask_cls
        return type_mask.unsqueeze(0)


querit_reranker_training_data = {
    "MIRACLRanking",  # https://huggingface.co/datasets/mteb/MIRACLReranking
    "MrTidyRetrieval",  # https://huggingface.co/datasets/mteb/mrtidy
    "ruri-v3-dataset-reranker",  # https://huggingface.co/datasets/cl-nagoya/ruri-v3-dataset-reranker
    "MultiLongDocReranking",  # https://huggingface.co/datasets/Shitao/MLDR
    "MindSmallReranking",  # https://huggingface.co/datasets/mteb/MindSmallReranking
    "MSMARCO",  # https://huggingface.co/datasets/mteb/msmarco
    "CQADupStack",  # https://huggingface.co/datasets/mteb/cqadupstack-*
    "AskUbuntuDupQuestions",  # https://github.com/taolei87/askubuntu & The corpus and queries that overlap with mteb/askubuntudupquestions-reranking have been removed.
    "T2Reranking",  # https://huggingface.co/datasets/THUIR/T2Ranking & The corpus and queries that overlap with mteb/T2Reranking have been removed.
}

model_meta = ModelMeta(
    loader=QueritWrapper,
    loader_kwargs={
        "fp_options": "bfloat16",
    },
    name="Querit/Querit",
    model_type=["cross-encoder"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="5ad2649cc4defb7e1361262260e9a781f14b08bc",
    release_date="2026-01-24",
    n_parameters=4919636992,
    n_embedding_parameters=131907584,
    embed_dim=1024,
    memory_usage_mb=9383.0,
    max_tokens=4096,
    reference="https://huggingface.co/Querit/Querit",
    similarity_fn_name=None,
    training_datasets=querit_reranker_training_data,
    license="apache-2.0",
    framework=["PyTorch"],
    use_instructions=None,
    public_training_code=None,
    public_training_data=None,
    citation=None,
)
