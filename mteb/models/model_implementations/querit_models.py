import json
import math
import os
import logging

from typing import Any, List, Tuple, Dict, Optional, TYPE_CHECKING

import pandas as pd
import torch
import torch.nn as nn
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from .rerankers_custom import RerankerWrapper
from mteb.types import Array, BatchedInput, PromptType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DeepseekV3ForCausalLM,
)


logger = logging.getLogger(__name__)

class QueritModel(DeepseekV3ForCausalLM):
    """Multi-task fine-tuned DeepSeekV3 model with a binary classification head for reranking."""

    def __init__(self, config, use_lm_head: bool = False):
        super().__init__(config)
        hidden_size = self.config.hidden_size

        # Binary classification head: relevant (1) vs irrelevant (0)
        self.head = nn.Linear(hidden_size, 2)

        # Optional language modeling head (usually disabled for reranking tasks)
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if use_lm_head
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        qids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for reranking / classification task.
        Returns loss (if labels and scores provided), ranking scores, predicted labels.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the last token's hidden state as representation (CLS-like pooling)
        cls_hidden = outputs.last_hidden_state[:, -1, :]   # [bs, hidden_size]
        logits = self.head(cls_hidden)                     # [bs, 2]
        probs = torch.softmax(logits, dim=-1)              # [bs, 2]
        pred_labels = torch.argmax(probs, dim=-1)          # [bs]

        # Ranking score = P(relevant) - P(irrelevant)
        rank_scores = self._compute_score(probs)

        loss = None
        if labels is not None and scores is not None:
            loss = self._pairwise_hinge_loss(rank_scores, scores, qids)

        return {
            "loss": loss,
            "qids": qids,
            "score": rank_scores,
            "pred_label": pred_labels,
        }

    def _pairwise_hinge_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        qids: torch.Tensor,
        margin_weight: float = 0.8,
        gamma: float = 1.0,
        topk: bool = False,
        pairdiff_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pairwise logistic hinge loss, focusing on optimizing positive-negative pair ratio.

        Args:
            logits: Predicted ranking scores [B, 1]
            labels: Ground-truth relevance labels [B, 1]
            qids: Query IDs for grouping samples [B]
            margin_weight: Scaling factor for margin
            gamma: Logistic steepness factor (currently unused in computation)
            topk: If True, emphasize top 30% samples within each query group
            pairdiff_mask: Optional additional margin adjustment matrix [B, B]

        Returns:
            Scalar loss value (averaged over valid pairs)
        """
        # Same-query mask [B, B]
        qid_mask = (qids.unsqueeze(0) == qids.unsqueeze(1)).float()

        if topk:
            qid_mask = qid_mask * self._get_topk_mask(qids, logits.squeeze(-1), labels)

        batch_size = logits.shape[0]
        labels = labels.unsqueeze(1)

        # Broadcast to pairwise matrices
        score_pos = logits.expand(-1, batch_size)      # score of sample i
        score_neg = score_pos.transpose(0, 1)          # score of sample j
        pos = labels.expand(-1, batch_size)
        neg = pos.transpose(0, 1)

        # Margin for each pair
        margin = (
            (pos - neg + pairdiff_mask) * qid_mask * margin_weight
            if pairdiff_mask is not None
            else (pos - neg) * qid_mask * margin_weight
        )
        pair_mask = (margin > 1e-6).float()

        score_diff = score_pos - score_neg
        margin_diff = margin + torch.clamp(-score_diff, min=-10.0)
        loss = torch.relu(margin_diff) * pair_mask

        # Normalize by number of valid pairs
        return torch.sum(loss) / (torch.sum(pair_mask) + 1e-5)

    def _get_topk_mask(
        self,
        qids: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Within each query group, assign weight 2.0 to top 30% samples (by predicted score),
        1.0 to others. Returns pairwise mask [B, B].

        Args:
            qids: Query IDs [B]
            logits: Predicted scores [B]
            labels: Relevance labels [B]

        Returns:
            [B, B] float mask for pairwise importance weighting
        """
        flatten_qids = qids.view(-1)
        flatten_logits = logits.view(-1)
        flatten_labels = labels.view(-1)
        unique_qids = torch.unique(flatten_qids)

        batch_size = qids.shape[0]
        position_mask = torch.ones(batch_size, dtype=torch.float32, device=logits.device)

        for uq in unique_qids:
            mask = (flatten_qids == uq)
            indices = mask.nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue

            cur_labels = flatten_labels[indices]
            valid_count = (cur_labels >= 0).sum().item()
            if valid_count == 0:
                continue

            k = math.ceil(valid_count * 0.3)
            if k == 0:
                continue

            cur_logits = flatten_logits[indices]
            topk_idx = indices[cur_logits.argsort(descending=True)[:k]]
            position_mask[topk_idx] = 2.0

        # Expand to pairwise symmetric mask
        pos_mask_2d = position_mask.unsqueeze(-1).expand(batch_size, batch_size)
        return pos_mask_2d.transpose(0, 1)

    def _compute_score(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute final ranking score: P(relevant) - P(irrelevant)"""
        weights = torch.tensor([-1.0, 1.0], device=probs.device)
        return (probs * weights).sum(dim=-1, keepdim=True)

class QueritWrapper(RerankerWrapper):
    """
    Multi-GPU / multi-process reranker wrapper for mteb.mteb evaluation.
    Supports flattening all query-passage pairs without explicit grouping.
    """

    def __init__(
        self,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options
        self.model = QueritModel.from_pretrained(
            model_name_or_path, **model_args
        )
        logger.info(f"Using model {model_name_or_path}")

        if kwargs.get("torch_compile"):
            self.torch_compile = kwargs["torch_compile"]
            self.model = torch.compile(self.model)
        else:
            self.torch_compile = False

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if "[CLS]" not in self.tokenizer.get_vocab():
            raise ValueError("Tokenizer missing required special token '[CLS]'")
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.pad_token_id = self.tokenizer.pad_token_id or 0

        self.max_length = min(kwargs.get("max_length", 4096), self.tokenizer.model_max_length)  # sometimes it's a v large number/max int
        logger.info(f"Using max_length of {self.max_length}")
        self.model.eval()

    def process_inputs(
        self,
        pairs: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of (query, document) pairs:
        - Concatenate prompt + Query + Content
        - Append [CLS] at the end
        - Left-pad to max_length
        - Generate custom attention mask based on block types
        """
        max_length = self.max_length
        # Construct input texts
        enc = self.tokenizer(
            pairs,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
            padding=False,
        )

        input_ids_list: List[List[int]] = []
        attn_mask_list: List[torch.Tensor] = []

        for ids in enc["input_ids"]:
            # Append [CLS] token
            ids = ids + [self.cls_token_id]
            block_types = [1] * (len(ids) - 1) + [2]  # content + CLS

            # Pad or truncate
            if len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = [self.pad_token_id] * pad_len + ids
                block_types = [0] * pad_len + block_types
            else:
                ids = ids[-max_length:]
                block_types = block_types[-max_length:]

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
        prompt_type: Optional[PromptType] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Predict relevance scores for query-passage pairs.
        Supports both single-process and multi-process/multi-GPU modes.
        """
        # Flatten all pairs from mteb.mteb DataLoaders
        queries = [text for batch in inputs1 for text in batch["text"]]
        passages = [text for batch in inputs2 for text in batch["text"]]
        assert len(queries) == len(passages)

        instructions = None
        if "instruction" in inputs2.dataset.features:
            instructions = [text for batch in inputs1 for text in batch["instruction"]]

        num_pairs = len(queries)
        if num_pairs == 0:
            return []
        final_scores: List[float] = []

        with tqdm(total=num_pairs, desc="Scoring", ncols=100) as pbar:
            for start in range(0, num_pairs, self.batch_size):
                end = min(start + self.batch_size, num_pairs)
                batch_q = queries[start:end]
                batch_d = passages[start:end]

                batch_instructions = (
                    instructions[start : end]
                    if instructions is not None
                    else [None] * len(batch_q)
                )
                pairs = [
                    self.format_instruction(instr, query, doc)
                    for instr, query, doc in zip(
                        batch_instructions, batch_q, batch_d
                    )
                ]
                enc = self.process_inputs(pairs)
                out = self.model(**enc)
                scores = out["score"].squeeze(-1).detach().float().cpu().tolist()

                if not isinstance(scores, list):
                    scores = [scores]

                final_scores.extend(scores)
                pbar.update(len(scores))

        assert len(final_scores) == num_pairs
        return final_scores

    @staticmethod
    def format_instruction(instruction: str | None, query: str, doc: str) -> str:
        if instruction is None:
            output = f"Judge whether the Content meets the requirements based on the Query. Query: {query}; Content: {doc}"
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    @staticmethod
    def compute_mask_content_cls(block_types: List[int]) -> torch.Tensor:
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

model_meta = ModelMeta(
    loader=QueritWrapper,
    loader_kwargs={
        "fp_options": "float16",
    },
    name="Querit/Querit",
    model_type=["cross-encoder"],
    languages=["eng"],
    open_weights=True,
    revision="eaa04c1017572116bccf077d418d79d9ffca062d",
    release_date='2026-01-24',
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    reference="https://huggingface.co/Querit/Querit",
    similarity_fn_name=None,
    training_datasets=set(),
    embed_dim=None,
    license="apache-2.0",
    framework=["PyTorch"],
    use_instructions=None,
    public_training_code=None,
    public_training_data=None,
    citation=None,
)