from __future__ import annotations

from functools import partial
from typing import Any, List

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


class CodeSageModelWrapper:
    """
    A wrapper for the codesage/codesage-large-v2 model.
    
    This model is designed to generate code embeddings. 
    Note: CodeSage requires adding an EOS token at the end of each tokenized sequence.
    """

    def __init__(
        self,
        checkpoint: str = "codesage/codesage-large-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.checkpoint = checkpoint
        self.device = device
        # Note: 'add_eos_token=True' ensures that the EOS token is appended to every sequence.
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, add_eos_token=True
        )
        self.model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def preprocess(self, texts: List[str], **kwargs: Any):
        """
        Tokenizes a list of code strings.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_code_embeddings(
        self,
        texts: List[str],
        batch_size: int = 8,
        **kwargs: Any,
    ):
        """
        Computes code embeddings for a list of code strings.
        This method tokenizes the input, passes it through the model,
        and performs mean pooling over token embeddings.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Code Embeddings"):
                batch_texts = texts[i : i + batch_size]
                inputs = self.preprocess(batch_texts, **kwargs)
                # Model returns a tuple; we take the first element (last hidden state)
                outputs = self.model(**inputs)[0]  # shape: (batch_size, seq_len, hidden_size)
                # Mean pooling over the token dimension
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(-1)
                    sum_embeddings = (outputs * attention_mask).sum(dim=1)
                    lengths = attention_mask.sum(dim=1)
                    embeddings = sum_embeddings / lengths.clamp(min=1e-9)
                else:
                    embeddings = outputs.mean(dim=1)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ):
        """
        Computes cosine similarity between two sets of embeddings.
        """
        emb1_norm = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2_norm = emb2 / emb2.norm(dim=-1, keepdim=True)
        return torch.matmul(emb1_norm, emb2_norm.T)


codesage_model_meta = ModelMeta(
    loader=partial(CodeSageModelWrapper, checkpoint="codesage/codesage-large-v2"),
    name="codesage/codesage-large-v2",
    languages=["code"],
    revision=None,
    release_date="2024-02-03",
    modalities=["code"],
    n_parameters=1300000000,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=2048,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-large-v2",
    similarity_fn_name="compute_similarity",
    use_instructions=False,
    training_datasets=None,
)
