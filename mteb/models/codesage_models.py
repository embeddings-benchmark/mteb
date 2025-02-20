from __future__ import annotations

from functools import partial
from typing import Any, List

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta, sentence_transformers_loader

codesage_languages = [
    "python-Code",
    "javascript-Code",
    "go-Code",
    "ruby-Code",
    "java-Code",
    "php-Code",
]

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

codesage_large = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="codesage/codesage-large-v2",
        revision="6e5d6dc15db3e310c37c6dbac072409f95ffa5c5",
    ),
    name="codesage/codesage-large-v2",
    languages=codesage_languages,
    revision="6e5d6dc15db3e310c37c6dbac072409f95ffa5c5",
    release_date="2024-02-03",
    modalities=["text"],
    n_parameters=1_300_000_000,
    memory_usage_mb=4959,
    max_tokens=2048,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-large-v2",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval": ["train"],
        "CodeSearchNetCCRetrieval": ["train"],
    },
)

codesage_base = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="codesage/codesage-base-v2",
        revision="92eac4f44c8674638f039f1b0d8280f2539cb4c7",
    ),
    name="codesage/codesage-base-v2",
    languages=codesage_languages,
    revision="92eac4f44c8674638f039f1b0d8280f2539cb4c7",
    release_date="2024-02-03",
    modalities=["text"],
    n_parameters=356_000_000,
    memory_usage_mb=1358,
    max_tokens=2048,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-base-v2",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval": ["train"],
        "CodeSearchNetCCRetrieval": ["train"],
    },
)

codesage_small = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="codesage/codesage-small-v2",
        revision="4844c2f24b25e181aa43ca058cc73dd2622565c1",
    ),
    name="codesage/codesage-small-v2",
    languages=codesage_languages,
    revision="4844c2f24b25e181aa43ca058cc73dd2622565c1",
    release_date="2024-02-03",
    modalities=["text"],
    n_parameters=130_000_000,
    memory_usage_mb=496,
    max_tokens=2048,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/codesage/codesage-small-v2",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "CodeSearchNetRetrieval": ["train"],
        "CodeSearchNetCCRetrieval": ["train"],
    },
)
