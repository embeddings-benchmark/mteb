from __future__ import annotations

from functools import partial

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from mteb.model_meta import ModelMeta

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from typing import Any, Dict, Optional

import torch.nn as nn
import torch


class NomicBertTransformer(Transformer):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        config_args: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        revision: str = None
    ) -> None:
        nn.Module.__init__(self)
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        config = AutoConfig.from_pretrained(
            model_name_or_path, **config_args, cache_dir=cache_dir)
        self.auto_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=config, revision=revision, cache_dir=cache_dir, **model_args
        )
        self.auto_model.cls = nn.Identity()
        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(
                    self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length
        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns token_embeddings, cls_token"""
        trans_features = {
            "input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.auto_model(**trans_features)
        output_tokens = output_states.logits

        features.update({"token_embeddings": output_tokens,
                        "attention_mask": features["attention_mask"]})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features


class SentenceTransformerWithNormalization(SentenceTransformer):
    def encode(self, sentences, *args, **kwargs):
        if "normalize_embeddings" not in kwargs:
            kwargs["normalize_embeddings"] = True

        return super().encode(sentences, *args, **kwargs)


def nomic_bert_loader(
    model_name: str, revision: str | None, **kwargs
) -> SentenceTransformer:
    nomic_bert_transformer = NomicBertTransformer(
        model_name_or_path=model_name,
        tokenizer_name_or_path='bert-base-uncased',
        config_args={'trust_remote_code': True},
        model_args={'trust_remote_code': True},
        revision=revision
    )

    pooling_model = Pooling(
        nomic_bert_transformer.get_word_embedding_dimension())

    return SentenceTransformerWithNormalization(modules=[nomic_bert_transformer, pooling_model])


def custom_nomic_bert_loader(
    model_name: str,
    tokenizer_name: str,
    revision: str | None, **kwargs
) -> SentenceTransformer:
    nomic_bert_transformer = NomicBertTransformer(
        model_name_or_path=model_name,
        tokenizer_name_or_path=tokenizer_name,
        config_args={'trust_remote_code': True},
        model_args={'trust_remote_code': True, 'use_auth_token': True},
        tokenizer_args={'use_auth_token': True},
        revision=revision
    )

    pooling_model = Pooling(
        nomic_bert_transformer.get_word_embedding_dimension())

    return SentenceTransformerWithNormalization(modules=[nomic_bert_transformer, pooling_model])


nomic_bert = ModelMeta(
    loader=partial(  # type: ignore
        nomic_bert_loader,
        model_name="nomic-ai/nomic-bert-2048",
        revision=None,
    ),
    name="nomic-ai/nomic-bert-2048",
    languages=["eng-Latn"],
    open_weights=True,
    revision="40b98394640e630d5276807046089b233113aa87",
    release_date="2024-01-03",  # first commit
    open_weights=True,
    license="apache-2.0",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/nomic-ai/nomic-bert-2048",
    public_training_data=True,
    public_training_code=True,
    max_tokens=2048,
)

