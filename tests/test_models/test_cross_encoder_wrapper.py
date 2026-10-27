from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder

from mteb._create_dataloaders import create_dataloader
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper


def test_cross_encoder_predicts_one_batch_at_a_time():
    """Pairs are scored per dataloader batch (not all at once) with unchanged scores."""
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    queries = ["what is a cat", "what is a dog", "capital of france", "red", "sky"]
    documents = ["a small feline", "a loyal pet", "paris", "a colour", "blue"]
    expected = cross_encoder.predict(list(zip(queries, documents, strict=True)))

    predict = cross_encoder.predict
    batch_sizes = []

    def recording_predict(pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
        batch_sizes.append(len(pairs))
        return predict(pairs, **kwargs)

    cross_encoder.predict = recording_predict
    model = CrossEncoderWrapper(cross_encoder)
    task_metadata = MockRetrievalTask.metadata
    scores = model.predict(
        create_dataloader(
            Dataset.from_dict({"text": queries}),
            task_metadata=task_metadata,
            batch_size=2,
        ),
        create_dataloader(
            Dataset.from_dict({"text": documents}),
            task_metadata=task_metadata,
            batch_size=2,
        ),
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
    )

    assert batch_sizes == [2, 2, 1]
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_low_precision_scoring_head_runs_in_fp32():
    """bf16 logits collapse nearby scores into ties, so the head is computed in fp32."""
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
    # cast after loading: model_kwargs needs sentence-transformers>=4
    cross_encoder.model.to(torch.bfloat16)
    pairs = [("what is a cat", "a small feline"), ("what is a cat", "a loyal pet")]
    CrossEncoderWrapper(cross_encoder)

    head = cross_encoder.model.classifier
    assert head.weight.dtype == torch.float32
    assert head(torch.randn(4, head.in_features, dtype=torch.bfloat16)).dtype == (
        torch.float32
    )
    assert cross_encoder.predict(pairs).dtype == np.float32


@pytest.mark.parametrize("use_residual", [False, True])
def test_dense_scoring_head_runs_in_fp32(use_residual: bool):
    """Modular cross-encoders score with a Dense; its residual projection also feeds the scores."""
    dense_module = pytest.importorskip("sentence_transformers.base.modules.dense")
    from sentence_transformers.base.modules.transformer import Transformer
    from sentence_transformers.sentence_transformer.modules.pooling import Pooling

    transformer = Transformer("hf-internal-testing/tiny-random-BertModel")
    dim = transformer.get_word_embedding_dimension()
    dense = dense_module.Dense(
        dim,
        1,
        activation_function=torch.nn.Identity(),
        module_input_name="sentence_embedding",
        module_output_name="scores",
        use_residual=use_residual,
    )
    cross_encoder = CrossEncoder(
        modules=[transformer, Pooling(dim), dense], device="cpu"
    ).to(torch.bfloat16)
    CrossEncoderWrapper(cross_encoder)

    heads = [cross_encoder[2].linear] + (
        [cross_encoder[2].residual] if use_residual else []
    )
    assert all(head.weight.dtype == torch.float32 for head in heads)


def test_logit_score_logits_run_in_fp32():
    """Only the LogitScore token logits are recomputed in fp32, not the whole lm_head."""
    logit_score = pytest.importorskip(
        "sentence_transformers.cross_encoder.modules.logit_score"
    )
    from sentence_transformers.base.modules.transformer import Transformer

    transformer = Transformer(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        transformer_task="text-generation",
    )
    token_ids = [5, 7]
    cross_encoder = CrossEncoder(
        modules=[transformer, logit_score.LogitScore(*token_ids)], device="cpu"
    ).to(torch.bfloat16)
    CrossEncoderWrapper(cross_encoder)

    lm_head = cross_encoder.model.get_output_embeddings()
    assert lm_head.weight.dtype == torch.bfloat16
    hidden = torch.randn(2, 1, lm_head.in_features, dtype=torch.bfloat16)
    logits = lm_head(hidden)
    assert logits.dtype == torch.float32
    expected = hidden.float() @ lm_head.weight[token_ids].float().T
    torch.testing.assert_close(logits[..., token_ids], expected)
