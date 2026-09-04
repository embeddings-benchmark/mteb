import pytest
from transformers import AutoTokenizer

import mteb


@pytest.mark.parametrize(
    ("model_name", "text", "expected_token_count"),
    [
        (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "This is a multilingual tokenizer test.",
            13,
        ),
        (
            "sdadas/mmlw-roberta-base",
            "To jest test wielojezycznego tokenizera.",
            14,
        ),
        (
            "sdadas/mmlw-roberta-large",
            "To jest test wielojezycznego tokenizera.",
            13,
        ),
    ],
)
def test_pinned_tokenizer_preserves_expected_segmentation(
    model_name: str, text: str, expected_token_count: int
):
    meta = mteb.get_model_meta(model_name)
    tokenizer = AutoTokenizer.from_pretrained(meta.name, revision=meta.revision)

    token_ids = tokenizer(text)["input_ids"]

    assert len(token_ids) == expected_token_count
    assert tokenizer.unk_token_id not in token_ids
