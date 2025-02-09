from __future__ import annotations

from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import Encoder


def test_sentence_is_encoder():
    model = SentenceTransformer("average_word_embeddings_komninos")
    assert isinstance(model, Encoder)
