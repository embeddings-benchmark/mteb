from __future__ import annotations

from pathlib import Path

import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask
from tests.test_benchmark.mock_tasks import MockRetrievalTask


def test_create_model_meta_from_sentence_transformers():
    model_name = "sentence-transformers/average_word_embeddings_levy_dependency"
    model = SentenceTransformer(model_name)

    meta = MTEB.create_model_meta(model)

    assert meta.similarity_fn_name == "cosine"
    assert meta.embed_dim == model.get_sentence_embedding_dimension()
    assert type(meta.framework) is list
    assert meta.framework[0] == "Sentence Transformers"
    assert meta.name == model_name
    assert meta.revision == model.model_card_data.base_model_revision


def test_create_model_meta_from_cross_encoder():
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    model = CrossEncoder(model_name)

    meta = MTEB.create_model_meta(model)
    # model.name_or_path
    # _commit_hash
    assert meta.name == model_name
    assert meta.revision == model.config._commit_hash

    return meta


@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_output_folder_model_meta(task: AbsTask, tmp_path: Path):
    mteb = MTEB(tasks=[task])
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    model = CrossEncoder(model_name)
    meta = mteb.create_model_meta(model)
    output_path = mteb.create_output_folder(
        model_meta=meta, output_folder=tmp_path.as_posix()
    )

    output_path = Path(output_path)
    assert output_path.exists()
    assert output_path.is_dir()
    assert output_path.name == model.config._commit_hash
    assert output_path.parent.name == "cross-encoder__ms-marco-TinyBERT-L-2-v2"
    assert output_path.parent.parent == tmp_path
