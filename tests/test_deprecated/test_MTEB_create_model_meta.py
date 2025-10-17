from __future__ import annotations

from pathlib import Path

import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb import MTEB, AbsTask
from tests.mock_tasks import MockRetrievalTask


def test_create_model_meta_from_sentence_transformers():
    model_name = "sentence-transformers/average_word_embeddings_levy_dependency"
    revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"
    model = SentenceTransformer(model_name, revision=revision)

    meta = MTEB.create_model_meta(model)

    assert meta.embed_dim == model.get_sentence_embedding_dimension()
    assert type(meta.framework) is list
    assert meta.framework[0] == "Sentence Transformers"
    assert meta.name == model_name
    assert meta.revision == revision


def test_create_model_meta_from_cross_encoder():
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    revision = "841d331b6f34b15d6ac0ab366ae3a3b36eeac691"
    model = CrossEncoder(model_name, revision=revision)

    meta = MTEB.create_model_meta(model)

    assert meta.name == model_name
    assert meta.revision == revision


@pytest.mark.parametrize("task", [MockRetrievalTask()])
def test_output_folder_model_meta(task: AbsTask, tmp_path: Path):
    mteb = MTEB(tasks=[task])
    model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    model = CrossEncoder(model_name)
    meta = mteb.create_model_meta(model)
    output_path = mteb._create_output_folder(
        model_meta=meta, output_folder=tmp_path.as_posix()
    )

    output_path = Path(output_path)
    assert output_path.exists()
    assert output_path.is_dir()
    assert output_path.name == model.config._commit_hash
    assert output_path.parent.name == "cross-encoder__ms-marco-TinyBERT-L-2-v2"
    assert output_path.parent.parent == tmp_path
