from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.model_meta import ScoringFunction
from tests.test_benchmark.mock_tasks import MockRetrievalTask


def test_create_model_meta_from_sentence_transformers():
    model_name = "sentence-transformers/average_word_embeddings_levy_dependency"
    revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"
    model = SentenceTransformer(model_name, revision=revision)

    meta = MTEB.create_model_meta(model)

    assert meta.similarity_fn_name == ScoringFunction.COSINE
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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10 or higher")
def test_model_meta_colbert():
    model_name = "colbert-ir/colbertv2.0"
    colbert_model = pytest.importorskip("pylate.models", reason="pylate not installed")
    revision = "c1e84128e85ef755c096a95bdb06b47793b13acf"
    model = colbert_model.ColBERT(model_name, revision=revision)

    meta = MTEB.create_model_meta(model)

    # assert meta.similarity_fn_name == "MaxSim" test with new release of pylate
    assert type(meta.framework) is list
    assert meta.framework[0] == "Sentence Transformers"
    assert meta.name == model_name
    assert meta.revision == revision
