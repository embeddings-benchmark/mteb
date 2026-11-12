from __future__ import annotations

import datetime as dt
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from mteb.models.model_implementations.microsoft_models import (
    MicrosoftEmbedding1Wrapper,
    microsoft_embedding_1_large,
)
from mteb.types import PromptType


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> MicrosoftEmbedding1Wrapper:
    monkeypatch.setenv("AZURE_DATA_AI_STAGING_ACCOUNT", "testaccount")
    monkeypatch.setenv("AZURE_DATA_AI_STAGING_CONTAINER", "vidore3")
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas", Mock(return_value="test-sas")
    )
    encoder = MicrosoftEmbedding1Wrapper.__new__(MicrosoftEmbedding1Wrapper)
    encoder.model_name = "microsoft-embedding-1-large"
    encoder.revision = "1"
    encoder.embed_dim = 256
    encoder.mteb_model_meta = microsoft_embedding_1_large
    encoder._blob_service = Mock()
    encoder._blob_service.get_blob_client.return_value.url = (
        "https://testaccount.blob.core.windows.net/vidore3/test.png"
    )
    encoder._delegation_key = None
    encoder._delegation_key_expiry = None
    encoder._client = Mock()
    encoder._client.get_embeddings.return_value = SimpleNamespace(
        model_version="1",
        data=[SimpleNamespace(multi_embedding=[[0.25] * 256] * 2, shape=[2, 256])],
    )
    return encoder


def test_staged_image_preserves_size_and_converts_to_rgb(
    model: MicrosoftEmbedding1Wrapper,
) -> None:
    image = Image.new("RGBA", (10, 20), "red")
    model._embed_image(image)
    blob = model._blob_service.get_blob_client.return_value
    location = model._blob_service.get_blob_client.call_args.kwargs
    assert location["container"] == "vidore3"
    assert location["blob"].startswith("tmp/mteb-microsoft-embedding-1-large/")
    uploaded = blob.upload_blob.call_args.args[0]
    with Image.open(BytesIO(uploaded)) as decoded:
        assert decoded.size == image.size
        assert decoded.mode == "RGB"
        assert decoded.getpixel((0, 0)) == (255, 0, 0)
    blob.delete_blob.assert_called_once()
    assert model._client.get_embeddings.call_args.kwargs["inputs"] == [
        {"kind": "image", "url": blob.url, "sasToken": "test-sas"}
    ]


@pytest.mark.parametrize(
    ("prompt_type", "kind"),
    [(PromptType.query, "query"), (PromptType.document, "text"), (None, "text")],
)
def test_text_inputs_are_sent_individually(
    model: MicrosoftEmbedding1Wrapper, prompt_type: PromptType | None, kind: str
) -> None:
    embeddings = model.encode(
        [{"text": ["first", "second"]}, {"text": ["third"]}],
        task_metadata=Mock(),
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
        show_progress_bar=False,
    )
    assert embeddings.shape == (3, 2, 256)
    assert embeddings.dtype == np.float32
    calls = model._client.get_embeddings.call_args_list
    assert [call.kwargs["inputs"] for call in calls] == [
        [{"kind": kind, "content": text}] for text in ["first", "second", "third"]
    ]
    for call in calls:
        assert call.kwargs["model"] == "microsoft-embedding-1-large"
        assert call.kwargs["model_version"] == "1"
        assert call.kwargs["delivery"] == "inline"
        assert call.kwargs["vector_kind"] == "multi"
        assert call.kwargs["dimensions"] == 256


def test_images_use_sas_and_embeddings_preserve_order(
    model: MicrosoftEmbedding1Wrapper,
) -> None:
    images = [Image.new("RGB", (10, 20), color) for color in ["red", "blue"]]
    model._client.get_embeddings.side_effect = [
        SimpleNamespace(
            model_version="1",
            data=[
                SimpleNamespace(
                    multi_embedding=[[value] * 256] * count, shape=[count, 256]
                )
            ],
        )
        for value, count in [(1.0, 2), (2.0, 3)]
    ]
    embeddings = model.encode(
        [{"image": images}],
        task_metadata=Mock(),
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.document,
        show_progress_bar=False,
    )
    assert embeddings.shape == (2, 3, 256)
    np.testing.assert_array_equal(embeddings[:, 0, 0], [1.0, 2.0])
    np.testing.assert_array_equal(embeddings[0, 2], np.zeros(256))
    np.testing.assert_array_equal(embeddings[1, 2], np.full(256, 2.0))
    assert embeddings.dtype == np.float32
    for call in model._client.get_embeddings.call_args_list:
        assert call.kwargs["inputs"][0]["sasToken"] == "test-sas"
        assert "content" not in call.kwargs["inputs"][0]
        assert call.kwargs["delivery"] == "inline"
        assert call.kwargs["vector_kind"] == "multi"
        assert call.kwargs["dimensions"] == 256
    assert model._blob_service.get_user_delegation_key.call_count == 1
    assert model._blob_service.get_blob_client.return_value.delete_blob.call_count == 2


def test_sas_is_short_lived_read_only_and_https(model: MicrosoftEmbedding1Wrapper) -> None:
    from azure.storage.blob import generate_blob_sas

    model._embed_image(Image.new("RGB", (10, 20)))
    options = generate_blob_sas.call_args.kwargs
    assert options["account_name"] == "testaccount"
    assert options["container_name"] == "vidore3"
    blob_name = model._blob_service.get_blob_client.call_args.kwargs["blob"]
    assert options["blob_name"] == blob_name
    assert options["protocol"] == "https"
    assert str(options["permission"]) == "r"
    assert options["expiry"] - options["start"] == dt.timedelta(minutes=20)
    assert options["user_delegation_key"] is model._delegation_key


def test_image_is_deleted_if_embedding_fails(model: MicrosoftEmbedding1Wrapper) -> None:
    model._client.get_embeddings.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        model._embed_image(Image.new("RGB", (10, 20)))
    model._blob_service.get_blob_client.return_value.delete_blob.assert_called_once()


def test_images_require_staging_configuration(
    model: MicrosoftEmbedding1Wrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZURE_DATA_AI_STAGING_ACCOUNT")
    with pytest.raises(ValueError, match="AZURE_DATA_AI_STAGING_ACCOUNT"):
        model._embed_image(Image.new("RGB", (10, 20)))
    model._blob_service.get_blob_client.assert_not_called()


@pytest.mark.parametrize(
    ("version", "data", "message"),
    [
        (
            "7",
            [SimpleNamespace(multi_embedding=[[0.0] * 256], shape=[1, 256])],
            "Expected Microsoft embedding version",
        ),
        ("1", [], "Expected one inline embedding"),
        (
            "1",
            [SimpleNamespace(multi_embedding=[[0.0]], shape=[1, 1])],
            "Unexpected embedding shape",
        ),
        (
            "1",
            [SimpleNamespace(multi_embedding=[[0.0] * 256], shape=[2, 256])],
            "Unexpected embedding shape",
        ),
        (
            "1",
            [SimpleNamespace(multi_embedding=[], shape=[0, 256])],
            "Unexpected embedding shape",
        ),
        (
            "1",
            [SimpleNamespace(multi_embedding=[[np.nan] * 256], shape=[1, 256])],
            "Unexpected embedding shape",
        ),
    ],
)
def test_invalid_responses_fail(
    model: MicrosoftEmbedding1Wrapper, version: str, data: list, message: str
) -> None:
    model._client.get_embeddings.return_value = SimpleNamespace(
        model_version=version, data=data
    )
    with pytest.raises(ValueError, match=message):
        model.encode(
            [{"text": ["query"]}],
            task_metadata=Mock(),
            hf_split="test",
            hf_subset="default",
            show_progress_bar=False,
        )


def test_metadata_is_pinned_to_version_one() -> None:
    from mteb import get_model_meta

    assert (
        get_model_meta("Microsoft/microsoft-embedding-1-large")
        is microsoft_embedding_1_large
    )
    assert microsoft_embedding_1_large.revision == "1"
    assert microsoft_embedding_1_large.embed_dim == [256, 1024]
    assert microsoft_embedding_1_large.loader_kwargs["embed_dim"] == 256
    assert microsoft_embedding_1_large.modalities == ["image", "text"]
    assert microsoft_embedding_1_large.model_type == ["late-interaction"]
    assert microsoft_embedding_1_large.similarity_fn_name.value == "MaxSim"
    assert microsoft_embedding_1_large.extra_requirements_groups == ["microsoft"]


@pytest.mark.parametrize("embed_dim", [256, 1024])
def test_model_loading_selects_token_width(
    monkeypatch: pytest.MonkeyPatch, embed_dim: int
) -> None:
    client = Mock()
    client.get_embeddings.return_value = SimpleNamespace(
        model_version="1",
        data=[
            SimpleNamespace(multi_embedding=[[0.25] * embed_dim], shape=[1, embed_dim])
        ],
    )
    monkeypatch.setattr("azure.data.ai.InferenceClient", Mock(return_value=client))
    encoder = microsoft_embedding_1_large.load_model(
        embed_dim=embed_dim,
        endpoint="https://inference.example.com",
        credential=Mock(),
    )
    try:
        result = encoder._embed({"kind": "query", "content": "query"})
        assert result.shape == (1, embed_dim)
        assert (
            client.get_embeddings.call_args.kwargs["model"]
            == "microsoft-embedding-1-large"
        )
        assert client.get_embeddings.call_args.kwargs["model_version"] == "1"
        assert client.get_embeddings.call_args.kwargs["dimensions"] == embed_dim
        assert encoder.mteb_model_meta.embed_dim == embed_dim
    finally:
        encoder.close()


def test_similarity_uses_token_maxsim(model: MicrosoftEmbedding1Wrapper) -> None:
    query = np.zeros((1, 2, 256), dtype=np.float32)
    query[0, 0, 0] = 1.0
    query[0, 1, 1] = 1.0
    documents = np.zeros((2, 3, 256), dtype=np.float32)
    documents[0, 0, 0] = 1.0
    documents[0, 1, 1] = 1.0
    documents[1, 0, 0] = 0.5

    np.testing.assert_allclose(model.similarity(query, documents), [[2.0, 0.5]])
    np.testing.assert_allclose(
        model.similarity_pairwise(np.repeat(query, 2, axis=0), documents), [2.0, 0.5]
    )
