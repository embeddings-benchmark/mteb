import pytest

import mteb
from mteb.models.model_meta import ModelMeta


class TestCrossEncoderDetection:
    """Test cases for cross-encoder model detection."""

    @pytest.mark.parametrize(
        "model_name,expected_type",
        [
            ("cross-encoder/ms-marco-TinyBERT-L-2-v2", "cross-encoder"),
            ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder"),
            ("sentence-transformers/all-MiniLM-L6-v2", "dense"),
            ("BAAI/bge-small-en-v1.5", "dense"),
        ],
    )
    def test_detect_model_type(self, model_name: str, expected_type: str):
        """Test that models are correctly detected as cross-encoder or dense."""
        loader, model_type = ModelMeta._detect_model_type_and_loader(
            model_name, revision=None
        )

        assert model_type == expected_type, (
            f"Expected '{expected_type}', got '{model_type}' for {model_name}"
        )

    def test_detect_none_model_name(self):
        """Test that None model name returns default values."""
        loader, model_type = ModelMeta._detect_model_type_and_loader(
            None, revision=None
        )

        assert model_type == "dense"
        assert loader.__name__ == "sentence_transformers_loader"


@pytest.mark.integration
class TestGetModelIntegration:
    """Integration tests for loading models with automatic detection via mteb.get_model()."""

    def test_load_cross_encoder_via_get_model(self):
        """Test loading cross-encoder via mteb.get_model() with automatic detection."""
        model = mteb.get_model("cross-encoder/ms-marco-TinyBERT-L-2-v2")

        assert model.mteb_model_meta.model_type == ["cross-encoder"]
        assert model.mteb_model_meta.is_cross_encoder
        assert model.mteb_model_meta.loader.__name__ == "CrossEncoderWrapper"

    def test_load_sentence_transformer_via_get_model(self):
        """Test loading sentence transformer via mteb.get_model()."""
        model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")

        assert model.mteb_model_meta.model_type == ["dense"]
        assert not model.mteb_model_meta.is_cross_encoder
        assert model.mteb_model_meta.loader.__name__ == "sentence_transformers_loader"
