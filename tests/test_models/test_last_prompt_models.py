"""Offline transport/recipe tests; these do not certify live model inference."""

import json
import unittest
from pathlib import Path

import httpx
import numpy as np
import pytest

import mteb
from mteb.models.model_implementations import last_prompt_models
from mteb.models.model_implementations.last_prompt_models import RECIPES, make_meta


class AdapterTests(unittest.TestCase):
    def encoder(self, recipe, *, wrong_tokens=False, vector=None):
        seen = []
        checks = json.loads(
            (
                Path(last_prompt_models.__file__).parent
                / "last_prompt_resources"
                / "prompt_checks.json"
            ).read_text()
        )

        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": RECIPES[recipe]["checkpoint"],
                                "root": RECIPES[recipe]["checkpoint"],
                            }
                        ]
                    },
                )
            payload = json.loads(request.content)
            if request.url.path == "/tokenize":
                return httpx.Response(
                    200,
                    json={"tokens": [] if wrong_tokens else checks[recipe]["tokens"]},
                )
            assert request.url.path == "/v1/chat/completions"
            seen.append(payload)
            values = vector if vector is not None else [1.0] * 2560
            return httpx.Response(
                200, json={"kv_transfer_params": {"last_hidden_state": values}}
            )

        meta = mteb.get_model_meta(RECIPES[recipe]["name"], RECIPES[recipe]["revision"])
        assert meta.loader is last_prompt_models.LastPromptEncoder
        return mteb.get_model(
            meta.name, meta.revision, transport=httpx.MockTransport(handler)
        ), seen

    def test_exact_payloads_match_frozen_clients(self):
        # Independent expected payload includes whitespace, kwargs and one-token contract.
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                model, seen = self.encoder(recipe)
                text = "  original whitespace\n{literal braces}  "
                try:
                    result = model.encode(
                        [{"text": [text]}],
                        task_metadata=None,
                        hf_split="test",
                        hf_subset="default",
                    )
                    assert result.shape == (1, 2560)
                    assert result.dtype == np.float32
                    assert np.all(result == 1)  # No L2 normalization.
                    assert seen == [
                        {
                            "model": RECIPES[recipe]["checkpoint"],
                            "messages": [
                                {
                                    "role": "user",
                                    "content": text
                                    + "\n Summary above sentence in one word: "
                                    if recipe == "qwen_e5v"
                                    else text,
                                }
                            ],
                            "max_completion_tokens": 1,
                            "temperature": 0,
                            "n": 1,
                            "stream": False,
                            "add_generation_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": False}
                            if recipe == "qwen_e5v"
                            else {"mode": "embeddings", "enable_thinking": False},
                            "kv_transfer_params": {"return_last_hidden_state": True},
                        }
                    ]
                finally:
                    model.close()

    def test_prompt_drift_rejected_before_embedding(self):
        for recipe in RECIPES:
            with (
                self.subTest(recipe=recipe),
                pytest.raises(ValueError, match="prompt tokens"),
            ):
                self.encoder(recipe, wrong_tokens=True)

    def test_retrieval_metadata_preserves_prepared_text(self):
        batches = [
            {
                "text": ["  prepared corpus text  "],
                "title": ["title"],
                "body": ["body"],
            },
            {
                "text": ["prepared query with instruction"],
                "query": ["query"],
                "instruction": ["instruction"],
                "conversation": [[{"role": "user", "content": "query"}]],
            },
        ]
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                model, seen = self.encoder(recipe)
                try:
                    result = model.encode(
                        batches,
                        task_metadata=None,
                        hf_split="test",
                        hf_subset="default",
                    )
                    assert result.shape == (2, 2560)
                    suffix = (
                        "\n Summary above sentence in one word: "
                        if recipe == "qwen_e5v"
                        else ""
                    )
                    assert [p["messages"][0]["content"] for p in seen] == [
                        b["text"][0] + suffix for b in batches
                    ]
                finally:
                    model.close()

    def test_nontext_modalities_rejected_before_requests(self):
        model, seen = self.encoder("nuextract_revised")
        try:
            for key in ["image", "audio", "video"]:
                with (
                    self.subTest(key=key),
                    pytest.raises(ValueError, match="text-only"),
                ):
                    model.encode(
                        [{"text": ["caption"], key: [object()]}],
                        task_metadata=None,
                        hf_split="test",
                        hf_subset="default",
                    )
            assert seen == []
        finally:
            model.close()

    def test_invalid_vector_rejected(self):
        for vector in [[1.0] * 2, [0.0] * 2560, [[1.0] * 2560]]:
            with self.subTest(shape=np.shape(vector)):
                model, _ = self.encoder("qwen_e5v", vector=vector)
                try:
                    with pytest.raises(ValueError, match="2560-dimensional"):
                        model._one("test")
                finally:
                    model.close()

    def test_empty_input(self):
        model, seen = self.encoder("nuextract_revised")
        try:
            result = model.encode(
                [], task_metadata=None, hf_split="test", hf_subset="default"
            )
            assert result.shape == (0, 2560)
            assert seen == []
        finally:
            model.close()

    def test_revision_and_override_rejected(self):
        meta = make_meta("qwen_e5v")
        with pytest.raises(ValueError, match="revision"):
            meta.model_copy(update={"revision": "main"}).load_model()
        with pytest.raises(ValueError, match="Unsupported"):
            meta.load_model(normalize=True)


if __name__ == "__main__":
    unittest.main()
