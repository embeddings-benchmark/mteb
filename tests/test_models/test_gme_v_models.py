from types import SimpleNamespace

import torch

from mteb.models.model_implementations.gme_v_models import Encoder, GmeQwen2VL


class _LanguageModel(torch.nn.Module):
    def forward(self, *, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _ImageOnlyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedded_batches: list[tuple[list[None], list[object]]] = []

    def embed(self, *, texts, images, **kwargs):
        self.embedded_batches.append((texts, images))
        return torch.ones(len(images), 2)


def test_encoder_supports_qwen2vl_language_model_layout():
    embedding = torch.nn.Embedding(10, 3)
    language_model = _LanguageModel()
    base = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model),
        get_input_embeddings=lambda: embedding,
    )
    encoder = Encoder(base, processor=SimpleNamespace(tokenizer=SimpleNamespace()))

    embeddings = encoder(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
    )

    expected = torch.nn.functional.normalize(embedding(torch.tensor([[1, 2]]))[:, -1])
    torch.testing.assert_close(embeddings, expected)


def test_gme_encodes_image_only_batches():
    model = object.__new__(GmeQwen2VL)
    model.model = _ImageOnlyEncoder()
    model.device = "cpu"
    model.get_instruction = lambda *args: None
    images = [object(), object()]

    embeddings = model.encode(
        [{"image": images}],
        task_metadata=SimpleNamespace(),
        hf_split="test",
        hf_subset="default",
        show_progress_bar=False,
    )

    assert model.model.embedded_batches == [([None, None], images)]
    torch.testing.assert_close(embeddings, torch.ones(2, 2))
