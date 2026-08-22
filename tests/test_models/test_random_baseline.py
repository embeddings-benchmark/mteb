import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from mteb.models.model_implementations.random_baseline import (
    _batch_to_embeddings,
    _image_to_vector,
    _string_to_vector,
)


def test_random_baseline_ignores_missing_image_in_fusion() -> None:
    image = Image.new("RGB", (8, 8), "red")
    dataloader = DataLoader(
        [
            {"text": "text only", "image": None},
            {"text": "with image", "image": image},
        ],
        batch_size=2,
        collate_fn=lambda rows: {
            "text": [row["text"] for row in rows],
            "image": [row["image"] for row in rows],
        },
    )

    embeddings = _batch_to_embeddings(dataloader, embedding_dim=8)

    np.testing.assert_allclose(embeddings[0], _string_to_vector("text only", 8))
    np.testing.assert_allclose(
        embeddings[1],
        (_string_to_vector("with image", 8) + _image_to_vector(image, 8)) / 2,
    )
