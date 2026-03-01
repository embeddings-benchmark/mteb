from __future__ import annotations

import logging

import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .colpali_models import ColPaliEngineWrapper
from .colqwen_models import COLNOMIC_LANGUAGES

CITATION = """
@misc{nomicembedmultimodal2025,
  title={Nomic Embed Multimodal: Interleaved Text, Image, and Screenshots for Visual Document Retrieval},
  author={Nomic Team},
  year={2025},
  publisher={Nomic AI},
  url={https://nomic.ai/blog/posts/nomic-embed-multimodal}
}"""

TRAINING_DATA = {
    # https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source
}


logger = logging.getLogger(__name__)


class BiQwen2_5Wrapper(ColPaliEngineWrapper):  # noqa: N801
    """Wrapper for BiQwen2_5 dense (single-vector) embedding model."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-multimodal-7b",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

        super().__init__(
            model_name=model_name,
            model_class=BiQwen2_5,
            processor_class=BiQwen2_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )

    def get_image_embeddings(self, images, batch_size: int = 32, **kwargs):
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                inputs = self.processor.process_images(imgs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.append(outs.cpu().to(torch.float32))

        return torch.cat(all_embeds, dim=0)

    def get_text_embeddings(self, texts, batch_size: int = 32, **kwargs):
        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts"):
                batch = [
                    self.processor.query_prefix
                    + t
                    + self.processor.query_augmentation_token * 10
                    for t in batch["text"]
                ]
                inputs = self.processor.process_queries(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.append(outs.cpu().to(torch.float32))

        return torch.cat(all_embeds, dim=0)

    def similarity(self, a, b):
        import torch.nn.functional as F

        a_norm = F.normalize(a, p=2, dim=-1)
        b_norm = F.normalize(b, p=2, dim=-1)
        return torch.mm(a_norm, b_norm.T)


nomic_embed_multimodal_7b = ModelMeta(
    loader=BiQwen2_5Wrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="nomic-ai/nomic-embed-multimodal-7b",
    model_type=["dense"],
    languages=COLNOMIC_LANGUAGES,
    revision="1291f1b6ca07061b0329df9d5713c09b294be576",
    release_date="2025-04-15",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/nomic-ai/colpali-queries-mined-20250321-by-source",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=CITATION,
)


if __name__ == "__main__":
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, data, features):
            self.data = data
            self.features = features

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def make_dataloader(data, features, batch_size=2):
        ds = SimpleDataset(data, features)
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]},
        )

    model = BiQwen2_5Wrapper()

    images = [
        Image.new("RGB", (128, 128), color="white"),
        Image.new("RGB", (64, 32), color="black"),
    ]
    queries = [
        "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year's financial performance?",
    ]

    image_loader = make_dataloader([{"image": img} for img in images], {"image"})
    query_loader = make_dataloader([{"text": q} for q in queries], {"text"})

    image_embeddings = model.get_image_embeddings(image_loader)
    query_embeddings = model.get_text_embeddings(query_loader)

    scores = model.similarity(query_embeddings, image_embeddings)
    print("Scores (queries x images):")
    print(scores)
