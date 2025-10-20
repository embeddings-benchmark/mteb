import logging
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class GraniteVisionEmbeddingWrapper:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        from transformers import AutoModel, AutoProcessor
        from transformers.utils.import_utils import is_flash_attn_2_available

        requires_image_dependencies()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if attn_implementation is None and is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"

        # Load model
        self.mdl = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            device_map=self.device,
            trust_remote_code=True,
            **kwargs,
        )

        self.mdl.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, revision=revision
        )

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        **kwargs,
    ):
        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                inputs = self.processor.process_images(batch["image"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs,
    ):
        all_embeds = []

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                inputs = self.processor.process_queries(batch["text"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(
                inputs, convert_to_numpy=False, convert_to_tensor=True, **kwargs
            )
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(
                inputs, convert_to_numpy=False, convert_to_tensor=True, **kwargs
            )

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError

    def calculate_probs(self, text_embeddings, image_embeddings):
        scores = self.similarity(text_embeddings, image_embeddings)
        return (scores * 100).softmax(dim=-1)

    def similarity(self, a, b):
        return self.processor.score_multi_vector(a, b)


granite_vision_embedding = ModelMeta(
    loader=GraniteVisionEmbeddingWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="ibm-granite/granite-vision-3.3-2b-embedding",
    languages=["eng-Latn"],
    revision="cee615db64d89d1552a4ee39c50f25c0fc5c66ca",
    release_date="2025-06-11",
    modalities=["image", "text"],
    n_parameters=2_980_000_000,
    memory_usage_mb=11351,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=None,  # proprietary, not public
    citation="""@article{karlinsky2025granitevision,
  title={Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence},
  author={Granite Vision Team and Karlinsky, Leonid and Arbelle, Assaf and Daniels, Abraham and Nassar, Ahmed and Alfassi, Amit and Wu, Bo and Schwartz, Eli and Joshi, Dhiraj and Kondic, Jovana and Shabtay, Nimrod and Li, Pengyuan and Herzig, Roei and Abedin, Shafiq and Perek, Shaked and Harary, Sivan and Barzelay, Udi and Raz Goldfarb, Adi and Oliva, Aude and Wieles, Ben and Bhattacharjee, Bishwaranjan and Huang, Brandon and Auer, Christoph and Gutfreund, Dan and Beymer, David and Wood, David and Kuehne, Hilde and Hansen, Jacob and Shtok, Joseph and Wong, Ken and Bathen, Luis Angel and Mishra, Mayank and Lysak, Maksym and Dolfi, Michele and Yurochkin, Mikhail and Livathinos, Nikolaos and Harel, Nimrod and Azulai, Ophir and Naparstek, Oshri and de Lima, Rafael Teixeira and Panda, Rameswar and Doveh, Sivan and Gupta, Shubham and Das, Subhro and Zawad, Syed and Kim, Yusik and He, Zexue and Brooks, Alexander and Goodhart, Gabe and Govindjee, Anita and Leist, Derek and Ibrahim, Ibrahim and Soffer, Aya and Cox, David and Soule, Kate and Lastras, Luis and Desai, Nirmit and Ofek-koifman, Shila and Raghavan, Sriram and Syeda-Mahmood, Tanveer and Staar, Peter and Drory, Tal and Feris, Rogerio},
  journal={arXiv preprint arXiv:2502.09927},
  year={2025}
}""",
)
