from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.models_protocols import PromptType
from mteb.types import Array, BatchedInput


class EagerEmbedV1Wrapper(AbsEncoder):
    """Wrapper for EagerEmbed single-vector embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        image_size: int = 784,
        use_peft: bool = False,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "peft", model_name, "pip install mteb[peft]"
        )
        requires_package(
            self, "qwen_vl_utils", model_name, "pip install mteb[qwen_vl_utils]"
        )
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_peft = use_peft

        # Handle deprecated torch_dtype parameter
        if 'torch_dtype' in kwargs:
            kwargs['dtype'] = kwargs.pop('torch_dtype')

        # Load model
        if self.use_peft:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(model_name)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.base_model_name_or_path,
                **kwargs
            )
            self.mdl = PeftModel.from_pretrained(base_model, model_name)
            self.mdl = self.mdl.merge_and_unload()
        else:
            self.mdl = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **kwargs
            )

        self.mdl = self.mdl.to(self.device)
        self.mdl.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

    def get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from last token of last hidden state."""
        reps = last_hidden_state[:, -1]
        return reps

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
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)

        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            # For multimodal inputs, concatenate or fuse embeddings
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings

        raise ValueError("No text or image inputs found")

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        """Encode images (documents) into embeddings."""
        from qwen_vl_utils import process_vision_info
        import torchvision.transforms.functional as F

        all_embeds = []

        # Create a new DataLoader with custom collate function to handle images
        def image_collate_fn(batch):
            """Custom collate function that keeps images as a list."""
            collated = {}
            for key in batch[0]:
                collated[key] = [item[key] for item in batch]
            return collated

        # Extract the dataset from the DataLoader and create a new one with proper collation
        dataset = images.dataset
        image_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=image_collate_fn,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in tqdm(image_loader, desc="Encoding images"):
                # Convert batch to PIL images if needed
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]

                # Create messages for each image
                doc_messages = []
                for img in imgs:
                    message = [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': ''},
                                {
                                    'type': 'image',
                                    'image': img,
                                    'resized_height': self.image_size,
                                    'resized_width': self.image_size
                                }
                            ]
                        }
                    ]
                    doc_messages.append(message)

                # Prepare inputs
                doc_texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    ) + "<|endoftext|>"
                    for msg in doc_messages
                ]

                doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
                doc_inputs = self.processor(
                    text=doc_texts,
                    images=doc_image_inputs,
                    videos=doc_video_inputs,
                    padding='longest',
                    return_tensors='pt'
                ).to(self.device)

                # Get embeddings
                output = self.mdl(**doc_inputs, return_dict=True, output_hidden_states=True)
                embeddings = self.get_embedding(output.hidden_states[-1])
                # Convert to float32 and ensure normalization is maintained
                embeddings = embeddings.cpu().to(torch.float32)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                all_embeds.append(embeddings)

        # Concatenate all embeddings
        all_embeds = torch.cat(all_embeds, dim=0)
        return all_embeds

    def get_text_embeddings(
        self,
        texts,
        batch_size: int = 32,
        **kwargs,
    ):
        """Encode texts (queries) into embeddings."""
        from qwen_vl_utils import process_vision_info

        all_embeds = []

        # Create a new DataLoader with custom collate function to handle variable-length texts
        def text_collate_fn(batch):
            """Custom collate function that doesn't try to stack text strings."""
            collated = {}
            for key in batch[0]:
                collated[key] = [item[key] for item in batch]
            return collated

        # Extract the dataset from the DataLoader and create a new one with proper collation
        dataset = texts.dataset
        text_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=text_collate_fn,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in tqdm(text_loader, desc="Encoding texts"):
                # Create query messages
                query_messages = []
                for query in batch["text"]:
                    message = [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': f'Query: {query}'},
                            ]
                        }
                    ]
                    query_messages.append(message)

                # Prepare inputs
                query_texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    ) + "<|endoftext|>"
                    for msg in query_messages
                ]

                query_image_inputs, query_video_inputs = process_vision_info(query_messages)
                query_inputs = self.processor(
                    text=query_texts,
                    images=query_image_inputs,
                    videos=query_video_inputs,
                    padding='longest',
                    return_tensors='pt'
                ).to(self.device)

                # Get embeddings
                output = self.mdl(**query_inputs, return_dict=True, output_hidden_states=True)
                embeddings = self.get_embedding(output.hidden_states[-1])
                # Convert to float32 and ensure normalization is maintained
                embeddings = embeddings.cpu().to(torch.float32)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                all_embeds.append(embeddings)

        # Concatenate all embeddings
        all_embeds = torch.cat(all_embeds, dim=0)
        return all_embeds

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

    def calculate_probs(self, text_embeddings, image_embeddings):
        """Calculate probabilities using softmax over cosine similarities."""
        scores = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1),
            image_embeddings.unsqueeze(0),
            dim=-1
        )
        return scores.softmax(dim=-1)

    def similarity(self, a, b):
        """Calculate cosine similarity between embeddings."""
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(1),
            b.unsqueeze(0),
            dim=-1
        )

EAGER_EMBED_V1_CITATION = """@article{EagerEmbed,
  title={Eager Embed V1: Multimodal Dense Embeddings for Retrieval},
  author={Juan Pablo Balarini},
  year={2025},
  publisher={Eagerworks},
  url={https://github.com/eagerworks/eager-embed},
}"""

Eager_Embed_V1 = ModelMeta(
    loader=EagerEmbedV1Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
        image_size=784,
    ),
    name='eagerworks/eager-embed-v1',
    languages=["fra-Latn", "spa-Latn", "eng-Latn", "deu-Latn"],
    revision='34ab386e65fea9187829bbd595b79622350c0a00',
    release_date='2025-11-20',
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    memory_usage_mb=16929,
    max_tokens=262144,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    framework=["Tevatron"],
    reference="https://huggingface.co/eagerworks/eager-embed-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    citation=EAGER_EMBED_V1_CITATION,
    public_training_code="https://github.com/eagerworks/eager-embed",
    public_training_data="https://github.com/eagerworks/eager-embed/blob/main/dataset_config.yaml"
)
