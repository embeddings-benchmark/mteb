from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

VISTA_CITATION = """@article{zhou2024vista,
  title={VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval},
  author={Zhou, Junjie and Liu, Zheng and Xiao, Shitao and Zhao, Bo and Xiong, Yongping},
  journal={arXiv preprint arXiv:2406.04292},
  year={2024}
}"""


def vista_loader(model_name, **kwargs):
    try:  # a temporal fix for the dependency issues of vista models.
        from visual_bge.modeling import Visualized_BGE
    except ImportError:
        raise ImportError(
            "Please install `visual_bge`, refer to https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge#install-flagembedding."
        )

    class VisualizedBGEWrapper(Visualized_BGE, AbsEncoder):
        """Setting up VISTA

        ```
        git clone https://github.com/FlagOpen/FlagEmbedding.git
        cd FlagEmbedding/research/visual_bge
        pip install -e .
        pip install torchvision timm einops ftfy
        ```
        back to the root folder of mteb; download the vision tower for bge-base
        ```
        cd ..
        wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth?download=true
        ```
        rename it to `visualized_base_en_V1.5.pth`
        ```
        mv Visualized_base_en_v1.5.pth?download=true visualized_base_en_V1.5.pth
        ```
        download the vision tower for bge-m3
        ```
        wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth?download=true
        ```
        rename it to `visualized_m3.pth`
        ```
        mv Visualized_m3.pth?download=true visualized_m3.pth
        ```
        """

        def __init__(
            self,
            model_name_bge: str | None = None,
            model_weight=None,
            normlized: bool = True,
            sentence_pooling_method: str = "cls",
            negatives_cross_device: bool = False,
            temperature: float = 0.02,
            from_pretrained=None,
            image_tokens_num: int | None = None,
            **kwargs: Any,
        ):
            requires_image_dependencies()

            super().__init__(
                model_name_bge=model_name_bge,
                model_weight=model_weight,
                normlized=normlized,
                sentence_pooling_method=sentence_pooling_method,
                negatives_cross_device=negatives_cross_device,
                temperature=temperature,
                from_pretrained=from_pretrained,
            )
            self.image_tokens_num = image_tokens_num
            self.max_text_len_with_image = (
                self.tokenizer.model_max_length - image_tokens_num
            )
            self.eval()

        def encode_text(self, texts: dict[str, torch.Tensor]) -> Array:
            """Currently override Visualized_BGE's the original implementation
            to fix attention_mask & embedding_output dtype misalignment

            Args:
                texts: {"input_ids": ..., "attention_mask": ...}

            Returns:
                Array of text embeddings
            """
            input_ids = texts["input_ids"]
            attention_mask = texts["attention_mask"]

            input_shape = input_ids.size()
            device = input_ids.device

            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            head_mask = [None] * self.depth
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape
            )

            embedding_output = self.bge_embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )

            # this line is missing in vista, currently override "encode_text" only to fix this.
            extended_attention_mask = extended_attention_mask.to(embedding_output.dtype)

            encoder_outputs = self.bge_encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            sequence_output = encoder_outputs[0]

            t_reps = self.sentence_embedding(
                sequence_output, texts["attention_mask"]
            )  # tensor: reps with pooling
            if self.normlized:
                t_reps = torch.nn.functional.normalize(t_reps, dim=-1)
            return t_reps.contiguous()

        def get_text_embeddings(
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            prompt_type: PromptType | None = None,
            input_type: Literal["document", "query"] | None = None,
            **kwargs: Any,
        ):
            all_text_embeddings = []
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                with torch.no_grad():
                    texts = self.tokenizer(
                        batch["text"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    batch_embeddings = self.encode_text(texts.to(self.device))
                all_text_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_text_embeddings, dim=0)

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            prompt_type: PromptType | None = None,
            input_type: Literal["document", "query"] | None = None,
            **kwargs: Any,
        ):
            all_image_embeddings = []
            with torch.no_grad():
                for batch in tqdm(images):
                    imgs = [self.preprocess_val(image) for image in batch["image"]]
                    imgs = torch.stack(imgs)

                    batch_embeddings = self.encode_image(images=imgs.to(self.device))
                    all_image_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_image_embeddings, dim=0)

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            show_progress_bar: bool = True,
            **kwargs: Any,
        ) -> Array:
            if "text" in inputs.dataset.features and "image" in inputs.dataset.features:
                all_fused_embeddings = []
                with torch.no_grad():
                    for batch in tqdm(
                        inputs,
                        disable=not show_progress_bar,
                        desc="Interleaved Encoding",
                    ):
                        texts = self.tokenizer(
                            batch["text"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.max_text_len_with_image,
                        )
                        images = [
                            self.preprocess_val(image) for image in batch["image"]
                        ]
                        images = torch.stack(images)
                        all_fused_embeddings.append(
                            self.encode_mm(
                                images.to(self.device), texts.to(self.device)
                            )
                            .cpu()
                            .to(torch.float32)
                        )
                return torch.cat(all_fused_embeddings, dim=0)
            elif "text" in inputs.dataset.features:
                return self.get_text_embeddings(
                    inputs,
                    task_metadata=task_metadata,
                    prompt_type=prompt_type,
                    **kwargs,
                )
            elif "image" in inputs.dataset.features:
                return self.get_image_embeddings(
                    inputs,
                    task_metadata=task_metadata,
                    prompt_type=prompt_type,
                    **kwargs,
                )
            raise ValueError

    return VisualizedBGEWrapper(model_name, **kwargs)


vista_training_datasets = set(
    # VISTA_S2
)

visualized_bge_base = ModelMeta(
    loader=vista_loader,
    loader_kwargs=dict(
        model_weight="visualized_base_en_V1.5.pth",
        image_tokens_num=196,
    ),
    name="BAAI/bge-visualized-base",
    languages=["eng-Latn"],
    revision="98db10b10d22620010d06f11733346e1c98c34aa",
    release_date="2024-06-06",
    modalities=["image", "text"],
    n_parameters=196_000_000,
    memory_usage_mb=1631,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/JUNJIE99/VISTA_S2",
    framework=["PyTorch"],
    reference="https://huggingface.co/BAAI/bge-visualized",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=vista_training_datasets,
    citation=VISTA_CITATION,
)

visualized_bge_m3 = ModelMeta(
    loader=vista_loader,
    loader_kwargs=dict(
        model_weight="visualized_m3.pth",
        image_tokens_num=256,
    ),
    name="BAAI/bge-visualized-m3",
    languages=["eng-Latn"],
    revision="98db10b10d22620010d06f11733346e1c98c34aa",
    release_date="2024-06-06",
    modalities=["image", "text"],
    n_parameters=872_909_505,
    memory_usage_mb=4263,
    max_tokens=8192,
    embed_dim=1024,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/JUNJIE99/VISTA_S2",
    framework=["PyTorch"],
    reference="https://huggingface.co/BAAI/bge-visualized",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=vista_training_datasets,
    citation=VISTA_CITATION,
)
