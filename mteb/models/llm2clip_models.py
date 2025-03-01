from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


def llm2clip_loader(**kwargs):
    try:
        from llm2vec import LLM2Vec
    except ImportError:
        # https://github.com/baaivision/EVA/tree/master/EVA-CLIP#setup
        raise ImportError(
            "To use the LLM2CLIP models `llm2vec` is required. Please install it with `pip install llm2vec`."
        )

    class LLM2CLIPWrapper:
        def __init__(
            self,
            model_name: str = "microsoft/LLM2CLIP-Openai-L-14-336",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            self.device = device
            from huggingface_hub import snapshot_download

            repo_id = "microsoft/LLM2CLIP-Openai-L-14-336"
            model_folder_path = snapshot_download(repo_id=repo_id)
            model_name_or_path = Path(model_folder_path)
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )

            llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
            config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
            llm_model = AutoModel.from_pretrained(
                llm_model_name,
                torch_dtype=torch.bfloat16,
                config=config,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            llm_model.config._name_or_path = (
                "meta-llama/Meta-Llama-3-8B-Instruct"  #  Workaround for LLM2VEC
            )
            self.l2v = LLM2Vec(
                llm_model,
                tokenizer,
                pooling_mode="mean",
                max_length=512,
                doc_max_length=512,
            )

        def get_text_embeddings(
            self,
            texts: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i : i + batch_size]
                    text_features = self.l2v.encode(
                        batch_texts, convert_to_tensor=True
                    ).to(self.device)
                    text_features = self.model.get_text_features(text_features)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    all_text_embeddings.append(text_features.cpu().to(torch.float32))

            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            return all_text_embeddings

    return LLM2CLIPWrapper(**kwargs)


llm2clip_openai_l_14_336 = ModelMeta(
    loader=partial(
        llm2clip_loader,
        model_name="microsoft/LLM2CLIP-Openai-L-14-336",
    ),
    name="microsoft/LLM2CLIP-Openai-L-14-336",
    languages=["eng_Latn"],
    revision="92512331f393a003c3d98404677f991c188162c9",
    release_date="2024-11-07",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1280,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/microsoft/LLM2CLIP",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-336",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=None,
)


if __name__ == "__main__":
    m = llm2clip_loader()
    m.get_text_embeddings(
        texts=["what is going on blah?", "this is a test for this model."]
    )
