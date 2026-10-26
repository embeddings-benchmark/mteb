"""Last-prompt embeddings from unchanged, pinned Qwen3.5 and NuExtract3 weights.

Each embedding is the last hidden state of the final active prompt token,
after the decoder's final RMSNorm: the state that predicts the first output
token. No token averaging or L2 normalization is applied.

The submitted English v2 results were generated with NuMind's extension:
https://github.com/numindai/vLLM-Last-Hidden-State
It exposes this prompt hidden state from vLLM's generation forward pass.
This MTEB implementation uses Transformers directly for that representation.
Weights are unchanged; this adapter constructs the complete prompts client-side.
Historical serving precision/backend settings were not recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class Recipe(TypedDict):
    name: str
    checkpoint: str
    revision: str
    parameters: int
    prompt: str


RECIPES: dict[str, Recipe] = {
    "qwen_e5v": {
        "name": "Qwen/Qwen3.5-4B-e5v-last-prompt",
        "checkpoint": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "parameters": 4659865088,
        "prompt": "{text}\n Summary above sentence in one word: ",
    },
    "nuextract_revised": {
        "name": "numind/NuExtract3-revised-last-prompt",
        "checkpoint": "numind/NuExtract3",
        "revision": "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1",
        "parameters": 4539265536,
        "prompt": "{text}",
    },
}


# The client constructs NuExtract's complete document-identity prompt, including
# the assistant JSON prefill, and passes its tokens directly to the model.
NUEXTRACT_PREFIX = (
    "【task】structured\n"
    '【template_start】{"short description": "string"}【template_end】\n'
    "【instructions_start】Describe the stable identity of this document so it "
    "can be matched to similar documents. Include its document type and purpose, "
    "issuing organization or publication, main subject or research area, and "
    "distinctive layout, headings, logos, or form fields. Emphasize "
    "characteristics shared by documents from the same source, subject area, "
    "or template. Ignore dates, amounts, identifiers, individual people, and "
    "other values that vary between instances. Be concise and specific."
    "【instructions_end】\n【document_start】\n"
)
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LastPromptEncoder(AbsEncoder):
    """One forward pass; select the last active prompt token, without L2 norm."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        recipe: str,
        device: str | None = None,
        dtype: str | torch.dtype = "bfloat16",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        attn_implementation: str = "sdpa",
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.recipe = RECIPES[recipe]
        self.recipe_name = recipe
        if model_name != self.recipe["name"] or revision != self.recipe["revision"]:
            raise ValueError("Model name/revision does not match the registered recipe")
        if kwargs:
            raise ValueError(f"Unsupported recipe overrides: {sorted(kwargs)}")
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.recipe["checkpoint"],
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # AutoModel omits the language-model head: last_hidden_state is already
        # post-final-RMSNorm. No generation or all-layer hidden-state storage.
        self.model = (
            AutoModel.from_pretrained(
                self.recipe["checkpoint"],
                revision=revision,
                dtype=dtype,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=False,
                attn_implementation=attn_implementation,
            )
            .to(self.device)
            .eval()
        )
        self.max_length = self.model.config.text_config.max_position_embeddings

    def render_prompt(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Only text inputs are supported")
        # Preserve evaluated whitespace handling. For Qwen the E5-V suffix is
        # added BEFORE trimming the whole message.
        if self.recipe_name == "qwen_e5v":
            content = self.recipe["prompt"].format(text=text).strip()
            return "<|im_start|>user\n" + content + ASSISTANT_PREFIX
        return (
            "<|im_start|>user\n"
            + NUEXTRACT_PREFIX
            + text.strip()
            + "\n【document_end】"
            + ASSISTANT_PREFIX
            + '{\n  "short description": "'
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
        import torch

        batches = []
        with torch.inference_mode():
            for batch in inputs:
                if "text" not in batch or {"image", "audio", "video"}.intersection(
                    batch
                ):
                    raise ValueError("This registration accepts text-only batches")
                # MTEB already combines retrieval title/body/query into text.
                prompts = [self.render_prompt(text) for text in batch["text"]]
                if not prompts:
                    continue
                tokens = self.tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                ).to(self.device)
                if tokens["input_ids"].shape[1] > self.max_length:
                    raise ValueError("Prompt exceeds the checkpoint context window")
                # Right padding leaves the active causal/recurrent prefix intact.
                # Explicit positions keep each document independent of batch length.
                positions = tokens["attention_mask"].long().cumsum(-1) - 1
                positions.masked_fill_(tokens["attention_mask"] == 0, 0)
                hidden = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    position_ids=positions,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                last = tokens["attention_mask"].sum(dim=1) - 1
                rows = hidden[torch.arange(len(prompts), device=hidden.device), last]
                vectors = rows.float().cpu().numpy()
                if (
                    vectors.shape != (len(prompts), 2560)
                    or not np.isfinite(vectors).all()
                    or np.any(np.linalg.norm(vectors, axis=1) == 0)
                ):
                    raise ValueError("Expected finite, nonzero 2560-dimensional states")
                batches.append(vectors)
        return (
            np.concatenate(batches)
            if batches
            else np.empty((0, 2560), dtype=np.float32)
        )


def make_meta(recipe: str) -> ModelMeta:
    config = RECIPES[recipe]
    return ModelMeta(
        loader=LastPromptEncoder,
        loader_kwargs={"recipe": recipe},
        name=config["name"],
        revision=config["revision"],
        release_date=None,
        languages=None,
        n_parameters=config["parameters"],
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=2560,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        extra_requirements_groups=["last-prompt"],
        framework=["PyTorch", "Transformers", "safetensors"],
        similarity_fn_name="cosine",
        use_instructions=True,
        training_datasets=None,
        modalities=["text"],
        model_type=["dense"],
        reference="https://huggingface.co/" + config["checkpoint"],
        adapted_from=config["checkpoint"],
    )


qwen35_e5v_last_prompt = make_meta("qwen_e5v")
nuextract3_revised_last_prompt = make_meta("nuextract_revised")
