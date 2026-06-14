from mteb.models.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
import mteb

minnow_em1_06b = ModelMeta(
    loader=_loader,
    name="KiteFishAI/Minnow-Em1-0.6B",
    languages=["eng_Latn"],
    open_weights=True,
    revision="no_revision_available",  # replace with actual HF commit hash
    release_date="2026-06-14",
    n_parameters=600_000_000,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1024,                    # Qwen3-0.6B hidden size
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/KiteFishAI/Minnow-Em1-0.6B",
    use_instructions=True,
    adapted_from="Qwen/Qwen3-0.6B",   # updated base
)
