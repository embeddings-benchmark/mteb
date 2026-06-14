from mteb.models.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
import mteb

minnow_em1 = ModelMeta(
    loader=None,
    name="KiteFishAI/Minnow-Em1-0.6B",
    languages=["eng_Latn"],
    open_weights=True,
    revision="no_revision_available",
    release_date="2026-06-14",
    n_parameters=600_000_000,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=896,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/KiteFishAI/Minnow-Em1-0.6B",
    use_instructions=True,
)
