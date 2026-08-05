from mteb.models import ModelMeta, sentence_transformers_loader
from mteb.models.model_meta import ScoringFunction

nemotron_3_embed_8b_legal = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs={"model_kwargs": {"torch_dtype": "bfloat16"}},
    name="minetta/nemotron-3-embed-8b-legal",
    revision="70d7e152f3a5e676478c9f947b1e23c4ba755019",
    release_date="2026-07-25",
    languages=["eng-Latn"],
    n_parameters=7_952_700_148,
    n_embedding_parameters=536_870_912,
    memory_usage_mb=15168,
    max_tokens=32768,
    embed_dim=4096,
    license="https://huggingface.co/minetta/nemotron-3-embed-8b-legal/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/minetta/nemotron-3-embed-8b-legal",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    # Fine-tuning sources: ECtHR case law, GDPR enforcement decisions, CaseHOLD (US case
    # holdings), SEC EDGAR agreements, public terms-of-service text, Australian tax
    # guidance and UK legislation. None of these is an MTEB dataset, so this set is
    # empty; the base model's declared training data is inherited via adapted_from.
    # Per-document containment audit of every training input against the evaluated
    # corpora: https://huggingface.co/minetta/nemotron-3-embed-8b-legal/blob/main/CONTAMINATION.md
    training_datasets=set(),
    adapted_from="nvidia/Nemotron-3-Embed-8B-BF16",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
)
