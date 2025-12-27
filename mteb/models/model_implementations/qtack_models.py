from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

mini_gte_datasets = {
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "HotPotQA",
    "HotPotQAHardNegatives",
    "FEVER",
    "FEVERHardNegatives",
    # Other Datasets (see GTE-series)
    # TriviaQA
    # SNLI
    # MNLI
    # QuoraDuplicateQuestions
    # StackExchange
    # MEDI
    # BERRI
}

mini_gte = ModelMeta(
    loader=sentence_transformers_loader,
    name="prdev/mini-gte",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="7fbe6f9b4cc42615e0747299f837ad7769025492",
    release_date="2025-01-28",
    n_parameters=int(66.3 * 1e6),
    memory_usage_mb=253,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/prdev/mini-gte",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=mini_gte_datasets,
    adapted_from="distilbert/distilbert-base-uncased",
)
