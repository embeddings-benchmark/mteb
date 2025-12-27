from mteb.models.instruct_wrapper import (
    InstructSentenceTransformerModel,
    instruct_wrapper,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

from .e5_instruct import E5_MISTRAL_TRAINING_DATA


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


SFR_TRAINING_DATA = {  # inherits from e5
    # From previously released blogpost which now have been taken down:
    "FiQA2018",
    "FiQA2018-NL",  # translation not trained on
    "FEVER",
    "FEVERHardNegatives",
    "FEVER-NL",  # translation not trained on
    "FEVER-PL",  # translation not trained on
    "HotpotQA",
    "HotpotQAHardNegatives",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQA-NL",  # translation not trained on
    # source: https://github.com/embeddings-benchmark/leaderboard/issues/41
    # quote: In the realm of Semantic Textual Similarity (STS), it is trained on STS12, STS22, and STSBenchmark
    "STS12",
    "STS22",
    "STSBenchmark",
} | E5_MISTRAL_TRAINING_DATA

SFR_Embedding_2_R = ModelMeta(
    loader=instruct_wrapper,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Salesforce/SFR-Embedding-2_R
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-2_R",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="91762139d94ed4371a9fa31db5551272e0b83818",
    release_date="2024-06-14",  # initial commit of hf model.
    n_parameters=7_110_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Salesforce/SFR-Embedding-2_R",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="intfloat/e5-mistral-7b-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=SFR_TRAINING_DATA,
    citation="""@misc{SFR-embedding-2,
      title={SFR-Embedding-2: Advanced Text Embedding with Multi-stage Training},
      author={Rui Meng*, Ye Liu*, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, Semih Yavuz},
      year={2024},
      url={https://huggingface.co/Salesforce/SFR-Embedding-2_R}
    }
    """,
)

SFR_Embedding_Code_2B_R = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-Code-2B_R",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="c73d8631a005876ed5abde34db514b1fb6566973",
    release_date="2025-01-17",  # initial commit of hf model.
    n_parameters=2_610_000_000,
    memory_usage_mb=4986,
    embed_dim=2304,
    license="cc-by-nc-4.0",
    max_tokens=8192,
    reference="https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="google/gemma-2-2b-it",
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation="""@article{liu2024codexembed,
  title={CodeXEmbed: A Generalist Embedding Model Family for Multiligual and Multi-task Code Retrieval},
  author={Liu, Ye and Meng, Rui and Jot, Shafiq and Savarese, Silvio and Xiong, Caiming and Zhou, Yingbo and Yavuz, Semih},
  journal={arXiv preprint arXiv:2411.12644},
  year={2024}
}
""",
)

SFR_Embedding_Mistral = ModelMeta(
    loader=instruct_wrapper,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-Mistral",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="938c560d1c236aa563b2dbdf084f28ab28bccb11",
    release_date="2024-01-24",  # initial commit of hf model.
    n_parameters=7_110_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Salesforce/SFR-Embedding-Mistral",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=SFR_TRAINING_DATA,
    citation="""
    @misc{SFRAIResearch2024,
  title={SFR-Embedding-Mistral:Enhance Text Retrieval with Transfer Learning},
  author={Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, Semih Yavuz},
  howpublished={Salesforce AI Research Blog},
  year={2024},
  url={https://www.salesforce.com/blog/sfr-embedding/}
}
""",
)
