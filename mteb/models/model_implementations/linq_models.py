import torch

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

from .e5_instruct import E5_MISTRAL_TRAINING_DATA

LINQ_EMBED_MISTRAL_CITATION = """@misc{LinqAIResearch2024,
  title={Linq-Embed-Mistral:Elevating Text Retrieval with Improved GPT Data Through Task-Specific Control and Quality Refinement},
  author={Junseong Kim and Seolhwa Lee and Jihoon Kwon and Sangmo Gu and Yejin Kim and Minkyung Cho and Jy-yong Sohn and Chanyeol Choi},
  howpublished={Linq AI Research Blog},
  year={2024},
  url={https://getlinq.com/blog/linq-embed-mistral/}
}"""


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


Linq_Embed_Mistral = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        normalized=True,
    ),
    name="Linq-AI-Research/Linq-Embed-Mistral",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0c1a0b0589177079acc552433cad51d7c9132379",
    release_date="2024-05-29",  # initial commit of hf model.
    n_parameters=7_110_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    adapted_from="intfloat/e5-mistral-7b-instruct",
    training_datasets=E5_MISTRAL_TRAINING_DATA,
    citation=LINQ_EMBED_MISTRAL_CITATION,
)
