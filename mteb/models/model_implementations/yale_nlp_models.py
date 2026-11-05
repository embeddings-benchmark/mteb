from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    """Prefix queries with "Instruct: ...\nQuery:" and leave documents bare.

    Matches the template in the checkpoint's config_sentence_transformers.json.
    """
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        instruction = (
            instruction[prompt_type]
            if prompt_type is not None
            else next(iter(instruction.values()))
        )
    return f"Instruct: {instruction}\nQuery:"


RTRIEVER_CITATION = """@inproceedings{zhao-etal-2026-rethinking,
    title = "Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems",
    author = "Zhao, Yilun  and
      Wei, Jinbiao  and
      Song, Tingyu  and
      Zhang, Siyue  and
      Zhao, Chen  and
      Cohan, Arman",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1705/",
    doi = "10.18653/v1/2026.acl-long.1705",
    pages = "36776--36806",
    ISBN = "979-8-89176-390-6",
}"""

RTRIEVER_4B = ModelMeta(
    # RTriever-4B keeps the Qwen3-Embedding interface it was fine-tuned from:
    # last-token pooling, cosine similarity, and an "Instruct: ...\nQuery:"
    # prefix on queries only.
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="yale-nlp/RTriever-4B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="2133b3d737c602f70b73642944e19ab4b8c0e70c",
    release_date="2026-04-30",
    n_parameters=4021774336,
    n_embedding_parameters=388262400,
    memory_usage_mb=7670,
    embed_dim=2560,
    max_tokens=32768,
    license="mit",
    reference="https://huggingface.co/yale-nlp/RTriever-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    # Trained on synthetic data generated for this model; no MTEB task data, and
    # in particular none of BRIGHT or BRIGHT-Pro (confirmed by the authors).
    training_datasets=set(),
    adapted_from="Qwen/Qwen3-Embedding-4B",
    citation=RTRIEVER_CITATION,
)
