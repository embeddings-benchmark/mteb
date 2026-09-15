from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_implementations.octen_models import (
    _PREDEFINED_PROMPTS,
    instruction_template,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

OCTEN_LAW_8B_V1_CITATION = "@misc{octen-law-8b-v1,\n  title={Octen Law 8B v1: a multilingual legal text embedding model},\n  author={{Litil Labs}},\n  year={2026},\n  howpublished={\\url{https://huggingface.co/litillabs/octen-law-8b-v1}}\n}"


litillabs_octen_law_8b_v1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="litillabs/octen-law-8b-v1",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "zho-Hans"],
    open_weights=True,
    revision="cc2b41645060edebf7246cb8b53064173a03b6c4",
    release_date="2026-08-25",
    n_parameters=7_567_295_488,
    n_embedding_parameters=621_219_840,
    memory_usage_mb=14_433,
    embed_dim=4096,
    max_tokens=40_960,
    license="apache-2.0",
    reference="https://huggingface.co/litillabs/octen-law-8b-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    # WikiQA TRAIN was used directly. GerDaLIRSmall and LeCaRDv2 are marked
    # conservatively because the training pack contains disclosed
    # cross-direction benchmark-adjacent near-duplicates for those tasks.
    training_datasets={"GerDaLIRSmall", "LeCaRDv2", "WikiQA"},
    citation=OCTEN_LAW_8B_V1_CITATION,
    adapted_from="Octen/Octen-Embedding-8B",
)


litillabs_litil_embed_0b6 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=True,
        prompts_dict=_PREDEFINED_PROMPTS,
        max_seq_length=18480,
        model_kwargs={"torch_dtype": "bfloat16"},
    ),
    name="litillabs/litil-embed-0.6b",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "zho-Hans"],
    open_weights=True,
    revision="6628f8975520a5d853927e81c483447c268b83e8",
    release_date="2026-09-13",
    n_parameters=595_776_512,
    n_embedding_parameters=155_309_056,
    memory_usage_mb=1_136,
    embed_dim=1024,
    max_tokens=32_768,
    license="apache-2.0",
    reference="https://huggingface.co/litillabs/litil-embed-0.6b",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"GerDaLIRSmall", "LeCaRDv2"},
    adapted_from="Octen/Octen-Embedding-0.6B",
)
