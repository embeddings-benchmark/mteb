from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol, PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))  # TODO
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

QWEN3_CITATION = """@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}"""

training_data = {
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "NQ",
    "MSMARCO",
    "HotpotQA",
    "FEVER",
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "CodeSearchNet",
}


def q3e_instruct_loader(
    model_name_or_path: str, revision: str, **kwargs
) -> EncoderProtocol:
    model = InstructSentenceTransformerModel(
        model_name_or_path,
        revision=revision,
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model


Qwen3_Embedding_0B6 = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-0.6B",
    languages=multilingual_langs,
    open_weights=True,
    revision="b22da495047858cce924d27d76261e96be6febc0",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=595776512,
    memory_usage_mb=2272,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)

Qwen3_Embedding_4B = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-4B",
    languages=multilingual_langs,
    open_weights=True,
    revision="636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=4021774336,
    memory_usage_mb=15341,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)

Qwen3_Embedding_8B = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-8B",
    languages=multilingual_langs,
    open_weights=True,
    revision="4e423935c619ae4df87b646a3ce949610c66241c",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=7567295488,
    memory_usage_mb=28866,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)
