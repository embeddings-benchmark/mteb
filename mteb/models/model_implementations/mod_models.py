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

MOD_CITATION = """@misc{mod-embedding-2025,
  title={MoD-Embedding: A Fine-tuned Multilingual Text Embedding Model},
  author={MoD Team},
  year={2025},
  url={https://huggingface.co/bflhc/MoD-Embedding}
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

# Predefined prompts for various RTEB tasks
PREDEFINED_PROMPTS = {
    # ========== Open Datasets ==========

    # Legal domain
    "AILACasedocs": "Instruct: Given a legal case scenario, retrieve the most relevant case documents\nQuery: ",
    "AILAStatutes": "Instruct: Given a legal scenario, retrieve the most relevant statute documents\nQuery: ",
    "LegalQuAD": "Instruct: Given a legal question, retrieve relevant legal documents that answer the question\nQuery: ",
    "LegalSummarization": "Instruct: Given a query, retrieve relevant legal documents for summarization\nQuery: ",

    # Code domain
    "AppsRetrieval": "Instruct: Given a query about mobile applications, retrieve relevant app information\nQuery: ",
    "HumanEvalRetrieval": "Instruct: Given a code problem description, retrieve relevant code examples\nQuery: ",
    "MBPPRetrieval": "Instruct: Given a programming problem description, retrieve relevant code solutions\nQuery: ",
    "DS1000Retrieval": "Instruct: Given a data science problem, retrieve relevant code snippets\nQuery: ",
    "FreshStackRetrieval": "Instruct: Given a programming question, retrieve relevant Stack Overflow posts\nQuery: ",

    # Finance domain
    "FinQARetrieval": "Instruct: Given a financial question, retrieve relevant financial documents\nQuery: ",
    "FinanceBenchRetrieval": "Instruct: Given a financial query, retrieve relevant financial information\nQuery: ",
    "HC3FinanceRetrieval": "Instruct: Given a finance-related query, retrieve relevant documents\nQuery: ",

    # Medical domain
    "CUREv1": "Instruct: Given a medical query, retrieve relevant clinical documents\nQuery: ",
    "ChatDoctorRetrieval": "Instruct: Given a medical question, retrieve relevant medical information\nQuery: ",

    # SQL domain
    "WikiSQLRetrieval": "Instruct: Given a natural language query, retrieve relevant SQL examples\nQuery: ",

    # Multilingual
    "MIRACLRetrievalHardNegatives": "Instruct: Given a query, retrieve relevant passages\nQuery: ",

    # ========== Private/Closed Datasets ==========

    # Code domain (Private)
    "Code1Retrieval": "Instruct: Given a code problem description, retrieve relevant code examples\nQuery: ",
    "JapaneseCode1Retrieval": "Instruct: Given a code problem description, retrieve relevant code examples\nQuery: ",

    # Finance domain (Private)
    "EnglishFinance1Retrieval": "Instruct: Given a financial query, retrieve relevant financial documents\nQuery: ",
    "EnglishFinance2Retrieval": "Instruct: Given a financial query, retrieve relevant financial documents\nQuery: ",
    "EnglishFinance3Retrieval": "Instruct: Given a financial query, retrieve relevant financial documents\nQuery: ",
    "EnglishFinance4Retrieval": "Instruct: Given a financial query, retrieve relevant financial documents\nQuery: ",

    # Healthcare domain (Private)
    "EnglishHealthcare1Retrieval": "Instruct: Given a medical question, retrieve relevant medical information\nQuery: ",
    "GermanHealthcare1Retrieval": "Instruct: Given a medical question, retrieve relevant medical information\nQuery: ",

    # Legal domain (Private)
    "FrenchLegal1Retrieval": "Instruct: Given a legal query, retrieve relevant legal documents\nQuery: ",
    "GermanLegal1Retrieval": "Instruct: Given a legal query, retrieve relevant legal documents\nQuery: ",
    "JapaneseLegal1Retrieval": "Instruct: Given a legal query, retrieve relevant legal documents\nQuery: ",

    # General/Multilingual (Private)
    "French1Retrieval": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
    "German1Retrieval": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
}


def mod_instruct_loader(
    model_name_or_path: str, revision: str, **kwargs
) -> EncoderProtocol:
    # Set default prompts_dict if not provided

    model = InstructSentenceTransformerModel(
        model_name_or_path,
        revision=revision,
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        prompt_dict=PREDEFINED_PROMPTS,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model

MoD_Embedding = ModelMeta(
    loader=mod_instruct_loader,
    name="bflhc/MoD-Embedding",
    languages=multilingual_langs,
    open_weights=True,
    revision="acbb5b70fdab262226a6af2bc62001de8021b05c",
    release_date="2025-12-14",
    n_parameters=4021774336,
    memory_usage_mb=7671,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/bflhc/MoD-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=MOD_CITATION,
    adapted_from="Qwen/Qwen3-Embedding-4B",
)
