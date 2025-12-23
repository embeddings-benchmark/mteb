from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if (
        prompt_type == PromptType.document
    ):  # to avoid this issue: https://huggingface.co/Qwen/Qwen3-Embedding-8B/discussions/21
        return " "
    if not instruction:
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

OCTEN_CITATION = """@misc{octen-embedding-2025,
  title={Octen-Embedding-8B: A Fine-tuned Multilingual Text Embedding Model},
  author={Octen Team},
  year={2025},
  url={https://huggingface.co/bflhc/bflhc/Octen-Embedding-8B}
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
_PREDEFINED_PROMPTS = {
    # ========== Open Datasets ==========
    # Legal domain
    "AILACasedocs": "Given a legal case scenario, retrieve the most relevant case documents",
    "AILAStatutes": "Given a legal scenario, retrieve the most relevant statute documents",
    "LegalQuAD": "Given a legal question, retrieve relevant legal documents that answer the question",
    "LegalSummarization": "Given a query, retrieve relevant legal documents for summarization",
    # Code domain
    "AppsRetrieval": "Given a query about mobile applications, retrieve relevant app information",
    "HumanEvalRetrieval": "Given a code problem description, retrieve relevant code examples",
    "MBPPRetrieval": "Given a programming problem description, retrieve relevant code solutions",
    "DS1000Retrieval": "Given a data science problem, retrieve relevant code snippets",
    "FreshStackRetrieval": "Given a programming question, retrieve relevant Stack Overflow posts",
    # Finance domain
    "FinQARetrieval": "Given a financial question, retrieve relevant financial documents",
    "FinanceBenchRetrieval": "Given a financial query, retrieve relevant financial information",
    "HC3FinanceRetrieval": "Given a finance-related query, retrieve relevant documents",
    # Medical domain
    "CUREv1": "Given a medical query, retrieve relevant clinical documents",
    "ChatDoctorRetrieval": "Given a medical question, retrieve relevant medical information",
    # SQL domain
    "WikiSQLRetrieval": "Given a natural language query, retrieve relevant SQL examples",
    # Multilingual
    "MIRACLRetrievalHardNegatives": "Given a question, retrieve Wikipedia passages that answer the question",
    # ========== Private/Closed Datasets ==========
    # Code domain (Private)
    "Code1Retrieval": "Given a code problem description, retrieve relevant code examples",
    "JapaneseCode1Retrieval": "Given a code problem description, retrieve relevant code examples",
    # Finance domain (Private)
    "EnglishFinance1Retrieval": "Given a financial query, retrieve relevant financial documents",
    "EnglishFinance2Retrieval": "Given a financial query, retrieve relevant financial documents",
    "EnglishFinance3Retrieval": "Given a financial query, retrieve relevant financial documents",
    "EnglishFinance4Retrieval": "Given a financial query, retrieve relevant financial documents",
    # Healthcare domain (Private)
    "EnglishHealthcare1Retrieval": "Given a medical question, retrieve relevant medical information",
    "GermanHealthcare1Retrieval": "Given a medical question, retrieve relevant medical information",
    # Legal domain (Private)
    "FrenchLegal1Retrieval": "Given a legal query, retrieve relevant legal documents",
    "GermanLegal1Retrieval": "Given a legal query, retrieve relevant legal documents",
    "JapaneseLegal1Retrieval": "Given a legal query, retrieve relevant legal documents",
    # General/Multilingual (Private)
    "French1Retrieval": "Given a query, retrieve relevant passages",
    "German1Retrieval": "Given a query, retrieve relevant passages",
}


Octen_Embedding_8B = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=True,
        prompts_dict=_PREDEFINED_PROMPTS,
        max_seq_length=18480,
        model_kwargs={"torch_dtype": "bfloat16"},
    ),
    name="bflhc/Octen-Embedding-8B",
    languages=multilingual_langs,
    open_weights=True,
    revision="2030603c2926ab005fafd824fac5911e271be21f",
    release_date="2025-12-23",
    n_parameters=7567295488,
    memory_usage_mb=14433,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/bflhc/Octen-Embedding-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=OCTEN_CITATION,
    adapted_from="Qwen/Qwen3-Embedding-8B",
)
