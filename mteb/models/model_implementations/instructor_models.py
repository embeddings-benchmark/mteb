from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction

# Instructor uses a single unified instruction format applied to BOTH queries
# and passages (unlike E5, which instructs queries only). The per-task
# instruction string is supplied by MTEB's prompt system; the template appends
# a trailing space so the encoded text follows the colon exactly as shown in
# the model card, e.g.
#   "Represent the Wikipedia document for retrieval: <text>"
#
# Paper: "One Embedder, Any Task: Instruction-Finetuned Text Embeddings"
# (Su et al., Findings of ACL 2023).
INSTRUCTOR_INSTRUCTION = "{instruction} "

INSTRUCTOR_CITATION = """@inproceedings{su-etal-2023-one,
    title = "One Embedder, Any Task: Instruction-Finetuned Text Embeddings",
    author = "Su, Hongjin and Shi, Weijia and Kasai, Jungo and Wang, Yizhong and
      Hu, Yushi and Ostendorf, Mari and Yih, Wen-tau and Smith, Noah A. and
      Zettlemoyer, Luke and Yu, Tao",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    url = "https://aclanthology.org/2023.findings-acl.71",
}"""

# hkunlp/instructor-* are trained on MEDI, a mixture of 330 datasets built from
# Super-NaturalInstructions and existing embedding training data. The subset
# that overlaps MTEB tasks is not exhaustively annotated upstream; left as None
# here rather than asserting unverified overlaps.
INSTRUCTOR_TRAINING_DATA = None

instructor_base = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=INSTRUCTOR_INSTRUCTION,
        apply_instruction_to_passages=True,
    ),
    name="hkunlp/instructor-base",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0b2f22542d2bd6d60dbff22d22f12c15e6562736",
    release_date="2022-12-19",
    n_parameters=110_000_000,
    n_embedding_parameters=24_652_800,
    memory_usage_mb=420,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/hkunlp/instructor-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/xlang-ai/instructor-embedding",
    public_training_data=None,
    training_datasets=INSTRUCTOR_TRAINING_DATA,
    citation=INSTRUCTOR_CITATION,
)

instructor_large = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=INSTRUCTOR_INSTRUCTION,
        apply_instruction_to_passages=True,
    ),
    name="hkunlp/instructor-large",
    languages=["eng-Latn"],
    open_weights=True,
    revision="54e5ffb8d484de506e59443b07dc819fb15c7233",
    release_date="2022-12-19",
    n_parameters=335_000_000,
    n_embedding_parameters=24_652_800,
    memory_usage_mb=1278,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/hkunlp/instructor-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/xlang-ai/instructor-embedding",
    public_training_data=None,
    training_datasets=INSTRUCTOR_TRAINING_DATA,
    citation=INSTRUCTOR_CITATION,
)

instructor_xl = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=INSTRUCTOR_INSTRUCTION,
        apply_instruction_to_passages=True,
    ),
    name="hkunlp/instructor-xl",
    languages=["eng-Latn"],
    open_weights=True,
    revision="ce48b213095e647a6c3536364b9fa00daf57f436",
    release_date="2023-01-21",
    n_parameters=1_300_000_000,
    n_embedding_parameters=24_652_800,
    memory_usage_mb=4900,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/hkunlp/instructor-xl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/xlang-ai/instructor-embedding",
    public_training_data=None,
    training_datasets=INSTRUCTOR_TRAINING_DATA,
    citation=INSTRUCTOR_CITATION,
)
