from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader
from mteb.types import PromptType

SARASHINA_V2_INSTRUCTIONS = {
    "Retrieval": {
        "query": "クエリを与えるので、もっともクエリに意味が似ている一節を探してください。",
        "document": "text: ",
    },
    "Reranking": {
        "query": "クエリを与えるので、もっともクエリに意味が似ている一節を探してください。",
        "document": "text: ",
    },
    "Classification": "与えられたドキュメントを適切なカテゴリに分類してください。",
    "Clustering": "与えられたドキュメントのトピックまたはテーマを特定してください。",
    # optimization regarding JMTEB
    "LivedoorNewsClustering.v2": "与えられたニュース記事のトピックを特定してください。",
    "MewsC16JaClustering": "与えられたニュース記事のトピックを特定してください。",
    "SIB200ClusteringS2S": "与えられたテキストのトピックを特定してください。",
    "AmazonReviewsClassification": "与えられたAmazonレビューを適切な評価カテゴリに分類してください。",
    "AmazonCounterfactualClassification": "与えられたAmazonのカスタマーレビューのテキストを反事実か反事実でないかに分類してください。",
    "MassiveIntentClassification": "ユーザーの発話をクエリとして与えるので、ユーザーの意図を見つけてください。",
    "MassiveScenarioClassification": "ユーザーの発話をクエリとして与えるので、ユーザーシナリオを見つけてください。",
    "JapaneseSentimentClassification": "与えられたテキストの感情極性をポジティブ(1)かネガティブか(0)に分類してください。",
    "SIB200Classification": "与えられたテキストのトピックを特定してください。",
    "WRIMEClassification": "与えられたテキストの感情極性（-2:強いネガティブ、-1:ネガティブ、0:ニュートラル、1:ポジティブ、2:強いポジティブ）を分類してください。",
    "JSTS": "クエリを与えるので，もっともクエリに意味が似ている一節を探してください。",
    "JSICK": "クエリを与えるので，もっともクエリに意味が似ている一節を探してください。",
    "JaqketRetrieval": {
        "query": "質問を与えるので、その質問に答えるのに役立つWikipediaの文章を検索してください。",
        "document": "text: ",
    },
    "MrTidyRetrieval": {
        "query": "質問を与えるので、その質問に答えるWikipediaの文章を検索するしてください。",
        "document": "text: ",
    },
    "JaGovFaqsRetrieval": {
        "query": "質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。",
        "document": "text: ",
    },
    "NLPJournalTitleAbsRetrieval.V2": {
        "query": "論文のタイトルを与えるので、タイトルに対応する要約を検索してください。",
        "document": "text: ",
    },
    "NLPJournalTitleIntroRetrieval.V2": {
        "query": "論文のタイトルを与えるので、タイトルに対応する要約を検索してください。",
        "document": "text: ",
    },
    "NLPJournalAbsIntroRetrieval.V2": {
        "query": "論文の序論を与えるので、序論に対応する全文を検索してください。",
        "document": "text: ",
    },
    "NLPJournalAbsArticleRetrieval.V2": {
        "query": "論文の序論を与えるので、序論に対応する全文を検索してください。",
        "document": "text: ",
    },
    "JaCWIRRetrieval": {
        "query": "記事のタイトルを与えるので、そのタイトルと合っている記事の中身を検索してください。",
        "document": "text: ",
    },
    "MIRACLRetrieval": {
        "query": "質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。",
        "document": "text: ",
    },
    "MintakaRetrieval": {
        "query": "質問を与えるので、その質問に答えられるテキストを検索してください。",
        "document": "text: ",
    },
    "MultiLongDocRetrieval": {
        "query": "質問を与えるので、その質問に答えるのに役立つWikipediaの文章を検索してください。",
        "document": "text: ",
    },
    "ESCIReranking": {
        "query": "クエリを与えるので、与えられたWeb検索クエリに答える関連文章を検索してください。",
        "document": "text: ",
    },
    "JQaRAReranking": {
        "query": "質問を与えるので、その質問に答えるのに役立つWikipediaの文章を検索してください。",
        "document": "text: ",
    },
    "JaCWIRReranking": {
        "query": "記事のタイトルを与えるので、そのタイトルと合っている記事の中身を検索してください。",
        "document": "text: ",
    },
    "MIRACLReranking": {
        "query": "質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。",
        "document": "text: ",
    },
    "MultiLongDocReranking": {
        "query": "質問を与えるので、その質問に答えるのに役立つWikipediaの文章を検索してください。",
        "document": "text: ",
    },
}


def sarashina_instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    """Instruction template for Sarashina v2 model.

    Returns the instruction as-is since the prompts already contain the full format.
    For document prompts, returns the instruction directly (e.g., "text: ").
    """
    if not instruction:
        return ""
    if prompt_type == PromptType.document:
        return "text: "
    return f"task: {instruction}\nquery: "


sbintuitions_sarashina_embedding_v2_1b = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=sarashina_instruction_template,
        apply_instruction_to_passages=True,
        prompts_dict=SARASHINA_V2_INSTRUCTIONS,
        max_seq_length=8192,
    ),
    name="sbintuitions/sarashina-embedding-v2-1b",
    model_type=["dense"],
    languages=["jpn-Jpan"],
    open_weights=True,
    revision="1f3408afaa7b617e3445d891310a9c26dd0c68a5",
    release_date="2025-07-30",
    n_parameters=1_224_038_144,
    memory_usage_mb=4669,
    embed_dim=1792,
    license="https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b/blob/main/LICENSE",
    max_tokens=8192,
    reference="https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="sbintuitions/sarashina2.2-1b",
    superseded_by=None,
    training_datasets={"NQ", "MrTidyRetrieval"},
    public_training_code=None,
    public_training_data="https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b",
    citation=None,
    contacts=["Sraym1217", "akiFQC", "lsz05"],
)

sbintuitions_sarashina_embedding_v1_1b = ModelMeta(
    loader=sentence_transformers_loader,
    name="sbintuitions/sarashina-embedding-v1-1b",
    model_type=["dense"],
    languages=["jpn-Jpan"],
    open_weights=True,
    revision="d060fcd8984075071e7fad81baff035cbb3b6c7e",
    release_date="2024-11-22",
    n_parameters=1_224_038_144,
    memory_usage_mb=4669,
    embed_dim=1792,
    license="https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b/blob/main/LICENSE",
    max_tokens=8192,
    reference="https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    adapted_from="sbintuitions/sarashina2.1-1b",
    superseded_by="sbintuitions/sarashina-embedding-v2-1b",
    training_datasets={"NQ", "MrTidyRetrieval"},
    public_training_code=None,
    public_training_data="https://huggingface.co/sbintuitions/sarashina-embedding-v1-1b",
    citation=None,
    contacts=["akiFQC", "lsz05"],
)
