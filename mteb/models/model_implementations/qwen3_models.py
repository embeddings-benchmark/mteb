from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


_QWEN3_INSTRUCTION_TEMPLATE = "Instruct: {instruction}\nQuery:"

# Hard-coded fallbacks from the Qwen3-Embedding reference implementation:
# https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/qwen3_embedding_model.py
_QWEN3_SYMMETRIC_INSTRUCTION = "Retrieve semantically similar text"
_QWEN3_BITEXT_INSTRUCTION = "Retrieve parallel sentences"
_QWEN3_RETRIEVAL_QUERY_FALLBACK = "Retrieval relevant passage for the given query."

# Per-task prompts mirrored from
# https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json
# The "passage" key in the original JSON is renamed to "document" to match
# `mteb.types.PromptType.document.value`.
QWEN3_TASK_PROMPTS: dict[str, str | dict[str, str]] = {
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given a online banking query, find the corresponding intents",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TNews": "Classify the fine-grained category of the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MmarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    "ArguAna": {
        "query": "Given a claim, find documents that refute the claim",
        "document": "Given a claim, find documents that refute the claim",
    },
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
    "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
    "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text",
    "SICKFr": "Retrieve semantically similar text",
    "SummEvalFr": "Given a news summary, retrieve other semantically similar summaries",
    "MasakhaNEWSClassification": "Classify the News in the given texts into one of the seven category: politics,sports,health,business,entertainment,technology,religion ",
    "OpusparcusPC": "Retrieve semantically similar text",
    "PawsX": "Retrieve semantically similar text",
    "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
    "AlloProfClusteringS2S": "Identify the topic of document titles from Allo Prof dataset",
    "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
    "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
    "MLSUMClusteringS2S": "Identify the topic or theme of the given articles based on the titles",
    "SyntecReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofRetrieval": "Given a question, retrieve passages that answer the question",
    "BSARDRetrieval": "Given a question, retrieve passages that answer the question",
    "SyntecRetrieval": "Given a question, retrieve passages that answer the question",
    "XPQARetrieval": "Given a question, retrieve passages that answer the question",
    "MintakaRetrieval": "Given a question, retrieve passages that answer the question",
    "CBD": "Classify the sentiment of polish tweet reviews",
    "PolEmo2.0-IN": "Classify the sentiment of in-domain (medicine and hotels) online reviews",
    "PolEmo2.0-OUT": "Classify the sentiment of out-of-domain (products and school) online reviews",
    "AllegroReviews": "Classify the sentiment of reviews from e-commerce marketplace Allegro",
    "PAC": 'Classify the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
    "SICK-E-PL": "Retrieve semantically similar text",
    "SICK-R-PL": "Retrieve semantically similar text",
    "STS22": "Retrieve semantically similar text",
    "AFQMC": "Retrieve semantically similar text",
    "BQ": "Retrieve semantically similar text",
    "LCQMC": "Retrieve semantically similar text",
    "PAWSX": "Retrieve semantically similar text",
    "QBQTC": "Retrieve semantically similar text",
    "STS12": "Retrieve semantically similar text",
    "PPC": "Retrieve semantically similar text",
    "CDSC-E": "Retrieve semantically similar text",
    "PSC": "Retrieve semantically similar text",
    "8TagsClustering": "Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
    "ArguAna-PL": "Given a claim, find documents that refute the claim",
    "DBPedia-PL": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FiQA-PL": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA-PL": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO-PL": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus-PL": "Given a question, retrieve relevant documents that best answer the question",
    "NQ-PL": "Given a question, retrieve Wikipedia passages that answer the question",
    "Quora-PL": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS-PL": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact-PL": "Given a scientific claim, retrieve documents that support or refute the claim",
    "TRECCOVID-PL": "Given a query on COVID-19, retrieve documents that answer the query",
    "GeoreviewClassification": "Classify the organization rating based on the reviews",
    "HeadlineClassification": "Classify the topic or theme of the given news headline",
    "InappropriatenessClassification": "Classify the given message as either sensitive topic or not",
    "KinopoiskClassification": "Classify the sentiment expressed in the given movie review text",
    "RuReviewsClassification": "Classify product reviews into positive, negative or neutral sentiment",
    "RuSciBenchGRNTIClassification": "Classify the category of scientific papers based on the titles and abstracts",
    "RuSciBenchOECDClassification": "Classify the category of scientific papers based on the titles and abstracts",
    "GeoreviewClusteringP2P": "Identify the organization category based on the reviews",
    "RuSciBenchGRNTIClusteringP2P": "Identify the category of scientific papers based on the titles and abstracts",
    "RuSciBenchOECDClusteringP2P": "Identify the category of scientific papers based on the titles and abstracts",
    "TERRa": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "RuBQReranking": "Given a question, retrieve Wikipedia passages that answer the question",
    "RiaNewsRetrieval": "Given a headline, retrieval relevant articles",
    "RuBQRetrieval": "Given a question, retrieve Wikipedia passages that answer the question",
    "RUParaPhraserSTS": "Retrieve semantically similar text",
    "RuSTSBenchmarkSTS": "Retrieve semantically similar text",
    "AppsRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "COIRCodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeEditSearchRetrieval": "Given a piece of code, retrieval code that in the ",
    "CodeFeedbackMT": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeFeedbackST": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeSearchNetCCRetrieval": "Given a code comment, retrieve the code snippet corresponding to that comment.",
    "CodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeTransOceanContest": "Given a piece for code, retrieval semantically similar code",
    "CodeTransOceanDL": "Given a piece for code, retrieval semantically similar code",
    "CosQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "StackOverflowQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "SyntheticText2SQL": "Given a user's question, retrieve SQL queries that are appropriate responses to the question",
    "BibleNLPBitextMining": "Retrieve parallel sentences",
    "BUCC.v2": "Retrieve parallel sentences",
    "DiaBlaBitextMining": "Retrieve parallel sentences",
    "FloresBitextMining": "Retrieve parallel sentences",
    "IN22GenBitextMining": "Retrieve parallel sentences",
    "IndicGenBenchFloresBitextMining": "Retrieve parallel sentences",
    "NollySentiBitextMining": "Retrieve parallel sentences",
    "NTREXBitextMining": "Retrieve parallel sentences",
    "NusaTranslationBitextMining": "Retrieve parallel sentences",
    "NusaXBitextMining": "Retrieve parallel sentences",
    "Tatoeba": "Retrieve parallel sentences",
    "BulgarianStoreReviewSentimentClassfication": "Classify user reviews into positive or negative sentiment",
    "CzechProductReviewSentimentClassification": "Classify product reviews into positive or negative sentiment",
    "GreekLegalCodeClassification": "Given a greek legal text, classify its topic",
    "DBpediaClassification": "Given a Wikipedia articles, categorized it into classes based on its DBpedia ontology",
    "FinancialPhrasebankClassification": "Given financial news, categorized by sentiment into positive, negative, or neutral",
    "PoemSentimentClassification": "Gvien a poem, categorized by sentiment into positive, no_impact, negative or mixed",
    "TweetTopicSingleClassification": "Gvien a twitter, classify its topic",
    "EstonianValenceClassification": "Given a news article, categorized by sentiment into negatiivne, positiivne, neutraalne or vastuolulin",
    "FilipinoShopeeReviewsClassification": "Given a shop review, classify its rating on a scale from 1 to 5",
    "GujaratiNewsClassification": "Given a Gujarati news articles, classify ist topic",
    "SentimentAnalysisHindi": "Given a hindi text, categorized by sentiment into positive, negative or neutral",
    "IndonesianIdClickbaitClassification": "Given an Indonesian news headlines, classify its into clickbait or non-clickbait",
    "ItaCaseholdClassification": "Given a judgments, classify its topic",
    "KorSarcasmClassification": "Given a twitter, categorized it into sarcasm or not_sarcasm",
    "KurdishSentimentClassification": "Given a text, categorized by sentiment into positive or negative",
    "MacedonianTweetSentimentClassification": "Given a Macedonian tweet, categorized by sentiment into positive, negative, or neutral",
    "AfriSentiClassification": "Given a text, categorized by sentiment into positive, negative, or neutral",
    "CataloniaTweetClassification": "Given a tweet, categorized by sentiment into AGAINST, FAVOR or NEUTRAL",
    "CyrillicTurkicLangClassification": "Given a text, classify its language",
    "IndicLangClassification": "Given a text, classify its language",
    "MultiHateClassification": "Given a text, categorized by sentiment into hate or non-hate",
    "NusaParagraphEmotionClassification": "Given a paragraph, classify its emotion",
    "NusaX-senti": "Given a text, categorized by sentiment into positive or negative",
    "SwissJudgementClassification": "Given a news article, categorized it into approval or dismissal",
    "NepaliNewsClassification": "Given a news article, categorized it into business, entertainment or sports",
    "OdiaNewsClassification": "Given a news article, categorized it into business, entertainment or sports",
    "PunjabiNewsClassification": "Given a news article, categorized it into two-classes",
    "SinhalaNewsClassification": "Given a news article, categorized it into political, business, technology, sports and Entertainment",
    "CSFDSKMovieReviewSentimentClassification": "Given a movie review, classify its rating on a scale from 0 to 5",
    "SiswatiNewsClassification": "Given a news article, classify its topic",
    "SlovakMovieReviewSentimentClassification": "Given a movie review, categorized it into positive or negative",
    "SwahiliNewsClassification": "Given a news article, classify its domain",
    "TswanaNewsClassification": "Given a news article, classify its topic",
    "IsiZuluNewsClassification": "Given a news article, classify its topic",
    "WikiCitiesClustering": "Identify of Wikipedia articles of cities by country",
    "RomaniBibleClustering": "Identify verses from the Bible in Kalderash Romani by book.",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BigPatentClustering.v2": "Identify the category of documents from the Big Patent dataset",
    "AlloProfClusteringS2S.v2": "Identify the topic of document titles from Allo Prof dataset",
    "HALClusteringS2S.v2": "Identify the topic of titles from HAL",
    "SIB200ClusteringS2S": "Identify the category of documents",
    "WikiClusteringP2P.v2": "Identify the category of wiki passages",
    "PlscClusteringP2P.v2": "Identify the category of titles+abstracts from Library of Science",
    "KorHateSpeechMLClassification": "Given a Korean online news comments, classify its fine-grained hate speech classes",
    "MalteseNewsClassification": "Given a maltese new, classify its topic",
    "MultiEURLEXMultilabelClassification": "Given a text, classify its topic",
    "BrazilianToxicTweetsClassification": "Given a tweet, classify its topic",
    "CTKFactsNLI": "Retrieve semantically similar text",
    "indonli": "Retrieve semantically similar text",
    "ArmenianParaphrasePC": "Retrieve semantically similar text",
    "PawsXPairClassification": "Retrieve semantically similar text",
    "RTE3": "Retrieve semantically similar text",
    "XNLI": "Retrieve semantically similar text",
    "PpcPC": "Retrieve semantically similar text",
    "GermanSTSBenchmark": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "FaroeseSTS": "Retrieve semantically similar text",
    "FinParaSTS": "Retrieve semantically similar text",
    "JSICK": "Retrieve semantically similar text",
    "IndicCrosslingualSTS": "Retrieve semantically similar text",
    "SemRel24STS": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22.v2": "Retrieve semantically similar text",
    "STSES": "Retrieve semantically similar text",
    "STSB": "Retrieve semantically similar text",
    "AILAStatutes": "Identifying the most relevant statutes for a given situation",
    "HagridRetrieval": "Retrieval the relevant passage for the given query",
    "LegalBenchCorporateLobbying": "Retrieval the relevant passage for the given query",
    "LEMBPasskeyRetrieval": "Retrieval the relevant passage for the given query",
    "BelebeleRetrieval": "Retrieval the relevant passage for the given query",
    "MLQARetrieval": "Retrieval the relevant passage for the given query",
    "StatcanDialogueDatasetRetrieval": "Retrieval the relevant passage for the given query",
    "WikipediaRetrievalMultilingual": "Retrieval the relevant passage for the given query",
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "WebLINXCandidatesReranking": "Retrieval the relevant passage for the given query",
    "WikipediaRerankingMultilingual": "Retrieval the relevant passage for the given query",
    "STS15": "Retrieve semantically similar text",
    "MIRACLRetrievalHardNegatives": "Retrieval relevant passage for the given query",
    "BIOSSES": "Retrieve semantically similar text",
    "CQADupstackRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "STS16": "Retrieve semantically similar text",
    "SummEval": "Retrieve semantically similar text",
    "ATEC": "Retrieve semantically similar text",
}


class Qwen3EmbeddingWrapper(InstructSentenceTransformerModel):
    """Wrapper that reproduces the instruction-selection behaviour of the
    official Qwen3-Embedding evaluation code so MTEB scores match the
    numbers published by the Qwen team.

    Reference: https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/qwen3_embedding_model.py
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        max_seq_length: int | None = 8192,
        padding_side: str | None = "left",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name,
            revision=revision,
            instruction_template=_QWEN3_INSTRUCTION_TEMPLATE,
            apply_instruction_to_passages=False,
            max_seq_length=max_seq_length,
            padding_side=padding_side,
            **kwargs,
        )

    def get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        task_name = task_metadata.name
        task_type = task_metadata.type

        sym_task = False
        instruction: str | None
        if task_name in QWEN3_TASK_PROMPTS:
            entry = QWEN3_TASK_PROMPTS[task_name]
            if isinstance(entry, dict):
                prompt_key = prompt_type.value if prompt_type else ""
                instruction = entry.get(prompt_key, "")
                sym_task = True
            else:
                instruction = entry
        else:
            instruction = super().get_instruction(task_metadata, prompt_type)

        if (
            "Retrieval" in task_type
            and not sym_task
            and prompt_type != PromptType.query
        ):
            return ""
        if task_type in {"STS", "PairClassification"}:
            return _QWEN3_SYMMETRIC_INSTRUCTION
        if task_type == "BitextMining":
            return _QWEN3_BITEXT_INSTRUCTION
        if (
            "Retrieval" in task_type
            and prompt_type == PromptType.query
            and not instruction
        ):
            return _QWEN3_RETRIEVAL_QUERY_FALLBACK
        return instruction or ""


def q3e_instruct_loader(model_name_or_path: str, revision: str, **kwargs: Any):
    model = Qwen3EmbeddingWrapper(model_name_or_path, revision=revision, **kwargs)
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only uses left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model


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


Qwen3_Embedding_0B6 = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-0.6B",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="b22da495047858cce924d27d76261e96be6febc0",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=595776512,
    n_embedding_parameters=155309056,
    memory_usage_mb=1136,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)

Qwen3_Embedding_4B = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-4B",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=4021774336,
    n_embedding_parameters=388262400,
    memory_usage_mb=7671,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)

Qwen3_Embedding_8B = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-8B",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="4e423935c619ae4df87b646a3ce949610c66241c",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=7567295488,
    n_embedding_parameters=621219840,
    memory_usage_mb=14433,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)
