from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


def instruction_template(
    instruction: str | dict[str, str], prompt_type: PromptType | None = None
) -> str:
    """Return the full prompt used for evaluation.

    `opensearch_task_prompts` stores complete prompt strings, so they are returned as-is.
    Retrieval prompts (ending in "Query: ") apply to queries only; retrieval documents are
    encoded without a prompt. Prompts of the other task types (ending in "Query:") apply to
    every input, including the candidates of reranking tasks. Tasks without an entry fall
    back to the task's default instruction in the retrieval format (queries only).
    """
    if not instruction:
        return ""
    if isinstance(instruction, dict):
        instruction = instruction.get(prompt_type.value if prompt_type else "query", "")
        if not instruction:
            return ""
    is_full_prompt = instruction.startswith("Instruct: ")
    if prompt_type == PromptType.document and (
        not is_full_prompt or instruction.endswith("Query: ")
    ):
        return ""
    if is_full_prompt:
        return instruction
    return f"Instruct: {instruction}\nQuery: "


# Prompts used to produce the reported MTEB(eng, v2) and BEIR results.
# Retrieval tasks end with "\nQuery: " (trailing space); the other task types with "\nQuery:".
opensearch_task_prompts = {
    "AmazonCounterfactualClassification": "Instruct: Classify a given Amazon customer review text as either counterfactual or not-counterfactual\nQuery:",
    "ArguAna": "Instruct: Given a claim, find documents that refute the claim\nQuery: ",
    "ArXivHierarchicalClusteringP2P": "Instruct: Identify the main and secondary category of Arxiv papers based on the titles and abstracts\nQuery:",
    "ArXivHierarchicalClusteringS2S": "Instruct: Identify the main and secondary category of Arxiv papers based on the titles\nQuery:",
    "AskUbuntuDupQuestions": "Instruct: Retrieve duplicate questions from AskUbuntu forum\nQuery:",
    "Banking77Classification": "Instruct: Given a online banking query, find the corresponding intents\nQuery:",
    "BiorxivClusteringP2P.v2": "Instruct: Identify the main category of Biorxiv papers based on the titles and abstracts\nQuery:",
    "BIOSSES": "Instruct: Retrieve semantically similar text\nQuery:",
    "ClimateFEVER": "Instruct: Given a claim about climate change, retrieve documents that support or refute the claim\nQuery: ",
    "ClimateFEVERHardNegatives": "Instruct: Given a claim about climate change, retrieve documents that support or refute the claim\nQuery: ",
    "CQADupstackAndroidRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackEnglishRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackGamingRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackGisRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackMathematicaRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackPhysicsRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackProgrammersRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackStatsRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackTexRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackUnixRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackWebmastersRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackWordpressRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "DBPedia": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "FEVER": "Instruct: Given a claim, retrieve documents that support or refute the claim\nQuery: ",
    "FEVERHardNegatives": "Instruct: Given a claim, retrieve documents that support or refute the claim\nQuery: ",
    "FiQA2018": "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ",
    "HotpotQA": "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
    "HotpotQAHardNegatives": "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
    "ImdbClassification": "Instruct: Classify the sentiment expressed in the given movie review text from the IMDB dataset\nQuery:",
    "MassiveIntentClassification": "Instruct: Given a user utterance as query, find the user intents\nQuery:",
    "MassiveScenarioClassification": "Instruct: Given a user utterance as query, find the user scenarios\nQuery:",
    "MedrxivClusteringP2P.v2": "Instruct: Identify the main category of Medrxiv papers based on the titles and abstracts\nQuery:",
    "MedrxivClusteringS2S.v2": "Instruct: Identify the main category of Medrxiv papers based on the titles\nQuery:",
    "MindSmallReranking": "Instruct: Retrieve relevant news articles based on user browsing history\nQuery:",
    "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "MTOPDomainClassification": "Instruct: Classify the intent domain of the given utterance in task-oriented conversation\nQuery:",
    "NFCorpus": "Instruct: Given a medical question, retrieve relevant health and nutrition documents\nQuery: ",
    "NQ": "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: ",
    "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\nQuery: ",
    "SCIDOCS": "Instruct: Given a scientific paper title, retrieve paper abstracts that are cited by the given paper\nQuery: ",
    "SciFact": "Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: ",
    "SICK-R": "Instruct: Retrieve semantically similar text\nQuery:",
    "SprintDuplicateQuestions": "Instruct: Retrieve duplicate questions from Sprint forum\nQuery:",
    "StackExchangeClustering.v2": "Instruct: Identify the topic or theme of StackExchange posts based on the titles\nQuery:",
    "StackExchangeClusteringP2P.v2": "Instruct: Identify the topic or theme of StackExchange posts based on the given paragraphs\nQuery:",
    "STS12": "Instruct: Retrieve semantically similar text\nQuery:",
    "STS13": "Instruct: Retrieve semantically similar text\nQuery:",
    "STS14": "Instruct: Retrieve semantically similar text\nQuery:",
    "STS15": "Instruct: Retrieve semantically similar text\nQuery:",
    "STS17": "Instruct: Retrieve semantically similar text\nQuery:",
    "STS22.v2": "Instruct: Retrieve semantically similar text\nQuery:",
    "STSBenchmark": "Instruct: Retrieve semantically similar text\nQuery:",
    "SummEvalSummarization.v2": "Instruct: Retrieve semantically similar text\nQuery:",
    "Touche2020": "Instruct: Given a question, retrieve detailed and persuasive arguments that answer the question\nQuery: ",
    "Touche2020Retrieval.v3": "Instruct: Given a question, retrieve detailed and persuasive arguments that answer the question\nQuery: ",
    "ToxicConversationsClassification": "Instruct: Classify the given comments as either toxic or not toxic\nQuery:",
    "TRECCOVID": "Instruct: Given a COVID-19 question, retrieve relevant scientific literature\nQuery: ",
    "TweetSentimentExtractionClassification": "Instruct: Classify the sentiment of a given tweet as either positive, negative, or neutral\nQuery:",
    "TwentyNewsgroupsClustering.v2": "Instruct: Identify the topic or theme of the given news articles\nQuery:",
    "TwitterSemEval2015": "Instruct: Retrieve tweets that are semantically similar to the given tweet\nQuery:",
    "TwitterURLCorpus": "Instruct: Retrieve tweets that are semantically similar to the given tweet\nQuery:",
}

opensearch_neural_embed_v1 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=True,
        prompts_dict=opensearch_task_prompts,
        max_seq_length=8192,
        model_kwargs={"torch_dtype": "bfloat16"},
    ),
    name="opensearch-project/opensearch-neural-embed-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="502ce8e51dcc166a80968cbb838cb7b1ed1d96f8",
    release_date="2026-11-06",
    n_parameters=597_875_712,
    n_embedding_parameters=155_309_056,
    memory_usage_mb=2281,
    embed_dim=2048,
    max_tokens=8192,
    license="apache-2.0",
    reference="https://huggingface.co/opensearch-project/opensearch-neural-embed-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="Qwen/Qwen3-Embedding-0.6B",
    superseded_by=None,
)
