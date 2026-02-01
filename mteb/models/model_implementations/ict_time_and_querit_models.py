from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


def instruction_template(
    instruction: str | dict, prompt_type: PromptType | None = None
) -> str:
    """Format instruction for the model."""
    if isinstance(instruction, dict):
        instruction = instruction.get(prompt_type.value if prompt_type else "", "")
    elif prompt_type == PromptType.document:
        return ""

    if not instruction:
        return ""
    return f"Instruct: {instruction}\nQuery:"


multilingual_langs = [
    "deu-Latn",
    "ita-Latn",
    "ara-Arab",
    "fas-Arab",
    "fra-Latn",
    "hin-Deva",
    "spa-Latn",
    "zho-Hans",
    "ben-Beng",
    "eng-Latn",
    "fin-Latn",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
]

training_data = [
    "FEVER",
    "DuRetrieval",
    "HotpotQA",
    "MSMARCO",
    "T2Retrieval",
    "NQ",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "ImdbClassification",
    "MTOPDomainClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

boom_4b_instructions = {
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual.",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment.",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category.",
    "Banking77Classification": "Given a online banking query, find the corresponding intents.",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise.",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset.",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation.",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic.",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
    "TNews": "Classify the fine-grained category of the given news title.",
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim.",
    "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve documents that support or refute the claim.",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia.",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim.",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim.",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question.",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question.",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question.",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query.",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question.",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question.",
}
# How the template actually renders each one at inference time:
#   instruction_template(boom_4b_instructions["Banking77Classification"], PromptType.query)
#   -> "Instruct: Given a online banking query, find the corresponding intents.\nQuery:"

boom_4b_v1 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
    ),
    name="ICT-TIME-and-Querit/BOOM_4B_v1",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    adapted_from="Qwen/Qwen3-4B",
    revision="33fb345468120e37c81eed2369aefe08b8f8222b",
    release_date="2026-01-31",
    n_parameters=4021774336,
    n_embedding_parameters=None,
    memory_usage_mb=7671,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/ICT-TIME-and-Querit/BOOM_4B_v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
