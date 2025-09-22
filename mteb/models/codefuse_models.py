from __future__ import annotations

from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta



from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper

training_datasets={
    "MSMARCO": ["train"],
    "ArguAna": ["train"],
    "SNLI": ["train"],
    "MNLI": ["train"],
    "ANLI": ["train"],
    "PAQ": ["train"],
    "SQuAD": ["train"],
    "StackExchange": ["train"],
    "MSMARCO": ["train"],
    "NQ": ["train"],
    "HotpotQA": ["train"],
    "FEVER": ["train"],
    "ELI5": ["train"],
    "FiQA2018": ["train"],
    "BioASQ": ["train"],
    "NFCorpus": ["train"],
    "MIRACLRetrieval": ["train"],
    "MrTidyRetrieval": ["train"],
    "SciFact": ["train"],
    "TriviaQA": ["train"],
    "COLIEE": ["train"],
    "PubMedQA": ["train"],
    "S2ORC": ["train"],
    "AmazonQA": ["train"],
    "SPECTER": ["train"],
    "XSum": ["train"],
    "CNNDM": ["train"],
    "SentenceCompression": ["train"],
    "StackExchangeDupQuestions": ["train"],
    "QQP": ["train"],
    "StackOverflowDupQuestions": ["train"],
    "STS12": ["train"],
    "STS22": ["train"],
    "STSBenchmark": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "AmazonPolarityClassification": ["train"],
    "ImdbClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "CoLA": ["train"],
    "AmazonReviewClassification": ["train"],
    "Banking77Classification": ["train"],
    "EmotionClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "MTOPDomainClassification": ["train"],
    "MassiveScenarioClassification": ["train"],
    "MassiveIntentClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "ArxivClusteringP2P": ["train"],
    "ArxivClusteringS2S": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "BiorxivClusteringS2S": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "MedrxivClusteringS2S": ["train"],
    "RedditClustering": ["train"],
    "RedditClusteringP2P": ["train"],
    "StackExchangeClustering": ["train"],
    "StackExchangeClusteringP2P": ["train"],
    "TwentyNewsgroupsClustering": ["train"],
}

INSTRUCTIONS = {
    'AmazonCounterfactualClassification': "Classify a given Amazon customer review text as either counterfactual or not counterfactual.",
    'Banking77Classification': "Given an online banking query, find the corresponding intents.",
    'ImdbClassification': "Classify the sentiment expressed in the given movie review text from the IMDB dataset.",
    'MTOPDomainClassification': "Classify the intent domain of the given utterance in task-oriented conversation.",
    'MassiveIntentClassification': "Given a user utterance as query, find the user intents.",
    'MassiveScenarioClassification': "Given a user utterance as query, find the user scenarios.",
    'ToxicConversationsClassification': "Classify the given comments as either toxic or not toxic.",
    'TweetSentimentExtractionClassification': "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    'ArXivHierarchicalClusteringP2P': "Identify the main and secondary category of arXiv papers based on the titles and abstracts.",
    'ArXivHierarchicalClusteringS2S': "Identify the main and secondary category of arXiv papers based on the titles.",
    'BiorxivClusteringP2P.v2': "Identify the main category of bioRxiv papers based on the titles and abstracts.",
    'MedrxivClusteringP2P.v2': "Identify the main category of medRxiv papers based on the titles and abstracts.",
    'MedrxivClusteringS2S.v2': "Identify the main category of medRxiv papers based on the titles.",
    'StackExchangeClustering.v2': "Identify the topic or theme of StackExchange posts based on the titles.",
    'StackExchangeClusteringP2P.v2': "Identify the topic or theme of StackExchange posts based on the given paragraphs.",
    'TwentyNewsgroupsClustering.v2': "Identify the topic or theme of the given news articles.",
    'SprintDuplicateQuestions': "Retrieve duplicate questions from Sprint forum.",
    'TwitterSemEval2015': "Retrieve tweets that are semantically similar to the given tweet.",
    'TwitterURLCorpus': "Retrieve tweets that are semantically similar to the given tweet.",
    'AskUbuntuDupQuestions': "Retrieve duplicate questions from AskUbuntu forum.",
    'MindSmallReranking': "Retrieve relevant news articles based on user browsing history.",
    'ArguAna': "Given a claim, find documents that refute the claim.",
    'CQADupstackGamingRetrieval': "Given a question, retrieve questions that are semantically equivalent.",
    'CQADupstackUnixRetrieval': "Given a question, retrieve questions that are semantically equivalent.",
    'ClimateFEVERHardNegatives': "Given a claim about climate change, retrieve documents that support or refute the claim.",
    'FEVERHardNegatives': "Given a claim, retrieve documents that support or refute the claim.",
    'FiQA2018': "Given a financial question, retrieve passages that answer the question.",
    'HotpotQAHardNegatives': "Given a multi-hop question, retrieve passages that answer the question.",
    'SCIDOCS': "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.",
    'TRECCOVID': "Given a query on COVID-19, retrieve documents that answer the query.",
    'Touche2020Retrieval.v3': "Given a question, retrieve passages that answer the question.",
    'BIOSSES': "Retrieve semantically similar text.",
    'SICK-R': "Retrieve semantically similar text.",
    'STS12': "Retrieve semantically similar text.",
    'STS13': "Retrieve semantically similar text.",
    'STS14': "Retrieve semantically similar text.",
    'STS15': "Retrieve semantically similar text.",
    'STS17': "Retrieve semantically similar text.",
    'STS22.v2': "Retrieve semantically similar text.",
    'STSBenchmark': "Retrieve semantically similar text.",
    'SummEvalSummarization.v2': "Given a news summary, retrieve other semantically similar summaries.",
}

def apply_instruct(instruct):
    return f"Instruct: {instruct}\nQuery: "


class InstructSentenceTransformerWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str | Callable[[str], str] | None = None,
        max_seq_length: int | None = None,
        apply_instruction_to_passages: bool = True,
        padding_side: str | None = None,
        add_eos_token: bool = False,
        prompts_dict: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """Instruct Sentence Transformer Wrapper. Wrapper that passes instructions to the Sentence Transformer model.
        Applied for models like NV-Embed, gte-Qwen, e5-mistral, etc.

        Arguments:
            model_name: Model name of the sentence transformers model.
            revision: Revision of the sentence transformers model.
            instruction_template: Model template. Should contain the string '{instruction}'.
            max_seq_length: Maximum sequence length. If None, the maximum sequence length will be read from the model config.
            apply_instruction_to_passages: Whether to apply the instruction template to the passages.
            padding_side: Padding side. If None, the padding side will be read from the model config.
            add_eos_token: Whether to add the eos token to each input example.
            prompts_dict: Dictionary of task names to prompt names. If None, the prompts will be read from the model config.
            **kwargs: Kwargs for Sentence Transformer model.
        """
        if (
            isinstance(instruction_template, str)
            and "{instruction}" not in instruction_template
        ):
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )
        if instruction_template is None:
            print(
                "No instruction template provided. Instructions will be used as-is."
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.add_eos_token = add_eos_token
        self.prompts_dict = prompts_dict
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        if padding_side is not None:
            self.model.tokenizer.padding_side = padding_side

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.add_eos_token:
            sentences = [
                example + self.model.tokenizer.eos_token for example in sentences
            ]

        instruction = apply_instruct(INSTRUCTIONS[task_name]) if task_name in INSTRUCTIONS else self.get_task_instruction(task_name, prompt_type, self.prompts_dict)

        # to passage prompts won't be applied to passages
        if (
            not self.apply_instruction_to_passages
            and prompt_type == PromptType.document
        ):
            instruction = None
            print(
                f"No instruction used, because prompt type = {prompt_type.document}"
            )

        if instruction:
            print(f"Using instruction: '{instruction}' for task: '{task_name}'")

        embeddings = self.model.encode(
            sentences,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings



def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery: "



F2LLM_0B6 = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name="codefuse-ai/F2LLM-0.6B",
        revision="36416618b83d4bd84a8ca30c2ee01ed518f9f2e7",
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        add_eos_token=True,
        max_seq_length=8192,
    ),
    name="codefuse-ai/F2LLM-0.6B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="36416618b83d4bd84a8ca30c2ee01ed518f9f2e7",
    release_date="2025-09-18",
    n_parameters=595_776_512,
    memory_usage_mb=1137,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/codefuse-ai/F2LLM-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/codefuse-ai/F2LLM",
    training_datasets=training_datasets,
)

F2LLM_1B7 = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name="codefuse-ai/F2LLM-1.7B",
        revision="fdce0e09655f42cea26f7f66f5a70cd4507ea45c",
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        add_eos_token=True,
        max_seq_length=8192,
    ),
    name="codefuse-ai/F2LLM-1.7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="fdce0e09655f42cea26f7f66f5a70cd4507ea45c",
    release_date="2025-09-18",
    n_parameters=1_720_574_976,
    memory_usage_mb=3282,
    embed_dim=2560,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/codefuse-ai/F2LLM-1.7B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/codefuse-ai/F2LLM",
    training_datasets=training_datasets,
)

F2LLM_4B = ModelMeta(
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name="codefuse-ai/F2LLM-4B",
        revision="9fe95901ed2b6b59dd7673d6e93c9d76766a1e25",
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        add_eos_token=True,
        max_seq_length=8192,
    ),
    name="codefuse-ai/F2LLM-4B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="9fe95901ed2b6b59dd7673d6e93c9d76766a1e25",
    release_date="2025-09-18",
    n_parameters=4_021_774_336,
    memory_usage_mb=7672,
    embed_dim=2560,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/codefuse-ai/F2LLM-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/codefuse-ai/F2LLM",
    training_datasets=training_datasets,
)