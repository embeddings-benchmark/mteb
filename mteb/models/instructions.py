"""This specifies the default instructions for tasks within MTEB. These are optional to use and some models might want to use their own instructions."""

from __future__ import annotations

import mteb

# This list is NOT comprehensive even for the tasks within MTEB
# TODO: We should probably move this prompt to the task object
TASKNAME2INSTRUCTIONS = {
    # Reranking and pair classification
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "VoyageMMarcoReranking": "Given a Japanese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    "TERRa": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    # Retrieval - 1st item is query instruction; 2nd is corpus instruction
    "TwitterHjerneRetrieval": (
        "Retrieve answers to questions asked in Danish tweets",
        "",
    ),
    "SwednRetrieval": (
        "Given a Swedish news headline retrieve summaries or news articles",
        "",
    ),
    "TV2Nordretrieval": (
        "Given a summary of a Danish news article retrieve the corresponding news article",
        "",
    ),
    "DanFEVER": (
        "Given a claim in Danish, retrieve documents that support the claim",
        "",
    ),
    "SNLRetrieval": ("Given a lexicon headline in Norwegian, retrieve its article", ""),
    "NorQuadRetrieval": (
        "Given a question in Norwegian, retrieve the answer from Wikipedia articles",
        "",
    ),
    "SweFaqRetrieval": ("Retrieve answers given questions in Swedish", ""),
    "ArguAna": ("Given a claim, find documents that refute the claim", ""),
    "ClimateFEVER": (
        "Given a claim about climate change, retrieve documents that support or refute the claim",
        "",
    ),
    "DBPedia": (
        "Given a query, retrieve relevant entity descriptions from DBPedia",
        "",
    ),
    "FEVER": ("Given a claim, retrieve documents that support or refute the claim", ""),
    "FiQA2018": (
        "Given a financial question, retrieve user replies that best answer the question",
        "",
    ),
    "HotpotQA": (
        "Given a multi-hop question, retrieve documents that can help answer the question",
        "",
    ),
    "MSMARCO": (
        "Given a web search query, retrieve relevant passages that answer the query",
        "",
    ),
    "NFCorpus": (
        "Given a question, retrieve relevant documents that best answer the question",
        "",
    ),
    "NQ": (
        "Given a question, retrieve Wikipedia passages that answer the question",
        "",
    ),
    "QuoraRetrieval": (
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "",
    ),
    "SCIDOCS": (
        "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
        "",
    ),
    "SciFact": (
        "Given a scientific claim, retrieve documents that support or refute the claim",
        "",
    ),
    "Touche2020": (
        "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "",
    ),
    "TRECCOVID": (
        "Given a query on COVID-19, retrieve documents that answer the query",
        "",
    ),
    "T2Retrieval": (
        "Given a Chinese search query, retrieve web passages that answer the question",
        "",
    ),
    "MMarcoRetrieval": (
        "Given a web search query, retrieve relevant passages that answer the query",
        "",
    ),
    "DuRetrieval": (
        "Given a Chinese search query, retrieve web passages that answer the question",
        "",
    ),
    "CovidRetrieval": (
        "Given a question on COVID-19, retrieve news articles that answer the question",
        "",
    ),
    "CmedqaRetrieval": (
        "Given a Chinese community medical question, retrieve replies that best answer the question",
        "",
    ),
    "EcomRetrieval": (
        "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
        "",
    ),
    "MedicalRetrieval": (
        "Given a medical question, retrieve user replies that best answer the question",
        "",
    ),
    "VideoRetrieval": (
        "Given a video search query, retrieve the titles of relevant videos",
        "",
    ),
    "ARCChallenge": (
        "Retrieve the answer to the question.",
        "",
    ),
    "AlphaNLI": (
        "Given the following start and end of a story, retrieve a possible reason that leads to the end.",
        "",
    ),
    "HellaSwag": (
        "Given the following unfinished context, retrieve the most plausible ending to finish it.",
        "",
    ),
    "PIQA": (
        "Given the following goal, retrieve a possible solution.",
        "",
    ),
    "Quail": (
        "Given the following context and question, retrieve the correct answer.",
        "",
    ),
    "SIQA": (
        "Given the following context and question, retrieve the correct answer.",
        "",
    ),
    "RARbCode": (
        "Retrieve the answer for the following coding problem.",
        "",
    ),
    "RARbMath": (
        "Retrieve the answer for the following math problem.",
        "",
    ),
    "SpartQA": (
        "Given the following spatial reasoning question, retrieve the right answer.",
        "",
    ),
    "TempReasonL1": (
        "Given the following question about time, retrieve the correct answer.",
        "",
    ),
    "TempReasonL2Pure": (
        "Given the following question, retrieve the correct answer.",
        "",
    ),
    "TempReasonL2Fact": (
        "Given the following question and facts, retrieve the correct answer.",
        "",
    ),
    "TempReasonL2Context": (
        "Given the following question, facts and contexts, retrieve the correct answer.",
        "",
    ),
    "TempReasonL3Pure": (
        "Given the following question, retrieve the correct answer.",
        "",
    ),
    "TempReasonL3Fact": (
        "Given the following question and facts, retrieve the correct answer.",
        "",
    ),
    "TempReasonL3Context": (
        "Given the following question, facts and contexts, retrieve the correct answer.",
        "",
    ),
    "WinoGrande": (
        "Given the following sentence, retrieve an appropriate answer to fill in the missing underscored part.",
        "",
    ),
    "RuBQRetrieval": (
        "Given a question, retrieve Wikipedia passages that answer the question",
        "",
    ),
    "MIRACLRetrieval": (
        "Given a question, retrieve Wikipedia passages that answer the question",
        "",
    ),
    "RiaNewsRetrieval": ("Given a news title, retrieve relevant news article", ""),
}


def task_to_instruction(task_name: str, is_query: bool = True) -> str:
    if task_name in TASKNAME2INSTRUCTIONS:
        if isinstance(TASKNAME2INSTRUCTIONS[task_name], tuple):
            return (
                TASKNAME2INSTRUCTIONS[task_name][0]
                if is_query
                else TASKNAME2INSTRUCTIONS[task_name][1]
            )
        return TASKNAME2INSTRUCTIONS[task_name]

    meta = mteb.get_task(task_name).metadata
    return ""
