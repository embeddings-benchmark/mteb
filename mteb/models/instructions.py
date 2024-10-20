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
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "VoyageMMarcoReranking": "Given a Japanese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
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
