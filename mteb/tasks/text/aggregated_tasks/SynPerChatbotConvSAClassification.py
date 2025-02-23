from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.Classification import (
    SynPerChatbotConvSAAnger,
    SynPerChatbotConvSAFear,
    SynPerChatbotConvSAFriendship,
    SynPerChatbotConvSAHappiness,
    SynPerChatbotConvSAJealousy,
    SynPerChatbotConvSALove,
    SynPerChatbotConvSASadness,
    SynPerChatbotConvSASatisfaction,
    SynPerChatbotConvSASurprise,
)

task_list_cqa: list[AbsTask] = [
    SynPerChatbotConvSAAnger(),
    SynPerChatbotConvSASatisfaction(),
    SynPerChatbotConvSAFriendship(),
    SynPerChatbotConvSAFear(),
    SynPerChatbotConvSAJealousy(),
    SynPerChatbotConvSASurprise(),
    SynPerChatbotConvSALove(),
    SynPerChatbotConvSASadness(),
    SynPerChatbotConvSAHappiness(),
]


class SynPerChatbotConvSAClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="SynPerChatbotConvSAClassification",
        description="SynPerChatbotConvSAClassification",
        reference="",
        tasks=task_list_cqa,
        main_score="accuracy",
        type="Classification",
        eval_splits=["test"],
        bibtex_citation=""" """,
    )
