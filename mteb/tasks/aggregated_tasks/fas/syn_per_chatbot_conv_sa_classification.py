from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.classification import (
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

task_list_cqa = [
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
