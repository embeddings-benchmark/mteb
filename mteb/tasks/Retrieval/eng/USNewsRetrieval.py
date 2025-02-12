from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class USNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="USNewsRetrieval",
        description="A dataset comprising finance news articles, each paired with its corresponding headline and stock ticker symbol.",
        reference="https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles",
        dataset={
            "path": "FinanceMTEB/USnews",
            "revision": "dda970333494c509262e91d2b44d43430e985b3c",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )
