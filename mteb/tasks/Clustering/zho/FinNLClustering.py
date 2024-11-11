from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinNLClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FinNLClustering",
        description="Financial news categorization dataset. For the given financial news, the model is needed to classify it with multiple labels into fifteen possible categories, the categories include company, industry, broad market, China, foreign, international, economy, policy, politics, futures, bonds, real estate, foreign exchange, virtual currency, new crowns, energy, and other.",
        reference="https://arxiv.org/abs/2302.09432",
        dataset={
            "path": "FinanceMTEB/FinNL",
            "revision": "e88f3020ce6bc8e67b1800fe20b071d58ddbb468",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
    )
