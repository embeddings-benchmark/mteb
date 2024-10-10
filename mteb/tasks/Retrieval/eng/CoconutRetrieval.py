from __future__ import annotations
import logging

from mteb.abstasks.TaskMetadata import TaskMetadata


from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)


class CoconutRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CoconutRetrieval",
        dataset={
            "path": "BASF-We-Create-Chemistry/SmallCoconutRetrieval",
            "revision": "831d292c3959eae59e4f89b8758738feee97d6cf",
        },
        description="COCONUT: the COlleCtion of Open NatUral producTs",
        reference="https://coconut.naturalproducts.net/",
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""""",
        descriptive_stats={}
    )
