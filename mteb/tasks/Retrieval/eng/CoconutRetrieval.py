from __future__ import annotations
import logging

from mteb.abstasks.TaskMetadata import TaskMetadata


from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)


class CoconutRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CoconutRetrieval",
        dataset={
            "path": "BASF-We-Create-Chemistry/CoconutRetrieval",
            "revision": "4c23111a06ff9162dc8521dfe8096c544ab9548b",
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
