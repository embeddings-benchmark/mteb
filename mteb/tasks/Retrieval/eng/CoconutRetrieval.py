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
            "revision": "fdb30de349565a819d481f1eb7ef6f851ff150fc",
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
