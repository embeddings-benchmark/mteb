from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BIRCORelicReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BIRCO-Relic",
        description=(
            "Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries which are excerpts from literary analyses "
            "with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. "
            "The objective is to retrieve the passage that best completes the literary analysis."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-Relic-Test",
            "revision": "3d593fde907860102a282b4a6ea7fb1e1f1f83dd",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Fiction"],  # Valid domain
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Given a literary analysis with a missing quotation (marked as [masked sentence(s)]), retrieve the passage that best completes the analysis.",
        bibtex_citation="""@misc{wang2024bircobenchmarkinformationretrieval,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives}, 
            author={Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2402.14151}, 
        }""",
    )
