from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class BIRCOClinicalTrialReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="BIRCO-ClinicalTrial",
        description=(
            "Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. "
            "Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-ClinicalTrial-Test",
            "revision": "023e7271b234db5cfdbc7e385a21afc6b0081a72",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Medical"],  # Valid domain (Medical)
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Given a patient case report, retrieve the clinical trial description that best matches the patient's eligibility criteria.",
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

    def evaluate(self, model, **kwargs):
        # Calculate metadata metrics first
        self.calculate_metadata_metrics()
        # Existing evaluation code
        return super().evaluate(model, **kwargs)