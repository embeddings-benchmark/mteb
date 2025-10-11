from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BIRCOClinicalTrialReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BIRCO-ClinicalTrial",
        description=(
            "Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. "
            "Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-ClinicalTrial-Test",
            "revision": "ef7b3a7f08dd2f0f7bdf0cf23391f8a1a26ad477",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Medical"],  # Valid domain (Medical)
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Given a patient case report, retrieve the clinical trial description that best matches the patient's eligibility criteria.",
        bibtex_citation=r"""
@misc{wang2024bircobenchmarkinformationretrieval,
  archiveprefix = {arXiv},
  author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
  eprint = {2402.14151},
  primaryclass = {cs.IR},
  title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  url = {https://arxiv.org/abs/2402.14151},
  year = {2024},
}
""",
    )
