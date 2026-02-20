from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CTKFactsNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CTKFactsNLI",
        dataset={
            "path": "mteb/CTKFactsNLI",
            "revision": "da834e176f18e4a4adc4cbc9ad9720c14168a4bb",
        },
        description="Czech Natural Language Inference dataset of around 3K evidence-claim pairs labelled with SUPPORTS, REFUTES or NOT ENOUGH INFO veracity labels. Extracted from a round of fact-checking experiments.",
        reference="https://arxiv.org/abs/2201.11115",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["ces-Latn"],
        main_score="max_ap",
        date=("2020-09-01", "2021-08-31"),  # academic year 2020/2021
        domains=["News", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{ullrich2023csfever,
  author = {Ullrich, Herbert and Drchal, Jan and R{\\`y}par, Martin and Vincourov{\\'a}, Hana and Moravec, V{\\'a}clav},
  journal = {Language Resources and Evaluation},
  number = {4},
  pages = {1571--1605},
  publisher = {Springer},
  title = {CsFEVER and CTKFacts: acquiring Czech data for fact verification},
  volume = {57},
  year = {2023},
}
""",  # after removing label 1=NOT ENOUGH INFO
    )
