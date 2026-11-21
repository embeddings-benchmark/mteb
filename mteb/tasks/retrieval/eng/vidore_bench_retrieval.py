from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VidoreArxivQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreArxivQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreArxivQARetrieval",
            "revision": "e1147f6f6c909d3e025844964820a6a4fe44a942",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )


class VidoreDocVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreDocVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreDocVQARetrieval",
            "revision": "7237bd4b327affd856b7f2e8a5ac9a35192e1254",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )


class VidoreInfoVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreInfoVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreInfoVQARetrieval",
            "revision": "f7393aded44587fdb7e6b2f36231f295873e9985",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )


class VidoreTabfquadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreTabfquadRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreTabfquadRetrieval",
            "revision": "44b120147bfb0be64a157e1c0edcfbc0378a977e",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )


class VidoreTatdqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreTatdqaRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreTatdqaRetrieval",
            "revision": "ac75d58db1dbf719d83b664a40dd10b7a9f3f12d",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )


class VidoreShiftProjectRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreShiftProjectRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreShiftProjectRetrieval",
            "revision": "6321aae326ff5822dfc23bb0ff4922f825257f8a",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )


class VidoreSyntheticDocQAAIRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAAIRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreSyntheticDocQAAIRetrieval",
            "revision": "93e4ba30490e8bc0dbbc7e0506e77e016dbf33cb",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )


class VidoreSyntheticDocQAEnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAEnergyRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreSyntheticDocQAEnergyRetrieval",
            "revision": "e0ccbf2c7236bf0ac1a5e91ca612a90403ac2489",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )


class VidoreSyntheticDocQAGovernmentReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAGovernmentReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "revision": "bbef2358af245a759ee2a4d612c0d1b30628fec6",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )


class VidoreSyntheticDocQAHealthcareIndustryRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://openreview.net/forum?id=ogjBpZ8uSi",
        dataset={
            "path": "mteb/VidoreSyntheticDocQAHealthcareIndustryRetrieval",
            "revision": "f27352cd8dcc0814aab6a560d920207ffcd81ece",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{faysse2024colpali,
  archiveprefix = {arXiv},
  author = {Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  url = {https://openreview.net/forum?id=ogjBpZ8uSi},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )
