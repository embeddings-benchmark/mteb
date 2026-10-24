from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class KoVidore2CybersecurityRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2CybersecurityRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Cybersecurity, is a corpus of technical reports on cyber threat trends and security incident responses in Korea, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-cybersecurity-mteb",
            "revision": "577d7c45f79d8eb4e7584db3990f91daa7e47956",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{choi-etal-2026-kovidore,
  address = {San Diego, USA},
  author = {Choi, Yongbin  and
Song, Yongwoo  and
Sung, Mujeen},
  booktitle = {Proceedings of the 2nd Workshop on Multimodal Augmented Generation via Multimodal Retrieval ({MAGM}a{R} 2026)},
  doi = {10.18653/v1/2026.magmar-main.11},
  editor = {Murray, Kenton  and
Kriz, Reno},
  isbn = {979-8-89176-425-5},
  month = jul,
  pages = {54--80},
  publisher = {Association for Computational Linguistics},
  title = {{K}o{V}i{D}o{R}e: A Benchmark for {K}orean Visual Document Retrieval},
  url = {https://aclanthology.org/2026.magmar-main.11/},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class KoVidore2EconomicRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2EconomicRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Economic trends, is a corpus of periodic reports on major economic indicators in Korea, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-economic-mteb",
            "revision": "0189c26211290a902cd9d41a0db932808a54c0a8",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{choi-etal-2026-kovidore,
  address = {San Diego, USA},
  author = {Choi, Yongbin  and
Song, Yongwoo  and
Sung, Mujeen},
  booktitle = {Proceedings of the 2nd Workshop on Multimodal Augmented Generation via Multimodal Retrieval ({MAGM}a{R} 2026)},
  doi = {10.18653/v1/2026.magmar-main.11},
  editor = {Murray, Kenton  and
Kriz, Reno},
  isbn = {979-8-89176-425-5},
  month = jul,
  pages = {54--80},
  publisher = {Association for Computational Linguistics},
  title = {{K}o{V}i{D}o{R}e: A Benchmark for {K}orean Visual Document Retrieval},
  url = {https://aclanthology.org/2026.magmar-main.11/},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class KoVidore2EnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2EnergyRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Energy, is a corpus of reports on energy market trends, policy planning, and industry statistics, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-energy-mteb",
            "revision": "8c09a3d22b1fa3a7f5e815e9521da9b048754211",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{choi-etal-2026-kovidore,
  address = {San Diego, USA},
  author = {Choi, Yongbin  and
Song, Yongwoo  and
Sung, Mujeen},
  booktitle = {Proceedings of the 2nd Workshop on Multimodal Augmented Generation via Multimodal Retrieval ({MAGM}a{R} 2026)},
  doi = {10.18653/v1/2026.magmar-main.11},
  editor = {Murray, Kenton  and
Kriz, Reno},
  isbn = {979-8-89176-425-5},
  month = jul,
  pages = {54--80},
  publisher = {Association for Computational Linguistics},
  title = {{K}o{V}i{D}o{R}e: A Benchmark for {K}orean Visual Document Retrieval},
  url = {https://aclanthology.org/2026.magmar-main.11/},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class KoVidore2HrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoVidore2HrRetrieval",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports on workforce outlook and employment policy in korea, intended for complex-document understanding tasks.",
        reference="https://github.com/whybe-choi/kovidore-data-generator",
        dataset={
            "path": "whybe-choi/kovidore-v2-hr-mteb",
            "revision": "d9432c782a9a3e2eed064f6fac08b4c967d92b99",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-12-21", "2026-01-06"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{choi-etal-2026-kovidore,
  address = {San Diego, USA},
  author = {Choi, Yongbin  and
Song, Yongwoo  and
Sung, Mujeen},
  booktitle = {Proceedings of the 2nd Workshop on Multimodal Augmented Generation via Multimodal Retrieval ({MAGM}a{R} 2026)},
  doi = {10.18653/v1/2026.magmar-main.11},
  editor = {Murray, Kenton  and
Kriz, Reno},
  isbn = {979-8-89176-425-5},
  month = jul,
  pages = {54--80},
  publisher = {Association for Computational Linguistics},
  title = {{K}o{V}i{D}o{R}e: A Benchmark for {K}orean Visual Document Retrieval},
  url = {https://aclanthology.org/2026.magmar-main.11/},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )
