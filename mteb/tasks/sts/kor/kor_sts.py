from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class KorSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KorSTS",
        dataset={
            "path": "dkoterwa/kor-sts",
            "revision": "016f35f9b961daaaa7a352e927084e3da662ac1f",
        },
        description="Benchmark dataset for STS in Korean. Created by machine translation and human post editing of the STS-B dataset.",
        reference="https://aclanthology.org/2020.findings-emnlp.39/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2017-01-01"),  # rough approximates
        domains=["News", "Web"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and localized",
        bibtex_citation=r"""
@inproceedings{ham2020kornli,
  address = {Online},
  author = {Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2020},
  doi = {10.18653/v1/2020.findings-emnlp.39},
  editor = {Cohn, Trevor and He, Yulan and Liu, Yang},
  month = {nov},
  pages = {422--430},
  publisher = {Association for Computational Linguistics},
  title = {{K}or{NLI} and {K}or{STS}: New Benchmark Datasets for {K}orean Natural Language Understanding},
  url = {https://aclanthology.org/2020.findings-emnlp.39/},
  year = {2020},
}
""",
    )

    min_score = 0
    max_score = 5
