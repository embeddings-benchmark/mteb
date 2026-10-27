from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class AskUbuntuDupQuestions(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AskUbuntuDupQuestions",
        description="AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar",
        reference="https://github.com/taolei87/askubuntu",
        dataset={
            "path": "mteb/AskUbuntuDupQuestions",
            "revision": "c5691e3c48741d5f83b5cc8e630653d7a8cfc048",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2014-01-01", "2021-12-31"),
        domains=["Programming", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve duplicate questions from AskUbuntu forum",
        bibtex_citation=r"""
@inproceedings{wang-2021-TSDAE,
  address = {Punta Cana, Dominican Republic},
  author = {Wang, Kexin and Reimers, Nils and Gurevych, Iryna},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2021},
  doi = {10.18653/v1/2021.findings-emnlp.59},
  editor = {Moens, Marie-Francine and Huang, Xuanjing and Specia, Lucia and Yih, Scott Wen-tau},
  month = {nov},
  pages = {671--688},
  publisher = {Association for Computational Linguistics},
  title = {{TSDAE}: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning},
  url = {https://aclanthology.org/2021.findings-emnlp.59/},
  year = {2021},
}
""",
    )
