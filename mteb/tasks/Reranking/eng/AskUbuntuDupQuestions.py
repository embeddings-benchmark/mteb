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
        date=None,
        domains=["Programming", "Web"],
        task_subtypes=None,
        license=None,
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve duplicate questions from AskUbuntu forum",
        bibtex_citation=r"""
@article{wang-2021-TSDAE,
  author = {Wang, Kexin and Reimers, Nils and  Gurevych, Iryna},
  journal = {arXiv preprint arXiv:2104.06979},
  month = {4},
  title = {TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning},
  url = {https://arxiv.org/abs/2104.06979},
  year = {2021},
}
""",
    )
