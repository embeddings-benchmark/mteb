from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class STSES(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSES",
        dataset={
            "path": "mteb/STSES",
            "revision": "fe7158e70012087b5a692ce175226a0d213936ad",
        },
        description="Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)",
        reference="https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["spa-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Written"],
        task_subtypes=None,
        license="cc-by-4.0",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{agirre2014semeval,
  author = {Agirre, Eneko and Banea, Carmen and Cardie, Claire and Cer, Daniel M and Diab, Mona T and Gonzalez-Agirre, Aitor and Guo, Weiwei and Mihalcea, Rada and Rigau, German and Wiebe, Janyce},
  booktitle = {SemEval@ COLING},
  pages = {81--91},
  title = {SemEval-2014 Task 10: Multilingual Semantic Textual Similarity.},
  year = {2014},
}

@inproceedings{agirre2015semeval,
  author = {Agirre, Eneko and Banea, Carmen and Cardie, Claire and Cer, Daniel and Diab, Mona and Gonzalez-Agirre, Aitor and Guo, Weiwei and Lopez-Gazpio, Inigo and Maritxalar, Montse and Mihalcea, Rada and others},
  booktitle = {Proceedings of the 9th international workshop on semantic evaluation (SemEval 2015)},
  pages = {252--263},
  title = {Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability},
  year = {2015},
}
""",
    )

    min_score = 0
    max_score = 5
