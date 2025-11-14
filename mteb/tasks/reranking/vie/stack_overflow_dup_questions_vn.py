from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class StackOverflowDupQuestionsVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StackOverflowDupQuestions-VN",
        description="A translated dataset from Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        dataset={
            "path": "mteb/StackOverflowDupQuestions-VN",
            "revision": "d8b31731385a16007cef386fd695ea3f2243e5b4",
        },
        type="Reranking",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="map_at_1000",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Scientific Reranking"],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["StackOverflowDupQuestions"],
    )
