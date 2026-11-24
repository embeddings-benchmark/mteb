from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SNLRetrieval",
        dataset={
            "path": "mteb/SNLRetrieval",
            "revision": "6c0b6700612eba05b9bed0b858dcceda1ded0174",
        },
        description="Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.",
        reference="https://huggingface.co/datasets/mteb/SNLRetrieval",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",  # version assumed (not specified beforehand)
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
        prompt={"query": "Given a lexicon headline in Norwegian, retrieve its article"},
        task_subtypes=["Article retrieval"],
    )
