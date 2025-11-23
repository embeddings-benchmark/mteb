from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class OpenTenderRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OpenTenderRetrieval",
        description="This dataset contains Belgian and Dutch tender calls from OpenTender in Dutch",
        reference="https://arxiv.org/abs/2509.12340",
        dataset={
            "path": "clips/mteb-nl-opentender-ret",
            "revision": "83eec1aa9c58f1dc8acfac015f653a9c25bda3f4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2009-11-01", "2010-01-01"),
        domains=["Government", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{banar2025mtebnle5nlembeddingbenchmark,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
  eprint = {2509.12340},
  primaryclass = {cs.CL},
  title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
  url = {https://arxiv.org/abs/2509.12340},
  year = {2025},
}
""",
        prompt={
            "query": "Gegeven een titel, haal de aanbestedingsbeschrijving op die het beste bij de titel past"
        },
    )
