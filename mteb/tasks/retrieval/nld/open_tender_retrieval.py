from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class OpenTenderRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OpenTenderRetrieval",
        description="This dataset contains Belgian and Dutch tender calls from OpenTender in Dutch",
        reference="https://aclanthology.org/2026.findings-acl.1236/",
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
@inproceedings{banar2025mtebnle5nlembeddingbenchmark,
  address = {San Diego, California, United States},
  author = {Banar, Nikolay and Lotfi, Ehsan and Van Nooten, Jens and Arhiliuc, Cristina and Kliocaite, Marija and Daelemans, Walter},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026},
  doi = {10.18653/v1/2026.findings-acl.1236},
  editor = {Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David},
  isbn = {979-8-89176-395-1},
  month = {jul},
  pages = {24684--24709},
  publisher = {Association for Computational Linguistics},
  title = {{MTEB}-{NL} and {E}5-{NL}: Embedding Benchmark and Models for {D}utch},
  url = {https://aclanthology.org/2026.findings-acl.1236/},
  year = {2026},
}
""",
        prompt={
            "query": "Gegeven een titel, haal de aanbestedingsbeschrijving op die het beste bij de titel past"
        },
    )
