from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_nf_corpus_metadata = dict(
    dataset={
        "path": "clips/beir-nl-nfcorpus",
        "revision": "942953e674fd0f619ff89897abb806dc3df5dd39",
    },
    reference="https://huggingface.co/datasets/clips/beir-nl-nfcorpus",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["nld-Latn"],
    main_score="ndcg_at_10",
    date=("2016-03-01", "2016-03-01"),  # best guess: based on publication date
    domains=["Medical", "Academic", "Written"],
    task_subtypes=[],
    license="cc-by-4.0",
    annotations_creators="derived",
    dialect=[],
    sample_creation="machine-translated and verified",  # manually checked a small subset
    bibtex_citation=r"""
@inproceedings{banar2024beirnlzeroshotinformationretrieval,
  address = {Abu Dhabi, UAE},
  archiveprefix = {arXiv},
  author = {Lotfi, Ehsan and Banar, Nikolay and Daelemans, Walter},
  booktitle = {Proceedings of the 18th Workshop on Building and Using Comparable Corpora ({BUCC})},
  editor = {Sharoff, Serge and Terryn, Ayla Rigouts and Zweigenbaum, Pierre and Rapp, Reinhard},
  eprint = {2412.08329},
  month = {jan},
  pages = {36--45},
  publisher = {Association for Computational Linguistics},
  title = {{BEIR}-{NL}: Zero-shot Information Retrieval Benchmark for the {D}utch Language},
  url = {https://aclanthology.org/2025.bucc-1.5/},
  year = {2025},
}
""",
)


class NFCorpusNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus-NL",
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is "
        "a Dutch translation.",
        adapted_from=["NFCorpus"],
        **_nf_corpus_metadata,
    )


class NFCorpusNLv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus-NL.v2",
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is "
        "a Dutch translation. This version adds a Dutch prompt to the dataset.",
        adapted_from=["NFCorpus-NL"],
        prompt={
            "query": "Gegeven een vraag, haal relevante documenten op die de vraag het beste beantwoorden"
        },
        **_nf_corpus_metadata,
    )
