from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_sci_fact_nl_metadata = dict(
    dataset={
        "path": "clips/beir-nl-scifact",
        "revision": "856d8dfc294b138856bbf3042450e3782321e44e",
    },
    reference="https://huggingface.co/datasets/clips/beir-nl-scifact",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["nld-Latn"],
    main_score="ndcg_at_10",
    date=("2020-05-01", "2020-05-01"),  # best guess: based on submission date
    domains=["Academic", "Medical", "Written"],
    task_subtypes=[],
    license="cc-by-4.0",
    annotations_creators="derived",
    dialect=[],
    sample_creation="machine-translated and verified",  # manually checked a small subset
    bibtex_citation=r"""
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
)


class SciFactNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-NL",
        description="SciFactNL verifies scientific claims in Dutch using evidence from the research literature "
        "containing scientific paper abstracts.",
        adapted_from=["SciFact"],
        **_sci_fact_nl_metadata,
    )


class SciFactNLv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-NL.v2",
        description="SciFactNL verifies scientific claims in Dutch using evidence from the research literature "
        "containing scientific paper abstracts. This version adds a Dutch prompt to the dataset.",
        adapted_from=["SciFact-NL"],
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
        **_sci_fact_nl_metadata,
    )
