from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_scidocsnl_metadata = dict(
    dataset={
        "path": "clips/beir-nl-scidocs",
        "revision": "4e018aa220029f9d1bd5a31de3650e322e32ea38",
    },
    reference="https://huggingface.co/datasets/clips/beir-nl-scidocs",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["nld-Latn"],
    main_score="ndcg_at_10",
    date=("2020-05-01", "2020-05-01"),  # best guess: based on submission date
    domains=["Academic", "Written", "Non-fiction"],
    task_subtypes=[],
    license="cc-by-sa-4.0",
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


class SCIDOCSNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-NL",
        description="SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from "
        "citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch "
        "translation.",
        adapted_from=["SCIDOCS"],
        **_scidocsnl_metadata,
    )


class SCIDOCSNLv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-NL.v2",
        description="SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from "
        "citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch "
        "translation. This version adds a Dutch prompt to the dataset.",
        adapted_from=["SCIDOCS-NL"],
        **_scidocsnl_metadata,
        prompt={
            "query": "Gegeven de titel van een wetenschappelijk artikel, haal de abstracts op van artikelen die door het gegeven artikel worden geciteerd"
        },
    )
