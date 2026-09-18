from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoDBPediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoDBPediaRetrieval",
        description="NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.",
        reference="https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia",
        dataset={
            "path": "mteb/NanoDBPediaRetrieval",
            "revision": "2d088f80d7a640eac0be5c66408d68c3f782baa1",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2015-01-01", "2015-12-31"],
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lehmann2015dbpedia,
  author = {Lehmann, Jens and et al.},
  journal = {Semantic Web},
  title = {DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia},
  year = {2015},
}
""",
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
        adapted_from=["DBPedia"],
    )
