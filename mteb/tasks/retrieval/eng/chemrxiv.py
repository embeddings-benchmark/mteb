from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ChemRxivRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChemRxivRetrieval",
        dataset={
            "path": "BASF-AI/ChemRxivRetrieval",
            "revision": "5377aa18f309ec440ff6325a4c2cd3362c2cb8d7",
        },
        description="A retrieval task based on ChemRxiv papers where queries are LLM-synthesized to match specific paragraphs.",
        reference="https://arxiv.org/abs/2508.01643",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-05-01"),
        domains=["Chemistry"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{kasmaee2025chembed,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Astaraki, Mahdi and Saloot, Mohammad Arshi and Sherck, Nicholas and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2508.01643},
  title = {Chembed: Enhancing chemical literature search through domain-specific text embeddings},
  year = {2025},
}""",
    )
