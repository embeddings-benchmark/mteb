from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class QuoraPLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quora-PL",
        description="QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "mteb/Quora-PL",
            "revision": "be9af3403db0afb13aa0837a047271b6eb073c58",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{wojtasik2024beirpl,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  year = {2024},
}
""",
        adapted_from=["QuoraRetrieval"],
    )


class QuoraPLRetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quora-PLHardNegatives",
        description="QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "mteb/Quora-PLHardNegatives",
            "revision": "f9135f0b4610db2225a957c1dd68960b5c572270",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{wojtasik2024beirpl,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  year = {2024},
}
""",
        adapted_from=["QuoraRetrieval"],
    )
