from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QuoraPLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quora-PL",
        description="QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "clarin-knext/quora-pl",
            "revision": "0be27e93455051e531182b85e85e425aba12e9d4",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
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
        bibtex_citation=""""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        adapted_from=["QuoraRetrieval"],
    )


class QuoraPLRetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quora-PLHardNegatives",
        description="QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "mteb/Quora_PL_test_top_250_only_w_correct-v2",
            "revision": "523ff30f3346cd9c36081c19fc6eaef0a2f8d53d",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
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
        bibtex_citation=""""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        adapted_from=["QuoraRetrieval"],
    )
