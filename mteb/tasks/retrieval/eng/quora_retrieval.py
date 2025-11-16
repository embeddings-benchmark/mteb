from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_quora_metadata = dict(
    reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=None,
    domains=["Written", "Web", "Blog"],
    task_subtypes=["Question answering"],
    license="not specified",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@misc{quora-question-pairs,
  author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
  publisher = {Kaggle},
  title = {Quora Question Pairs},
  url = {https://kaggle.com/competitions/quora-question-pairs},
  year = {2017},
}
""",
)


class QuoraRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrieval",
        dataset={
            "path": "mteb/quora",
            "revision": "e4e08e0b7dbe3c8700f0daef558ff32256715259",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            + " question, find other (duplicate) questions."
        ),
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        **_quora_metadata,
    )


class QuoraRetrievalHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrievalHardNegatives",
        dataset={
            "path": "mteb/QuoraRetrieval_test_top_250_only_w_correct-v2",
            "revision": "907a33577e9506221d3ba20f5a851b7c3f8dc6d3",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            + " question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
        ),
        adapted_from=["QuoraRetrieval"],
        superseded_by="QuoraRetrievalHardNegatives.v2",
        **_quora_metadata,
    )


class QuoraRetrievalHardNegativesV2(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrievalHardNegatives.v2",
        dataset={
            "path": "mteb/QuoraRetrieval_test_top_250_only_w_correct-v2",
            "revision": "907a33577e9506221d3ba20f5a851b7c3f8dc6d3",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a "
            "question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)"
        ),
        adapted_from=["QuoraRetrieval"],
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        **_quora_metadata,
    )
