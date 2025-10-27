from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_ria_news_metadata = dict(
    reference="https://arxiv.org/abs/1901.07786",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["rus-Cyrl"],
    main_score="ndcg_at_10",
    date=("2010-01-01", "2014-12-31"),
    domains=["News", "Written"],
    task_subtypes=["Article retrieval"],
    license="cc-by-nc-nd-4.0",
    annotations_creators="derived",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@inproceedings{gavrilov2018self,
  author = {Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
  booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
  title = {Self-Attentive Model for Headline Generation},
  year = {2019},
}
""",
)


class RiaNewsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RiaNewsRetrieval",
        dataset={
            "path": "ai-forever/ria-news-retrieval",
            "revision": "82374b0bbacda6114f39ff9c5b925fa1512ca5d7",
        },
        description="News article retrieval by headline. Based on Rossiya Segodnya dataset.",
        prompt={"query": "Given a news title, retrieve relevant news article"},
        **_ria_news_metadata,
    )


class RiaNewsRetrievalHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RiaNewsRetrievalHardNegatives",
        dataset={
            "path": "mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2",
            "revision": "d42860a6c15f0a2c4485bda10c6e5b641fdfe479",
        },
        description="News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        adapted_from=["RiaNewsRetrieval"],
        superseded_by="RiaNewsRetrievalHardNegatives.v2",
        **_ria_news_metadata,
    )


class RiaNewsRetrievalHardNegativesV2(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RiaNewsRetrievalHardNegatives.v2",
        dataset={
            "path": "mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2",
            "revision": "d42860a6c15f0a2c4485bda10c6e5b641fdfe479",
        },
        description=(
            "News article retrieval by headline. Based on Rossiya Segodnya dataset. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)"
        ),
        adapted_from=["RiaNewsRetrieval"],
        prompt={"query": "Given a news title, retrieve relevant news article"},
        **_ria_news_metadata,
    )
