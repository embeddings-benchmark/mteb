from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_hotpot_qa_metadata = dict(
    reference="https://hotpotqa.github.io/",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=("2018-01-01", "2018-12-31"),  # best guess: based on publication date
    domains=["Web", "Written"],
    task_subtypes=["Question answering"],
    license="cc-by-sa-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@inproceedings{yang-etal-2018-hotpotqa,
  address = {Brussels, Belgium},
  author = {Yang, Zhilin  and
Qi, Peng  and
Zhang, Saizheng  and
Bengio, Yoshua  and
Cohen, William  and
Salakhutdinov, Ruslan  and
Manning, Christopher D.},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1259},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {2369--2380},
  publisher = {Association for Computational Linguistics},
  title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  url = {https://aclanthology.org/D18-1259},
  year = {2018},
}
""",
)


class HotpotQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA",
        dataset={
            "path": "mteb/hotpotqa",
            "revision": "ab518f4d6fcca38d87c25209f94beba119d02014",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong "
            "supervision for supporting facts to enable more explainable question answering systems."
        ),
        prompt={
            "query": "Given a multi-hop question, retrieve documents that can help answer the question"
        },
        **_hotpot_qa_metadata,
    )


class HotpotQAHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQAHardNegatives",
        dataset={
            "path": "mteb/HotpotQA_test_top_250_only_w_correct-v2",
            "revision": "617612fa63afcb60e3b134bed8b7216a99707c37",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong "
            "supervision for supporting facts to enable more explainable question answering systems. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
        ),
        adapted_from=["HotpotQA"],
        superseded_by="HotpotQAHardNegatives.v2",
        **_hotpot_qa_metadata,
    )


class HotpotQAHardNegativesV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQAHardNegatives.v2",
        dataset={
            "path": "mteb/HotpotQA_test_top_250_only_w_correct-v2",
            "revision": "617612fa63afcb60e3b134bed8b7216a99707c37",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong "
            "supervision for supporting facts to enable more explainable question answering systems. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)"
        ),
        adapted_from=["HotpotQA"],
        prompt={
            "query": "Given a multi-hop question, retrieve documents that can help answer the question"
        },
        **_hotpot_qa_metadata,
    )
