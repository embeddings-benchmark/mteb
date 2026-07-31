from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SHS100KA2ARetrieval(AbsTaskRetrieval):
    # Query audio is also in the corpus under a different id; drop that self-hit.
    skip_first_result = True

    metadata = TaskMetadata(
        name="SHS100KA2ARetrieval",
        description=(
            "Audio-to-audio cover-song retrieval on the classic SHS100K-TEST "
            "split (SecondHandSongs 100K): 116 musical works with multiple "
            "YouTube performances each. A query recording should retrieve other "
            "recordings of the same work (cover / alternate performance). "
            "Packaged from NovaFrost/SHS100K metadata with audio downloaded "
            "from the listed YouTube URLs."
        ),
        reference="https://github.com/NovaFrost/SHS100K",
        dataset={
            "path": "mteb/SHS100K-A2A-1k",
            "revision": "dec904a50796f536dce6395a571a973bdb7fd2c3",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Music"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{xu2018key,
  author = {Xu, Xiaoshuo and Chen, Xiaoou and Yang, Deshun},
  booktitle = {2018 IEEE International Conference on Multimedia and Expo (ICME)},
  organization = {IEEE},
  pages = {1--6},
  title = {Key-invariant convolutional neural network toward efficient cover song identification},
  year = {2018},
}
""",
        prompt={
            "query": "Retrieve another recording (cover) of the same musical work."
        },
        is_beta=True,
    )
