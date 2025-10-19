from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GermanDPR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GermanDPR",
        description="GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer",
        reference="https://huggingface.co/datasets/deepset/germandpr",
        dataset={
            "path": "mteb/GermanDPR",
            "revision": "64a4860e55ba6d8fcb923d5306d08e08b1c72794",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=("2020-05-19", "2021-04-26"),
        domains=["Written", "Non-fiction", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{möller2021germanquad,
  archiveprefix = {arXiv},
  author = {Timo Möller and Julian Risch and Malte Pietsch},
  eprint = {2104.12741},
  primaryclass = {cs.CL},
  title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
  year = {2021},
}
""",
    )
