from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{figma2026,
  author = {Anand, Nishit and Seth, Ashish and Ghosh, Sreyan and Manocha, Dinesh and Duraiswami, Ramani},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  title = {FIGMA: Towards FIne-Grained Music retrievAl},
  url = {https://arxiv.org/abs/2606.06615},
  year = {2026},
}
"""


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
    num_proc: int | None = None,
    *,
    audio_to_text: bool = False,
):
    corpus = {}
    queries = {}
    relevant_docs = {}

    for split in splits:
        dataset = load_dataset(
            path,
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        text_dataset = dataset.select_columns(["id", "caption"]).rename_column(
            "caption", "text"
        )
        audio_dataset = dataset.select_columns(["id", "audio"])

        if audio_to_text:
            queries[split] = audio_dataset
            corpus[split] = text_dataset
        else:
            queries[split] = text_dataset
            corpus[split] = audio_dataset

        relevant_docs[split] = {id_: {id_: 1} for id_ in dataset["id"]}

    return corpus, queries, relevant_docs


class FGMCapsT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FGMCapsT2ARetrieval",
        description=(
            "FGMCaps-Test is a fine-grained music retrieval benchmark containing "
            "10,000 10-second music clips. Each clip is paired with an English "
            "caption describing precise musical attributes such as tempo, key, "
            "chord progression, and time signature. Text-to-audio retrieval uses "
            "each caption to retrieve its corresponding music clip."
        ),
        reference="https://arxiv.org/abs/2606.06615",
        dataset={
            "path": "nishitanand/FGMCaps-benchmark",
            "revision": "a2f3b8bccc85b7e63563a0976b155e48e7be5278",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2026-06-01", "2026-06-30"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the music clip described by this caption."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )
        self.data_loaded = True


class FGMCapsA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FGMCapsA2TRetrieval",
        description=(
            "FGMCaps-Test is a fine-grained music retrieval benchmark containing "
            "10,000 10-second music clips. Each clip is paired with an English "
            "caption describing precise musical attributes such as tempo, key, "
            "chord progression, and time signature. Audio-to-text retrieval uses "
            "each music clip to retrieve its corresponding caption."
        ),
        reference="https://arxiv.org/abs/2606.06615",
        dataset={
            "path": "nishitanand/FGMCaps-benchmark",
            "revision": "a2f3b8bccc85b7e63563a0976b155e48e7be5278",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2026-06-01", "2026-06-30"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve the caption that describes this music clip."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
            audio_to_text=True,
        )
        self.data_loaded = True
