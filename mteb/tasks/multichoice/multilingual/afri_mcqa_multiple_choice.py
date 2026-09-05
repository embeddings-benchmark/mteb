from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/AfriMCQA-multiple-choice"
_DATASET_REVISION = "3ed9383894a3ff0261af4097d1e4e9d593868f1d"

_LANGUAGES = {
    "twi": ["twi-Latn"],
    "amh": ["amh-Ethi"],
    "nya": ["nya-Latn"],
    "hau": ["hau-Latn"],
    "ibo": ["ibo-Latn"],
    "kik": ["kik-Latn"],
    "kin": ["kin-Latn"],
    "lin": ["lin-Latn"],
    "lug": ["lug-Latn"],
    # Afri-MCQA's Oromo is the West Central variety, so `gaz` rather than the `orm`
    # macrolanguage
    "orm": ["gaz-Latn"],
    "sot": ["sot-Latn"],
    "tsn": ["tsn-Latn"],
    "som": ["som-Latn"],
    "tir": ["tir-Ethi"],
    "yor": ["yor-Latn"],
    "zul": ["zul-Latn"],
}


class AfriMCQAVisionCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AfriMCQAVisionCentricQA",
        description=(
            "Culturally grounded multiple choice questions about photographs, written by "
            "native speakers of 16 African languages. Given a photograph and a question "
            "about it in the native language, pick the correct answer from that "
            "question's own options. Every multiple choice task in `mteb` is currently "
            "English only, so this is the first multilingual one. #5356 added a2i/i2a "
            "retrieval over the same source; this is the QA formulation. Built from the "
            "official dev split, the only one where the answer is labelled, since the "
            "test split ships the four options already shuffled with no key. Options are "
            "shuffled by a hash of the question, because the source lists the correct "
            "answer first in every row. Questions with no photograph, questions listing "
            "their own answer among their distractors, and repeats are dropped. "
            "Construction script: scripts/data/afri_mcqa_multiple_choice/create_data.py."
        ),
        reference="https://arxiv.org/abs/2601.05699",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="VisionCentricQA",
        category="it2t",
        modalities=["image", "text"],
        eval_splits=["dev"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2025-01-01", "2026-01-15"),
        domains=["Scene"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        prompt={"query": "Answer this question about the image."},
        bibtex_citation=r"""
@inproceedings{tonja2026afrimcqa,
  author = {Tonja, Atnafu Lambebo and Anand, Srija and Villa-Cueva, Emilio and Azime, Israel Abebe and Alabi, Jesujoba Oluwadara and Mohamed, Muhidin A. and Yadeta, Debela Desalegn and Abadi, Negasi Haile and Oppong, Abigail and Obiefuna, Nnaemeka Casmir and Abdulmumin, Idris and Etori, Naome A},
  title = {{Afri-MCQA}: Multimodal Cultural Question Answering for {African} Languages},
  year = {2026},
}
""",
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        split = self.metadata.eval_splits[0]
        self.dataset = {}
        for lang in _LANGUAGES:
            rows = load_dataset(
                _DATASET_PATH, lang, revision=_DATASET_REVISION, split=split
            )
            queries = rows.select_columns(["id", "text", "image"])

            corpus_rows: list[dict] = []
            relevant_docs: dict[str, dict[str, int]] = {}
            top_ranked: dict[str, list[str]] = {}
            # Candidates are read without the image column so that scanning the options
            # does not decode a photograph for every row.
            for row in rows.select_columns(["id", "candidates", "answer"]):
                qid = row["id"]
                top_ranked[qid] = []
                for j, candidate in enumerate(row["candidates"]):
                    doc_id = f"{qid}_c{j}"
                    corpus_rows.append({"id": doc_id, "text": candidate})
                    top_ranked[qid].append(doc_id)
                    if candidate == row["answer"]:
                        relevant_docs[qid] = {doc_id: 1}

            self.dataset[lang] = {
                split: RetrievalSplitData(
                    queries=queries,
                    corpus=Dataset.from_list(corpus_rows),
                    relevant_docs=relevant_docs,
                    top_ranked=top_ranked,
                )
            }
        self.data_loaded = True
