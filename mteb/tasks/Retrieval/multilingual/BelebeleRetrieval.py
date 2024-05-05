from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"

_LANGS = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Hira"],
    "ko": ["kor-Hang"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ta": ["tam-Taml"],
    "zh": ["cmn-Hans"],
}



class BelebeleRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BelebeleRetrieval",
        dataset={
            "path": "facebook/belebele",
            "revision": "75b399394a9803252cfec289d103de462763db7c",
        },
        description=(
            "Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants."
        ),
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        license="CC-BY-SA-4.0",
        domains=["Web"],
        text_creation="created",
        n_samples={_EVAL_SPLIT: 2566},
        reference="https://github.com/facebookresearch/belebele",
        date=("2024-05-02", "2024-05-03"),
        form=["written"],
        task_subtypes=["Question answering"],
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
        avg_character_length={_EVAL_SPLIT: 572},
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = load_dataset(**self.metadata_dict["dataset"])

        split = "test"

        self.queries = {lang: {split: {}} for lang in self.langs}
        self.corpus = {lang: {split: {}} for lang in self.langs}
        self.relevant_docs = {lang: {split: {}} for lang in self.langs}

        for lang in self.langs:
            ds = self.dataset[_LANGS[lang]]

            question_ids = {
                question: _id for _id, question in enumerate(set(ds["question"]))
            }
            context_ids = {
                passage: _id for _id, passage in enumerate(set(ds["flores_passage"]))
            }
            correct_answer_nums = [int(num) for num in ds['correct_answer_num']]
            answers = [row[f"mc_answer{num}"] for row, num in zip(ds, correct_answer_nums)]
            answer_ids = {
                answer: _id for _id, answer in enumerate(set(answers))
            }

            for row in ds:
                query = row["question"]
                query_id = f"Q{question_ids[query]}"
                self.queries[lang][split][query_id] = query
                context = row['flores_passage']
                context_id = f"C{context_ids[context]}"
                self.corpus[lang][split][context_id] = {"title": "", "text": context}
                answer = row[f"mc_answer{row['correct_answer_num']}"]
                answer_id = f"A{answer_ids[answer]}"
                self.corpus[lang][split][answer_id] = {"title": "", "text": answer}

                self.relevant_docs[split][query_id] = {
                    context_id: 1,
                    answer_id: 1,
                }
            self.data_loaded = True