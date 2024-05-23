from __future__ import annotations

import random

from datasets import load_dataset
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

TEST_SAMPLES = 2048


class VieQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VieQuADRetrieval",
        description="A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.",
        reference="https://aclanthology.org/2020.coling-main.233.pdf",
        dataset={
            "path": "taidng/UIT-ViQuAD2.0",
            "revision": "406f09a45cc106a8f7b7fd0c25078883fe58cb1f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["validation"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2022-03-02", "2022-03-02"),
        form=["written"],
        domains=["Encyclopaedic", "Non-fiction"],
        task_subtypes=["Question answering"],
        license="mit",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{nguyen-etal-2020-vietnamese,
title = "A Vietnamese Dataset for Evaluating Machine Reading Comprehension",
author = "Nguyen, Kiet  and
    Nguyen, Vu  and
    Nguyen, Anh  and
    Nguyen, Ngan",
editor = "Scott, Donia  and
    Bel, Nuria  and
    Zong, Chengqing",
booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
month = dec,
year = "2020",
address = "Barcelona, Spain (Online)",
publisher = "International Committee on Computational Linguistics",
url = "https://aclanthology.org/2020.coling-main.233",
doi = "10.18653/v1/2020.coling-main.233",
pages = "2595--2605"}""",
        n_samples={"validation": TEST_SAMPLES},
        avg_character_length={"validation": 790.24},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        seed = 42
        random.seed(seed)
        split = self.metadata_dict["eval_splits"][0]
        ds = load_dataset(**self.metadata_dict["dataset"], split=split)
        ds = ds.shuffle(seed=seed)

        titles, questions, contexts, answers = [], [], [], []
        for row in ds:
            answer = row["answers"]["text"]
            if not answer:
                continue
            titles.append(row["title"])
            questions.append(row["question"])
            contexts.append(row["context"])
            answers.append(answer[0])

        # Downsample after filtering
        max_samples = min(TEST_SAMPLES, len(contexts))
        indices = list(range(len(contexts)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        titles = [titles[idx] for idx in indices]
        questions = [questions[idx] for idx in indices]
        contexts = [contexts[idx] for idx in indices]
        answers = [answers[idx] for idx in indices]

        self.corpus = {split: {}}
        self.relevant_docs = {split: {}}
        self.queries = {split: {}}

        text2id = {}
        n = 0
        for t, q, cont, ans in zip(titles, questions, contexts, answers):
            self.queries[split][str(n)] = q
            q_n = n
            n += 1
            if cont not in text2id:
                text2id[cont] = n
                self.corpus[split][str(n)] = {"title": t, "text": cont}
                n += 1
            if ans not in text2id and ans:
                text2id[ans] = n
                self.corpus[split][str(n)] = {"title": t, "text": ans}
                n += 1

            self.relevant_docs[split][str(q_n)] = {
                str(text2id[ans]): 1,
                str(text2id[cont]): 1,
            }  # only two correct matches

        self.data_loaded = True
