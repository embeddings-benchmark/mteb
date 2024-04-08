import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class TwitterHjerneRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TwitterHjerneRetrieval",
        dataset={
            "path": "sorenmulli/da-hashtag-twitterhjerne",
            "revision": "099ee143c7fdfa6bd7965be8c801cb161c313b29",
        },
        description="Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer.",
        reference="https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne",
        type="Retrieval",
        category="p2p",
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2006-01-01", "2024-12-31"),  # best guess
        form=["written"],
        domains=["Social"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
@article{holm2024gllms,
  title={Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish},
  author={Holm, S{\o}ren Vejlgaard},
  year={2024}
}
""",
        n_samples={"train": 340},
        avg_character_length={"train": 138.23},
        task_subtypes=["Question answering"],
    )

    def dataset_transform(self) -> None:
        """
        and transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.map(answers_to_list)

            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            questions = ds["Question"]
            answers = ds["answers"]

            n = 0
            for q, answ in zip(questions, answers):
                if len(q.split(" ")) < 4 and answ:
                    continue
                query_id = str(n)
                self.queries[split][query_id] = q
                n += 1
                answer_ids = []
                for a in answ:
                    if a not in text2id:
                        text2id[a] = n
                        answer_id = str(n)
                        self.corpus[split][answer_id] = {"title": "", "text": a}
                        n += 1
                    else:
                        answer_id = str(text2id[a])
                    answer_ids.append(answer_id)

                self.relevant_docs[split][query_id] = {
                    answer_id: 1 for answer_id in answer_ids
                }


def answers_to_list(example: dict) -> dict:
    example["answers"] = [
        v
        for k, v in example.items()
        if k.startswith("Answer") and v and len(v.split(" ")) > 3
    ]
    return example
