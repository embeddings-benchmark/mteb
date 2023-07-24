import datasets

from mteb.abstasks import AbsTaskRetrieval


class SweFAQRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SweFAQRetrieval",
            "hf_hub_name": "AI-Sweden/SuperLim",
            "description": "A Swedish FAQ dataset. Available as a part of Superlim",
            "reference": "https://spraakbanken.gu.se/en/resources/superlim",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "ndcg_at_10",
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
            "beir_name": "NA",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and convert to MTEB format
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"],
            "swefaq",  # select the right subset
            revision=self.description.get("revision"),
        )

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in self.description["eval_splits"]:
            dataset = self.dataset[split]  # type: ignore
            # answers = dataset["candidate_answer"]
            answers = dataset["correct_answer"]
            self.corpus[split] = {
                str(idx): {"_id": str(idx), "title": "", "text": answer} for idx, answer in enumerate(answers)
            }

            questions = dataset["question"]
            self.queries[split] = {str(idx): question for idx, question in enumerate(questions)}

            # Relevant documents is a mapping between each query and the desired responses (most relevant documents) for that query
            self.relevant_docs[split] = {str(idx): {str(idx): 1} for idx in range(len(answers))}

        self.data_loaded = True
