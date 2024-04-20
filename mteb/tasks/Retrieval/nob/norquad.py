import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class NorQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NorQuadRetrieval",
        dataset={
            "path": "mteb/norquad_retrieval",
            "revision": "9dcfcdb2aa578dd178330d49bf564248935f7fbe",
        },
        description="Human-created question for Norwegian wikipedia passages.",
        reference="https://aclanthology.org/2023.nodalida-1.17/",
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2023-12-31"),
        form=["written"],
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Non-fiction"],
        license="CC-BY-SA-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{ivanova-etal-2023-norquad,
    title = "{N}or{Q}u{AD}: {N}orwegian Question Answering Dataset",
    author = "Ivanova, Sardana  and
      Andreassen, Fredrik  and
      Jentoft, Matias  and
      Wold, Sondre  and
      {\O}vrelid, Lilja",
    editor = {Alum{\"a}e, Tanel  and
      Fishel, Mark},
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.17",
    pages = "159--168",
    abstract = "In this paper we present NorQuAD: the first Norwegian question answering dataset for machine reading comprehension. The dataset consists of 4,752 manually created question-answer pairs. We here detail the data collection procedure and present statistics of the dataset. We also benchmark several multilingual and Norwegian monolingual language models on the dataset and compare them against human performance. The dataset will be made freely available.",
}""",
        n_samples={"test": 2602},
        avg_character_length={"test": 502.19},
    )

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

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
            ds = ds.shuffle(seed=42)
            max_samples = min(1024, len(ds))
            ds = ds.select(
                range(max_samples)
            )  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            question = ds["question"]
            context = ds["context"]
            answer = [a["text"][0] for a in ds["answers"]]

            n = 0
            for q, cont, ans in zip(question, context, answer):
                self.queries[split][str(n)] = q
                q_n = n
                n += 1
                if cont not in text2id:
                    text2id[cont] = n
                    self.corpus[split][str(n)] = {"title": "", "text": cont}
                    n += 1
                if ans not in text2id:
                    text2id[ans] = n
                    self.corpus[split][str(n)] = {"title": "", "text": ans}
                    n += 1

                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[ans]): 1,
                    str(text2id[cont]): 1,
                }  # only two correct matches
