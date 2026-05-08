import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TurHistQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TurHistQuadRetrieval",
        dataset={
            "path": "asparius/TurHistQuAD",
            "revision": "2a2b8ddecf1189f530676244d0751e1d0a569e03",
        },
        description="Question Answering dataset on Ottoman History in Turkish",
        reference="https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-10-13"),
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Non-fiction", "Academic", "Written"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{9559013,
  author = {Soygazi, Fatih and Çiftçi, Okan and Kök, Uğurcan and Cengiz, Soner},
  booktitle = {2021 6th International Conference on Computer Science and Engineering (UBMK)},
  doi = {10.1109/UBMK52708.2021.9559013},
  keywords = {Computer science;Computational modeling;Neural networks;Knowledge discovery;Information retrieval;Natural language processing;History;question answering;information retrieval;natural language understanding;deep learning;contextualized word embeddings},
  number = {},
  pages = {215-220},
  title = {THQuAD: Turkish Historic Question Answering Dataset for Reading Comprehension},
  volume = {},
  year = {2021},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        if self.data_loaded:
            return

        hf_dataset = datasets.load_dataset(**self.metadata.dataset)

        self.dataset = {}
        text2id = {}

        for split in self.metadata.eval_splits:
            ds: datasets.Dataset = hf_dataset[split]
            ds = ds.shuffle(seed=42)
            max_samples = min(1024, len(ds))
            ds = ds.select(
                range(max_samples)
            )  # limit the dataset size to make sure the task does not take too long to run

            queries_dict = {}
            corpus_dict = {}
            relevant_docs = {}

            question = ds["question"]
            context = ds["context"]
            answer = [a["text"] for a in ds["answers"]]

            n = 0
            for q, cont, ans in zip(question, context, answer):
                queries_dict[str(n)] = q
                q_n = n
                n += 1
                if cont not in text2id:
                    text2id[cont] = n
                    corpus_dict[str(n)] = {"title": "", "text": cont}
                    n += 1
                if ans not in text2id:
                    text2id[ans] = n
                    corpus_dict[str(n)] = {"title": "", "text": ans}
                    n += 1

                relevant_docs[str(q_n)] = {
                    str(text2id[ans]): 1,
                    str(text2id[cont]): 1,
                }  # only two correct matches

            corpus_dataset = Dataset.from_list(
                [
                    {"id": k, "text": v["text"], "title": v["title"]}
                    for k, v in corpus_dict.items()
                ]
            )
            queries_dataset = Dataset.from_list(
                [{"id": k, "text": v} for k, v in queries_dict.items()]
            )

            self.dataset.setdefault("default", {})[split] = {
                "corpus": corpus_dataset,
                "queries": queries_dataset,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }
        self.data_loaded = True
