import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata
import json

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
        category="p2p",
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-10-13"),
        form=["written"],
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Non-fiction", "Academic"],
        license="MIT",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @INPROCEEDINGS{9559013,
                author={Soygazi, Fatih and Çiftçi, Okan and Kök, Uğurcan and Cengiz, Soner},
                booktitle={2021 6th International Conference on Computer Science and Engineering (UBMK)}, 
                title={THQuAD: Turkish Historic Question Answering Dataset for Reading Comprehension}, 
                year={2021},
                volume={},
                number={},
                pages={215-220},
                keywords={Computer science;Computational modeling;Neural networks;Knowledge discovery;Information retrieval;Natural language processing;History;question answering;information retrieval;natural language understanding;deep learning;contextualized word embeddings},
                doi={10.1109/UBMK52708.2021.9559013}}

        """,
        n_samples={"test": 1330, "train": 14221},
        avg_character_length={"train": 1219.37, "test": 1513.83},
    )

    def load_data(self,**kwargs) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """

        if self.data_loaded:
            return

        
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        
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
            answer = [a["text"] for a in ds["answers"]]

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
            self.data_loaded = True