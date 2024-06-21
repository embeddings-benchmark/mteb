from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval, HFDataLoader


class ARCChallenge(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ARCChallenge",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.",
        reference="https://allenai.org/data/arc",
        dataset={
            "path": "RAR-b/ARC-Challenge",
            "revision": "c481e0da3dcbbad8bce7721dea9085b74320a0a3",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Reasoning as Retrieval"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{clark2018think,
  title={Think you have solved question answering? try arc, the ai2 reasoning challenge},
  author={Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
  journal={arXiv preprint arXiv:1803.05457},
  year={2018}
}
""",
        n_samples={"test": 1172},
        avg_character_length={"test": 161.7},
    )

    default_instruction = "Retrieve the answer to the question."
    
    def __init__(
        self, 
        include_task_instruction: bool = True, 
        instruction: str = None,
        format_func = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.include_instruction = include_task_instruction
        self.instruction = instruction if instruction is not None else self.default_instruction
        self.format_func = format_func if format_func is not None else self.default_format_func
    
    def default_format_func(self, instruction: str, query: str) -> str:
        return f"{instruction} {query}".strip()
    
    def concatenate_instruction(self, query: str) -> str:
        if self.include_instruction:
            return self.format_func(self.instruction, query)
        return query
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = HFDataLoader(
                hf_repo=dataset_path,
                hf_repo_qrels=hf_repo_qrels,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # Conversion from DataSet
            queries = {
                query["id"]: self.concatenate_instruction(query["text"]) if self.include_instruction else query["text"]
                for query in queries
            }
            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True