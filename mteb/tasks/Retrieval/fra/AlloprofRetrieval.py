from __future__ import annotations

import datasets

from mteb.abstasks.task_metadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlloprofRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlloprofRetrieval",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "mteb/AlloprofRetrieval",
            "revision": "96a52a1744e2d1981a66d93ff3b5a595f9a150d5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=None,  # no date specified.
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lef23,
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  doi = {10.48550/ARXIV.2302.07738},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  url = {https://arxiv.org/abs/2302.07738},
  year = {2023},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(
            name="documents",
            **self.metadata.dataset,
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata.dataset,
        )
        eval_split = self.metadata.eval_splits[0]
        self.queries = {
            eval_split: {str(q["id"]): q["text"] for q in queries_raw[eval_split]}
        }
        self.corpus = {
            eval_split: {
                str(d["uuid"]): {"text": d["text"]} for d in corpus_raw[eval_split]
            }
        }

        self.relevant_docs = {eval_split: {}}
        for q in queries_raw[eval_split]:
            for r in q["relevant"]:
                self.relevant_docs[eval_split][str(q["id"])] = {r: 1}

        self.data_loaded = True
