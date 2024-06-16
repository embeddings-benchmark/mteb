from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlloprofRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlloprofRetrieval",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "lyon-nlp/alloprof",
            "revision": "fcf295ea64c750f41fadbaa37b9b861558e1bfbd",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=None,  # no date specified.
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{lef23,
  doi = {10.48550/ARXIV.2302.07738},
  url = {https://arxiv.org/abs/2302.07738},
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}""",
        n_samples={"train": 2048},
        avg_character_length=None,  # unable to extract due to invalid hf dataset
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(
            name="documents",
            **self.metadata_dict["dataset"],
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata_dict["dataset"],
        )
        eval_split = self.metadata_dict["eval_splits"][0]
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
