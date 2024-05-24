from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class AlloprofReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="AlloprofReranking",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-alloprof-s2p",
            "revision": "65393d0d7a08a10b4e348135e824f385d420b0fd",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map",
        date=("2020-01-01", "2023-04-14"), # supposition
        form=["written"],
        domains=["Web", "Academic"],
        task_subtypes=None,
        license="CC BY-NC-SA 4.0",
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=None,
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
        n_samples={"test": 2316, "train": 9264},
        avg_character_length=None,
    )

    def load_data(self, **kwargs):

        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            name="queries",
            **self.metadata_dict["dataset"],
            split=self.metadata.eval_splits[0]
        )
        documents = datasets.load_dataset(
            name="documents",
            **self.metadata_dict["dataset"],
            split="test"
        )
        # replace documents ids in positive and negative column by their respective texts
        doc_id2txt = dict(list(zip(documents["doc_id"], documents["text"])))

        self.dataset = self.dataset.map(lambda x: {
            "positive": [
                doc_id2txt[docid] for docid in x["positive"]
            ],
            "negative": [
                doc_id2txt[docid] for docid in x["negative"]
            ]
        })
        self.dataset = datasets.DatasetDict({"test": self.dataset})

        self.data_loaded = True


    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed,
        )