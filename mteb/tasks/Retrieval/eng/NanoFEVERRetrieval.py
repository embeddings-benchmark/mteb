from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoFEVERRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoFEVERRetrieval",
        description="NanoFEVER is a smaller version of "
        + "FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences"
        + " extracted from Wikipedia and subsequently verified without knowledge of the sentence they were"
        + " derived from.",
        reference="https://fever.ai/",
        dataset={
            "path": "zeta-alpha-ai/NanoFEVER",
            "revision": "a8bfdf1bf15181167a7e22e69cf8754bdea9b4c8",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-12-31"],
        domains=["Academic", "Encyclopaedic"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{thorne-etal-2018-fever,
    title = "{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification",
    author = "Thorne, James  and
      Vlachos, Andreas  and
      Christodoulopoulos, Christos  and
      Mittal, Arpit",
    editor = "Walker, Marilyn  and
      Ji, Heng  and
      Stent, Amanda",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1074",
    doi = "10.18653/v1/N18-1074",
    pages = "809--819",
    abstract = "In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss kappa. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. To characterize the challenge of the dataset presented, we develop a pipeline approach and compare it to suitably designed oracles. The best accuracy we achieve on labeling a claim accompanied by the correct evidence is 31.87{\%}, while if we ignore the evidence we achieve 50.91{\%}. Thus we believe that FEVER is a challenging testbed that will help stimulate progress on claim verification against textual sources.",
}""",
        prompt={
            "query": "Given a claim, retrieve documents that support or refute the claim"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoFEVER",
            "corpus",
            revision="a8bfdf1bf15181167a7e22e69cf8754bdea9b4c8",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoFEVER",
            "queries",
            revision="a8bfdf1bf15181167a7e22e69cf8754bdea9b4c8",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoFEVER",
            "qrels",
            revision="a8bfdf1bf15181167a7e22e69cf8754bdea9b4c8",
        )

        self.corpus = {
            split: {
                sample["_id"]: {"_id": sample["_id"], "text": sample["text"]}
                for sample in self.corpus[split]
            }
            for split in self.corpus
        }

        self.queries = {
            split: {sample["_id"]: sample["text"] for sample in self.queries[split]}
            for split in self.queries
        }

        self.relevant_docs = {
            split: {
                sample["query-id"]: {sample["corpus-id"]: 1}
                for sample in self.relevant_docs[split]
            }
            for split in self.relevant_docs
        }

        self.data_loaded = True
