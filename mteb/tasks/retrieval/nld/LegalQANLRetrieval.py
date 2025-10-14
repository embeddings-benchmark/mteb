import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LegalQANLRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="LegalQANLRetrieval",
        description="To this end, we create and publish a Dutch legal QA dataset, consisting of question-answer pairs "
        "with attributions to Dutch law articles.",
        reference="https://aclanthology.org/2024.nllp-1.12/",
        dataset={
            "path": "clips/mteb-nl-legalqa",
            "revision": "099ecf6131a826c6f81cc74f0a3a2452fa51644d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            name="corpus",
            split="corpus",
            **self.metadata.dataset,
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            split="queries",
            **self.metadata.dataset,
        )

        self.queries = {
            self.metadata.eval_splits[0]: {
                str(q["question_id"]): q["question"].strip() for q in queries_raw
            }
        }

        self.corpus = {
            self.metadata.eval_splits[0]: {
                str(d["DOC_ID"]): {
                    "text": ", ".join(
                        field
                        for field in (
                            d.get("law_name"),
                            d.get("hoofdstuk"),
                            d.get("hoofdstuk_titel"),
                            d.get("afdeling"),
                            d.get("afdeling_titel"),
                            d.get("paragraaf"),
                            d.get("paragraaf_titel"),
                            d.get("subparagraaf_titel"),
                            d.get("titel_titel"),
                            d.get("artikel"),
                            d.get("article_name"),
                        )
                        if field and field.strip()
                    )
                    + "\n"
                    + d.get("text")
                }
                for d in corpus_raw
            }
        }

        self.relevant_docs = {"test": {str(q["question_id"]): {} for q in queries_raw}}

        for q in queries_raw:
            for doc_id in q["human_attribution"].split(","):
                self.relevant_docs[self.metadata.eval_splits[0]][str(q["question_id"])][
                    doc_id
                ] = 1

        self.data_loaded = True
