from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TopiOCQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TopiOCQA",
        dataset={
            "path": "McGill-NLP/TopiOCQA",
            "revision": "66cd1dbf5577c653ecb99b385200f08e15e12f30",
        },
        reference="https://mcgill-nlp.github.io/topiocqa",
        description=(
            "TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)"
            "is information-seeking conversational dataset with challenging topic switching phenomena."
            "It consists of conversation histories along with manually labelled relevant/gold passage."
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=["validation"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status=None,  # TODO: Not sure that this refers to
        annotations_creators="human-annotated",
        dialect=None,
        text_creation=None,  # TODO: Not sure what to put here, queries are "created", documents are "found"
        bibtex_citation="""
            @inproceedings{adlakha2022topiocqa,
            title={Topi{OCQA}: Open-domain Conversational Question Answering with Topic Switching},
            author={Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
            journal={Transactions of the Association for Computational Linguistics},
            volume = {10},
            pages = {468-483},
            year = {2022},
            month = {04},
            year={2022},
            issn = {2307-387X},
            doi = {10.1162/tacl_a_00471},
            url = {https://doi.org/10.1162/tacl\_a\_00471},
            eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00471/2008126/tacl\_a\_00471.pdf},
            }
        """,
        n_samples={"dev": 2514},
        avg_character_length=None,  # TODO: calculate and update
    )

    # TODO: Will be removed if curated and added to mteb HF
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = self._load_data_for_split(dataset_path, split)
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self, dataset_path, split):
        revision = self.metadata_dict["dataset"].get("revision", None)
        ds = load_dataset(
            dataset_path,
            split=split,
            revision=revision,
        )
        queries, corpus, qrels = {}, {}, {}
        for sample in ds:
            query_id = f"{sample['Conversation_no']}-{sample['Turn_no']}"
            query = sample["Context"] + [sample["Question"]]
            doc_id = sample["Gold_passage"]["id"]
            doc = {
                "title": "; ".join(sample["Gold_passage"]["title"].split(" [SEP] ")),
                "text": sample["Gold_passage"]["text"],
            }
            queries[query_id] = query
            corpus[doc_id] = doc
            qrels[query_id] = {doc_id: 1}

        return corpus, queries, qrels
