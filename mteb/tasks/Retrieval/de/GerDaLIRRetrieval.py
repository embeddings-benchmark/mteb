import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GerDaLIR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GerDaLIR",
        description="GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        hf_hub_name="jinaai/ger_da_lir",
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["de"],
        main_score="ndcg_at_10",
        revision="0bb47f1d73827e96964edb84dfe552f62f4fd5eb",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "queries",
            revision=self.metadata_dict.get("revision", None),
            split=self._EVAL_SPLIT,
        )
        corpus_rows = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "corpus",
            revision=self.metadata_dict.get("revision", None),
            split=self._EVAL_SPLIT,
        )
        qrels_rows = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "qrels",
            revision=self.metadata_dict.get("revision", None),
            split=self._EVAL_SPLIT,
        )

        self.queries = {
            self._EVAL_SPLIT: {row["_id"]: row["text"] for row in query_rows}
        }
        self.corpus = {self._EVAL_SPLIT: {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows
            }
        }

        self.data_loaded = True
