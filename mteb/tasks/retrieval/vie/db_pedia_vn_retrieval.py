from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DBPediaVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-VN",
        description="A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "GreenNode/dbpedia-vn",
            "revision": "c3e20179fbcee16217ef9461a14a54b7faca9b63",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Written", "Encyclopaedic"],
        task_subtypes=[],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["DBPedia"],
    )


class NanoDBPediaVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoDBPedia-VN",
        description="A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "GreenNode/dbpedia-vn",
            "revision": "c3e20179fbcee16217ef9461a14a54b7faca9b63",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Written", "Encyclopaedic"],
        task_subtypes=[],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["DBPedia-VN"],
    )

    def load_data(self) -> None:
        """Load and optionally downsample the DBPedia-VN dataset.

        This override:
        - uses the standard `AbsTaskRetrieval.load_data` / `RetrievalDatasetLoader` pipeline
        - then downsamples the corpus and queries
        - and guarantees that **all** `query-id` and `corpus-id` in `relevant_docs`
          still exist in the final `queries` / `corpus`.
        """
        if self.data_loaded:
            return

        # First, use the default retrieval loading logic (v2 format with `self.dataset`)
        super().load_data()

        # You can tune these limits if you want a smaller / larger debug subset.
        max_corpus_samples = 1000000
        max_query_samples = 100000

        for hf_subset, split_dict in self.dataset.items():
            for split_name, split_data in split_dict.items():
                corpus_ds = split_data["corpus"]
                queries_ds = split_data["queries"]
                qrels = split_data["relevant_docs"]
                top_ranked = split_data["top_ranked"]

                if len(qrels) == 0:
                    # Nothing to filter; keep original split
                    continue

                # Keep original full datasets / qrels so we can "put back"
                # all documents and queries that appear in the qrels after
                # we downsample.
                orig_corpus_ds = corpus_ds
                orig_queries_ds = queries_ds
                orig_qrels = qrels

                # 1) Optionally downsample corpus
                if len(corpus_ds) > max_corpus_samples:
                    corpus_ds = corpus_ds.shuffle(seed=42).select(
                        range(max_corpus_samples)
                    )

                # 2) Optionally downsample queries
                if len(queries_ds) > max_query_samples:
                    queries_ds = queries_ds.shuffle(seed=42).select(
                        range(max_query_samples)
                    )

                # 3) Ensure ALL queries-id and corpus-id that appear in qrels
                #    are present in the final datasets, even if they were
                #    dropped by the random downsampling above.
                qrel_query_ids = set(orig_qrels.keys())
                qrel_corpus_ids = {
                    cid for docs in orig_qrels.values() for cid in docs.keys()
                }

                # Rebuild queries from the FULL original set, keeping:
                # - all queries that appear in qrels (to preserve all positives)
                # - plus the downsampled queries we kept above (for extra negatives)
                sampled_query_ids = set(queries_ds["id"])

                def _keep_query(row):
                    rid = row["id"]
                    return (rid in qrel_query_ids) or (rid in sampled_query_ids)

                queries_ds = orig_queries_ds.filter(
                    _keep_query,
                    desc=f"Re-adding all positive queries for {hf_subset}/{split_name}",
                )

                # Rebuild corpus from the FULL original set, similarly.
                sampled_corpus_ids = set(corpus_ds["id"])

                def _keep_corpus(row):
                    rid = row["id"]
                    return (rid in qrel_corpus_ids) or (rid in sampled_corpus_ids)

                corpus_ds = orig_corpus_ds.filter(
                    _keep_corpus,
                    desc=f"Re-adding all positive corpus docs for {hf_subset}/{split_name}",
                )

                # 4) Use the ORIGINAL qrels so we keep all positives.
                qrels = orig_qrels

                # Replace the split with the downsampled / aligned data
                self.dataset[hf_subset][split_name]["corpus"] = corpus_ds
                self.dataset[hf_subset][split_name]["queries"] = queries_ds
                self.dataset[hf_subset][split_name]["relevant_docs"] = qrels
                self.dataset[hf_subset][split_name]["top_ranked"] = top_ranked

        self.data_loaded = True
