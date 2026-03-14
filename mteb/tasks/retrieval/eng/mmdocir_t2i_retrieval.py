from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

DOMAIN_MAP = {
    "academic_paper": "Academic paper",
    "admin_industry": "Administration/Industry file",
    "brochure": "Brochure",
    "financial_report": "Financial report",
    "government": "Government",
    "guidebook": "Guidebook",
    "laws": "Laws",
    "news": "News",
    "research_report": "Research report / Introduction",
    "tutorial_workshop": "Tutorial/Workshop",
}

DOMAINS = list(DOMAIN_MAP.keys())
DOMAINS_LANGS = {domain: ["eng-Latn"] for domain in DOMAINS}


def _load_data(
    path: str,
    domains: list[str],
    eval_splits: list[str],
    revision: str | None = None,
    num_proc: int | None = None,
):
    page_features = Features(
        {
            "doc_name": Value("string"),
            "domain": Value("string"),
            "passage_id": Value("string"),
            "image_path": Value("string"),
            "image_binary": Value("binary"),
            "ocr_text": Value("string"),
            "vlm_text": Value("string"),
        }
    )

    pages_ds = load_dataset(
        path,
        data_files="MMDocIR_pages.parquet",
        features=page_features,
        split="train",
        revision=revision,
        num_proc=num_proc,
    )

    annotations_ds = load_dataset(
        path,
        data_files="MMDocIR_annotations.jsonl",
        split="train",
        revision=revision,
        num_proc=num_proc,
    )

    # Group pages and annotations by domain
    name_to_key = {v: k for k, v in DOMAIN_MAP.items()}

    pages_by_domain = {domain: [] for domain in domains}
    for global_idx, row in enumerate(pages_ds):
        key = name_to_key.get(row["domain"])
        if key in pages_by_domain:
            pages_by_domain[key].append((global_idx, row))

    annotations_by_domain = {domain: [] for domain in domains}
    for ann in annotations_ds:
        key = name_to_key.get(ann["domain"])
        if key in annotations_by_domain:
            annotations_by_domain[key].append(ann)

    corpus = {domain: dict.fromkeys(eval_splits) for domain in domains}
    queries = {domain: dict.fromkeys(eval_splits) for domain in domains}
    relevant_docs = {domain: dict.fromkeys(eval_splits) for domain in domains}
    top_ranked = {domain: dict.fromkeys(eval_splits) for domain in domains}

    for domain in domains:
        # Build corpus per domain (image only)
        corpus_records = []
        for global_idx, row in pages_by_domain[domain]:
            corpus_records.append(
                {
                    "id": f"corpus-{global_idx}",
                    "image": {"bytes": row["image_binary"]},
                    "modality": "image",
                }
            )
        corpus_ds = Dataset.from_list(corpus_records)
        corpus_ds = corpus_ds.cast_column("image", Image())

        # Build queries, relevant_docs, and top_ranked per domain
        query_records = []
        relevant_docs_split = {}
        top_ranked_split = {}
        query_idx = 0
        for ann in annotations_by_domain[domain]:
            start_pid, end_pid = ann["page_indices"]
            # All pages of this document are candidates
            doc_corpus_ids = [f"corpus-{i}" for i in range(start_pid, end_pid + 1)]
            for q in ann["questions"]:
                qid = f"query-{query_idx}"
                query_records.append(
                    {
                        "id": qid,
                        "text": q["Q"],
                        "modality": "text",
                    }
                )
                relevant_docs_split[qid] = {
                    f"corpus-{start_pid + page_id}": 1 for page_id in q["page_id"]
                }
                top_ranked_split[qid] = doc_corpus_ids
                query_idx += 1

        queries_ds = Dataset.from_list(query_records)

        for split in eval_splits:
            corpus[domain][split] = corpus_ds
            queries[domain][split] = queries_ds
            relevant_docs[domain][split] = relevant_docs_split
            top_ranked[domain][split] = top_ranked_split

    corpus = DatasetDict(corpus)
    queries = DatasetDict(queries)
    relevant_docs = DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs, top_ranked


class MMDocIRT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMDocIRT2IRetrieval",
        description="MMDocIR evaluation set includes 313 long documents averaging 65.1 pages, categorized into ten main domains: research reports, administration&industry, tutorials&workshops, academic papers, brochures, financial reports, guidebooks, government documents, laws, and news articles. Different domains feature distinct distributions of multi-modal information. Overall, the modality distribution is: Text (60.4%), Image (18.8%), Table (16.7%), and other modalities (4.1%).",
        reference="https://arxiv.org/abs/2501.08828",
        dataset={
            "path": "MMDocIR/MMDocIR_Evaluation_Dataset",
            "revision": "bdcb36ecb3eee73667180ee3fb24fe433f6dd2a4",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["train"],
        eval_langs=DOMAINS_LANGS,
        main_score="recall_at_5",
        date=("2025-01-01", "2025-01-01"),
        domains=["Academic", "Non-fiction", "Government", "Legal", "News"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{dong2025mmdocirbenchmarkingmultimodalretrieval,
  archiveprefix = {arXiv},
  author = {Kuicai Dong and Yujing Chang and Xin Deik Goh and Dexun Li and Ruiming Tang and Yong Liu},
  eprint = {2501.08828},
  primaryclass = {cs.IR},
  title = {MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents},
  url = {https://arxiv.org/abs/2501.08828},
  year = {2025},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            domains=list(self.metadata.eval_langs.keys()),
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )

        self.data_loaded = True
