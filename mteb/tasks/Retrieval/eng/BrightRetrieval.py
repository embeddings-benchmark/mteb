from collections import defaultdict

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval, MultilingualTask

DOMAINS_LONG = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
]

DOMAINS = DOMAINS_LONG + [
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

DOMAINS_langs = {split: ["eng-Latn"] for split in DOMAINS}


EVAL_SPLITS = ["standard", "long"]


class BrightRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description=("Bright retrieval dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=EVAL_SPLITS,
        eval_langs=DOMAINS_langs,
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        form=["written"],
        domains=["Non-fiction"],
        task_subtypes=["Article retrieval"],
        license="CC-BY-4.0",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation="""
            @misc{BRIGHT,
            title={BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
            author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and Sun, Ruoxi and Yoon, Jinsung and Arik, Sercan O and Chen, Danqi and Yu, Tao},
            year={2024},
            }
        """,
        descriptive_stats={
            "n_samples": {"standard": 1334914, "long": 7048},
            "avg_character_length": {
                "standard": 800.3994729248476,
                "long": 46527.35839954597,
            },
        },
    )

    def load_bright_data(
        self,
        path: str,
        domains: list,
        eval_splits: list,
        cache_dir: str = None,
        revision: str = None,
    ):
        corpus = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
        queries = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
        relevant_docs = {
            domain: {split: None for split in eval_splits} for domain in DOMAINS
        }

        for domain in domains:
            domain_corpus = datasets.load_dataset(
                path, "documents", split=domain, cache_dir=cache_dir, revision=revision
            )
            examples = datasets.load_dataset(
                path, "examples", split=domain, cache_dir=cache_dir, revision=revision
            )
            if domain in DOMAINS_LONG:
                domain_corpus_long = datasets.load_dataset(
                    path,
                    "long_documents",
                    split=domain,
                    cache_dir=cache_dir,
                    revision=revision,
                )
            corpus[domain]["standard"] = {
                e["id"]: {"text": e["content"]} for e in domain_corpus
            }
            if domain in DOMAINS_LONG:
                corpus[domain]["long"] = {
                    e["id"]: {"text": e["content"]} for e in domain_corpus_long
                }
            queries[domain]["standard"] = queries[domain]["long"] = {
                e["id"]: e["query"] for e in examples
            }
            relevant_docs[domain]["standard"] = {}
            relevant_docs[domain]["long"] = {}

            for e in examples:
                qid = e["id"]
                gold_ids = e["gold_ids"]
                gold_ids_long = e["gold_ids_long"]
                relevant_docs[domain]["standard"][qid] = defaultdict(dict)
                relevant_docs[domain]["long"][qid] = defaultdict(dict)
                for gid in gold_ids:
                    relevant_docs[domain]["standard"][qid].update({gid: 1})
                for gid in gold_ids_long:
                    relevant_docs[domain]["long"][qid].update({gid: 1})

        corpus = datasets.DatasetDict(corpus)
        queries = datasets.DatasetDict(queries)
        relevant_docs = datasets.DatasetDict(relevant_docs)
        return corpus, queries, relevant_docs

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = self.load_bright_data(
            path=self.metadata_dict["dataset"]["path"],
            domains=DOMAINS,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True
