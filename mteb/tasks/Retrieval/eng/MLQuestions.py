import csv

from huggingface_hub import snapshot_download

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MLQuestionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MLQuestions",
        dataset={
            "path": "McGill-NLP/mlquestions",
            "revision": "83b690cb666c5a8869e7f213a877bbd24a642d7c",
        },
        reference="https://github.com/McGill-NLP/MLQuestions",
        description=(
            "MLQuestions is a domain adaptation dataset for the machine learning domain"
            "It consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)"
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-03-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @inproceedings{kulshreshtha-etal-2021-back,
                title = "Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval",
                author = "Kulshreshtha, Devang  and
                Belfer, Robert  and
                Serban, Iulian Vlad  and
                Reddy, Siva",
                booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
                month = nov,
                year = "2021",
                address = "Online and Punta Cana, Dominican Republic",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.emnlp-main.566",
                pages = "7064--7078",
                abstract = "In this work, we introduce back-training, an alternative to self-training for unsupervised domain adaptation (UDA). While self-training generates synthetic training data where natural inputs are aligned with noisy outputs, back-training results in natural outputs aligned with noisy inputs. This significantly reduces the gap between target domain and synthetic data distribution, and reduces model overfitting to source domain. We run UDA experiments on question generation and passage retrieval from the Natural Questions domain to machine learning and biomedical domains. We find that back-training vastly outperforms self-training by a mean improvement of 7.8 BLEU-4 points on generation, and 17.6{\%} top-20 retrieval accuracy across both domains. We further propose consistency filters to remove low-quality synthetic data before training. We also release a new domain-adaptation dataset - MLQuestions containing 35K unaligned questions, 50K unaligned passages, and 3K aligned question-passage pairs.",
            }
        """,
        n_samples={"dev": 1500, "test": 1500},
        avg_character_length={"dev": 305, "test": 307},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        revision = self.metadata_dict["dataset"].get("revision", None)
        download_dir = snapshot_download(
            repo_id=dataset_path, repo_type="dataset", revision=revision
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = self._load_data_for_split(download_dir, split)
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self, download_dir, split):
        queries, corpus, qrels = {}, {}, {}

        dataset_path = f"{download_dir}/{split}.csv"
        with open(dataset_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                query_id = str(i)
                doc_id = row["indexes"]
                query = row["target_text"]
                queries[query_id] = query
                qrels[query_id] = {doc_id: 1}

        # Same corpus for all splits
        corpus_path = f"{download_dir}/test_passages.csv"
        with open(corpus_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                doc_id = str(i)
                corpus[doc_id] = {
                    "title": "",
                    "text": row["input_text"],
                }

        return corpus, queries, qrels
