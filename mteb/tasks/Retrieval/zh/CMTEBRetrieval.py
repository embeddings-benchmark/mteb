from collections import defaultdict

from datasets import DatasetDict, load_dataset

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(hf_hub_name)
    qrels = load_dataset(hf_hub_name + "-qrels")[eval_split]

    corpus = {e["id"]: {"text": e["text"]} for e in dataset["corpus"]}
    queries = {e["id"]: e["text"] for e in dataset["queries"]}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["qid"]][e["pid"]] = e["score"]

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs


class T2Retrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "T2Retrieval",
            "hf_hub_name": "C-MTEB/T2Retrieval",
            "reference": "https://arxiv.org/abs/2304.03679",
            "description": "T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "8731a845f1bf500a4f111cf1070785c793d10e64",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class MMarcoRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MMarcoRetrieval",
            "hf_hub_name": "C-MTEB/MMarcoRetrieval",
            "reference": "https://github.com/unicamp-dl/mMARCO",
            "description": "mMARCO is a multilingual version of the MS MARCO passage ranking dataset",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "539bbde593d947e2a124ba72651aafc09eb33fc2",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class DuRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "DuRetrieval",
            "hf_hub_name": "C-MTEB/DuRetrieval",
            "reference": "https://aclanthology.org/2022.emnlp-main.357.pdf",
            "description": "A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "a1a333e290fe30b10f3f56498e3a0d911a693ced",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class CovidRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CovidRetrieval",
            "hf_hub_name": "C-MTEB/CovidRetrieval",
            "reference": "https://aclanthology.org/2022.emnlp-main.357.pdf",
            "description": "COVID-19 news articles",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "1271c7809071a13532e05f25fb53511ffce77117",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class CmedqaRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "CmedqaRetrieval",
            "hf_hub_name": "C-MTEB/CmedqaRetrieval",
            "reference": "https://aclanthology.org/2022.emnlp-main.357.pdf",
            "description": "Online medical consultation text",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "cd540c506dae1cf9e9a59c3e06f42030d54e7301",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class EcomRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "EcomRetrieval",
            "hf_hub_name": "C-MTEB/EcomRetrieval",
            "reference": "https://arxiv.org/abs/2203.03367",
            "description": "Passage retrieval dataset collected from Alibaba search engine systems in ecom domain",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "687de13dc7294d6fd9be10c6945f9e8fec8166b9",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class MedicalRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MedicalRetrieval",
            "hf_hub_name": "C-MTEB/MedicalRetrieval",
            "reference": "https://arxiv.org/abs/2203.03367",
            "description": "Passage retrieval dataset collected from Alibaba search engine systems in medical domain",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "2039188fb5800a9803ba5048df7b76e6fb151fc6",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True


class VideoRetrieval(AbsTaskRetrieval):
    metadata = 

@property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "VideoRetrieval",
            "hf_hub_name": "C-MTEB/VideoRetrieval",
            "reference": "https://arxiv.org/abs/2203.03367",
            "description": "Passage retrieval dataset collected from Alibaba search engine systems in video domain",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
            "revision": "58c2597a5943a2ba48f4668c3b90d796283c5639",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True
