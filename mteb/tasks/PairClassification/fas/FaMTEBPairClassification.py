from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CExaPPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CExaPPC",
        description="ExaPPC is a large paraphrase corpus consisting of monolingual sentence-level paraphrases using different sources.",
        reference="https://github.com/exaco/exappc",
        dataset={
            "path": "PNLPhub/C-ExaPPC",
            "revision": "68a0ff474739367a36c8066ee04802a65aefc117",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=["Social", "Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        self.dataset = self.dataset.map(
            lambda example: {"label": 1 if example["label"] == "paraphrase" else 0}
        )
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sentence1"],
                    "sentence2": self.dataset[split]["sentence2"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset


class SynPerChatbotRAGFAQPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGFAQPC",
        description="Synthetic Persian Chatbot RAG FAQ Pair Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-faq-pair-classification",
            "revision": "2128d809e27ab8528906e2231f8e824516fb8e5a",
        },
        type="PairClassification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sent1"][0],
                    "sentence2": self.dataset[split]["sent2"][0],
                    "labels": self.dataset[split]["labels"][0],
                }
            ]
        self.dataset = _dataset


class FarsiParaphraseDetection(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FarsiParaphraseDetection",
        description="Farsi Paraphrase Detection",
        reference="https://huggingface.co/datasets/alighasemi/farsi_paraphrase_detection",
        dataset={
            "path": "alighasemi/farsi_paraphrase_detection",
            "revision": "c8129741af418d9ae43cfc1fc4f285704e26035f",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sentence1"],
                    "sentence2": self.dataset[split]["sentence2"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset


class SynPerTextKeywordsPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SynPerTextKeywordsPC",
        description="Synthetic Persian Text Keywords Pair Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-text-keyword-pair-classification",
            "revision": "ea9a840cb163b415cc70b2f7adf2554feae159dc",
        },
        type="PairClassification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web", "News", "Religious", "Blog"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sent1"][0],
                    "sentence2": self.dataset[split]["sent2"][0],
                    "labels": self.dataset[split]["labels"][0],
                }
            ]
        self.dataset = _dataset


class SynPerQAPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SynPerQAPC",
        description="Synthetic Persian QA Pair Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-qa-pair-classification",
            "revision": "d1b62ef31bebbb48ae01867993a1e583c2ce7d93",
        },
        type="PairClassification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web", "News", "Religious", "Blog"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sent1"][0],
                    "sentence2": self.dataset[split]["sent2"][0],
                    "labels": self.dataset[split]["labels"][0],
                }
            ]
        self.dataset = _dataset


class ParsinluEntail(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ParsinluEntail",
        description="A Persian textual entailment task (deciding sent1 entails sent2). The questions are partially translated from the SNLI dataset and partially generated by expert annotators.",
        reference="https://github.com/persiannlp/parsinlu",
        dataset={
            "path": "persiannlp/parsinlu_entailment",
            "revision": "c49b2d8fa0d6476520695c52207690b7ec854043",
            "trust_remote_code": True,
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        self.dataset = self.dataset.filter(lambda x: x["label"] != "n")
        self.dataset = self.dataset.map(
            lambda example: {"label": 1 if example["label"] == "e" else 0}
        )
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sent1"],
                    "sentence2": self.dataset[split]["sent2"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset


class ParsinluQueryParaphPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ParsinluQueryParaphPC",
        description="A Persian query paraphrasng task (deciding whether two questions are paraphrases of each other). The questions are partially generated from Google auto-complete, and partially translated from the Quora paraphrasing dataset.",
        reference="https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing",
        dataset={
            "path": "persiannlp/parsinlu_query_paraphrasing",
            "revision": "ec675bb3ac50c1a52317c101fe1d724b4601f47a",
            "trust_remote_code": True,
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

    def dataset_transform(self):
        _dataset = {}
        self.dataset = self.dataset.map(
            lambda example: {"label": 1 if example["label"] == "1" else 0}
        )
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["q1"],
                    "sentence2": self.dataset[split]["q2"],
                    "labels": self.dataset[split]["label"],
                }
            ]
        self.dataset = _dataset
