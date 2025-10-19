from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
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
        bibtex_citation=r"""
@inproceedings{9786243,
  author = {Sadeghi, Reyhaneh and Karbasi, Hamed and Akbari, Ahmad},
  booktitle = {2022 8th International Conference on Web Research (ICWR)},
  doi = {10.1109/ICWR54782.2022.9786243},
  keywords = {Data mining;Task analysis;Paraphrase Identification;Semantic Similarity;Deep Learning;Paraphrasing Corpora},
  number = {},
  pages = {168-175},
  title = {ExaPPC: a Large-Scale Persian Paraphrase Detection Corpus},
  volume = {},
  year = {2022},
}
""",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
            "path": "mteb/ParsinluEntail",
            "revision": "7fe32997bdf5f50bd710df8d0cec33ae2844b47f",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{khashabi2021parsinlusuitelanguageunderstanding,
  archiveprefix = {arXiv},
  author = {Daniel Khashabi and Arman Cohan and Siamak Shakeri and Pedram Hosseini and Pouya Pezeshkpour and Malihe Alikhani and Moin Aminnaseri and Marzieh Bitaab and Faeze Brahman and Sarik Ghazarian and Mozhdeh Gheini and Arman Kabiri and Rabeeh Karimi Mahabadi and Omid Memarrast and Ahmadreza Mosallanezhad and Erfan Noury and Shahab Raji and Mohammad Sadegh Rasooli and Sepideh Sadeghi and Erfan Sadeqi Azer and Niloofar Safi Samghabadi and Mahsa Shafaei and Saber Sheybani and Ali Tazarv and Yadollah Yaghoobzadeh},
  eprint = {2012.06154},
  primaryclass = {cs.CL},
  title = {ParsiNLU: A Suite of Language Understanding Challenges for Persian},
  url = {https://arxiv.org/abs/2012.06154},
  year = {2021},
}
""",
    )


class ParsinluQueryParaphPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ParsinluQueryParaphPC",
        description="A Persian query paraphrasng task (deciding whether two questions are paraphrases of each other). The questions are partially generated from Google auto-complete, and partially translated from the Quora paraphrasing dataset.",
        reference="https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing",
        dataset={
            "path": "mteb/ParsinluQueryParaphPC",
            "revision": "456643cfd055434b05aee7c914f031f4e9108bb2",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="max_ap",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{khashabi2021parsinlusuitelanguageunderstanding,
  archiveprefix = {arXiv},
  author = {Daniel Khashabi and Arman Cohan and Siamak Shakeri and Pedram Hosseini and Pouya Pezeshkpour and Malihe Alikhani and Moin Aminnaseri and Marzieh Bitaab and Faeze Brahman and Sarik Ghazarian and Mozhdeh Gheini and Arman Kabiri and Rabeeh Karimi Mahabadi and Omid Memarrast and Ahmadreza Mosallanezhad and Erfan Noury and Shahab Raji and Mohammad Sadegh Rasooli and Sepideh Sadeghi and Erfan Sadeqi Azer and Niloofar Safi Samghabadi and Mahsa Shafaei and Saber Sheybani and Ali Tazarv and Yadollah Yaghoobzadeh},
  eprint = {2012.06154},
  primaryclass = {cs.CL},
  title = {ParsiNLU: A Suite of Language Understanding Challenges for Persian},
  url = {https://arxiv.org/abs/2012.06154},
  year = {2021},
}
""",
    )
