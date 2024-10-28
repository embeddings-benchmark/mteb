from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 2048


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(".")
    return record


class ArXivHierarchicalClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringP2P",
        description="Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "0bbdb47bcbe3a90093699aefeed338a0f28a7ee8",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": N_SAMPLES},
            "test": {
                "num_samples": 2048,
                "average_text_length": 1008.439453125,
                "average_labels_per_text": 1.46337890625,
                "unique_labels": 129,
                "labels": {
                    "cs": {"count": 356},
                    "math": {"count": 381},
                    "OC": {"count": 11},
                    "hep-lat": {"count": 13},
                    "hep": {"count": 98},
                    "astro-ph": {"count": 213},
                    "eess": {"count": 76},
                    "quant-ph": {"count": 135},
                    "DC": {"count": 5},
                    "cond-mat": {"count": 274},
                    "hep-th": {"count": 66},
                    "SP": {"count": 33},
                    "hep-ph": {"count": 69},
                    "FA": {"count": 6},
                    "nucl-th": {"count": 17},
                    "q-bio": {"count": 80},
                    "HE": {"count": 22},
                    "HC": {"count": 2},
                    "stat": {"count": 60},
                    "ML": {"count": 16},
                    "IV": {"count": 13},
                    "stat-mech": {"count": 47},
                    "DS": {"count": 14},
                    "ME": {"count": 12},
                    "CC": {"count": 2},
                    "mtrl-sci": {"count": 22},
                    "PE": {"count": 16},
                    "NT": {"count": 11},
                    "SC": {"count": 6},
                    "AG": {"count": 13},
                    "physics": {"count": 81},
                    "ins-det": {"count": 9},
                    "GA": {"count": 18},
                    "BM": {"count": 6},
                    "GN": {"count": 17},
                    "NA": {"count": 15},
                    "app-ph": {"count": 7},
                    "RT": {"count": 6},
                    "other": {"count": 37},
                    "soft": {"count": 15},
                    "CO": {"count": 33},
                    "supr-con": {"count": 21},
                    "chem-ph": {"count": 3},
                    "DM": {"count": 2},
                    "MN": {"count": 12},
                    "q-fin": {"count": 27},
                    "PM": {"count": 2},
                    "AP": {"count": 27},
                    "gr-qc": {"count": 15},
                    "quant-gas": {"count": 8},
                    "mes-hall": {"count": 33},
                    "IT": {"count": 19},
                    "SI": {"count": 6},
                    "SG": {"count": 3},
                    "bio-ph": {"count": 2},
                    "SR": {"count": 16},
                    "soc-ph": {"count": 5},
                    "hep-ex": {"count": 15},
                    "DG": {"count": 11},
                    "NE": {"count": 5},
                    "CR": {"count": 6},
                    "CL": {"count": 12},
                    "RM": {"count": 3},
                    "econ": {"count": 17},
                    "nlin": {"count": 5},
                    "PS": {"count": 1},
                    "LG": {"count": 26},
                    "QA": {"count": 9},
                    "str-el": {"count": 26},
                    "CV": {"count": 34},
                    "MF": {"count": 6},
                    "IM": {"count": 7},
                    "EM": {"count": 6},
                    "TH": {"count": 5},
                    "PR": {"count": 20},
                    "AT": {"count": 4},
                    "OA": {"count": 4},
                    "CP": {"count": 6},
                    "LO": {"count": 14},
                    "flu-dyn": {"count": 6},
                    "atom-ph": {"count": 8},
                    "class-ph": {"count": 1},
                    "SY": {"count": 20},
                    "IR": {"count": 1},
                    "plasm-ph": {"count": 8},
                    "CE": {"count": 2},
                    "AO": {"count": 1},
                    "comp-ph": {"count": 3},
                    "optics": {"count": 12},
                    "MG": {"count": 4},
                    "ST": {"count": 6},
                    "nucl-ex": {"count": 6},
                    "CY": {"count": 9},
                    "ao-ph": {"count": 2},
                    "DB": {"count": 1},
                    "math-ph": {"count": 10},
                    "NC": {"count": 13},
                    "GT": {"count": 11},
                    "TO": {"count": 2},
                    "AI": {"count": 9},
                    "NI": {"count": 2},
                    "gen-ph": {"count": 4},
                    "OT": {"count": 4},
                    "SD": {"count": 2},
                    "dis-nn": {"count": 4},
                    "RO": {"count": 7},
                    "CA": {"count": 6},
                    "FL": {"count": 1},
                    "SE": {"count": 5},
                    "EP": {"count": 9},
                    "hist-ph": {"count": 1},
                    "QM": {"count": 9},
                    "ed-ph": {"count": 2},
                    "GR": {"count": 4},
                    "MS": {"count": 1},
                    "CD": {"count": 1},
                    "ET": {"count": 1},
                    "acc-ph": {"count": 5},
                    "AC": {"count": 2},
                    "OH": {"count": 1},
                    "EC": {"count": 2},
                    "DL": {"count": 1},
                    "AS": {"count": 3},
                    "geo-ph": {"count": 2},
                    "CG": {"count": 3},
                    "CB": {"count": 1},
                    "AR": {"count": 1},
                    "TR": {"count": 1},
                    "atm-clus": {"count": 1},
                },
            },
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.dataset.map(split_labels)
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=N_SAMPLES, seed=self.seed
        )["test"]


class ArXivHierarchicalClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringS2S",
        description="Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-s2s",
            "revision": "b73bd54100e5abfa6e3a23dcafb46fe4d2438dc3",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),  # 1991-01-01 is the first arxiv paper
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": N_SAMPLES},
            "avg_character_length": {"test": 1009.98},
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )
            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.dataset.map(split_labels)
        self.dataset["test"] = self.dataset["test"].train_test_split(
            test_size=N_SAMPLES, seed=self.seed
        )["test"]
