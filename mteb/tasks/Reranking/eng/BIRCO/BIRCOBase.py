from __future__ import annotations
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
import json
from pathlib import Path

class BIRCOBase(AbsTaskReranking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        # Load dataset.json from the HF hub directory specified in metadata.
        # NOTE: Ensure the dataset is available at the path provided in metadata.
        dataset_path = Path(self.metadata.dataset["path"]) / "dataset.json"
        with open(dataset_path) as f:
            data = json.load(f)
        
        # Convert BIRCO format to MTEB format:
        # Assumes data contains keys "query", "qrel", and "corpus".
        self.samples = []
        for qid, qrel in data["qrel"].items():
            self.samples.append({
                "query": data["query"][qid],
                "positive": [did for did, score in qrel.items() if score >= 1],
                "negative": [did for did, score in qrel.items() if score < 1],
                "docs": {did: data["corpus"][did] for did in qrel.keys()},
                "scores": list(qrel.values())
            })
        
        # Create a test split from all samples
        self.dataset = {"test": self.samples}
        self.data_loaded = True

        # Overwrite 'docs' with a list of document texts for evaluation.
        for sample in self.samples:
            sample["docs"] = [data["corpus"][did] for did in sample["positive"] + sample["negative"]]
