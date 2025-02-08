from __future__ import annotations

from mteb.tasks.Retrieval.eng.LoTTERetrieval import LoTTERetrieval

task = LoTTERetrieval()
data = task.load_data(eval_splits=["test"])

print(
    "Loaded Queries:", list(data["queries"]["test"].keys())[:5]
)  # Show sample queries
print("Loaded Corpus:", list(data["corpus"]["test"].keys())[:5])  # Show sample corpus
print(
    "Loaded Relevant Docs:", list(data["relevant_docs"]["test"].keys())[:5]
)  # Show sample relevant docs
