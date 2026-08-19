"""Construction script for ADVANCE-A2I and ADVANCE-I2A.

Reshapes blanchon/ADVANCE (5,075 rows of image/audio/label pairs) into
BEIR-style corpus/queries/qrels layout, one push per retrieval direction.
Each location contributes exactly one image and one audio recording, so
`id` gives clean 1:1 instance-level ground truth for qrels.
"""

from datasets import Dataset, DatasetDict, load_dataset

ds = load_dataset("blanchon/ADVANCE", split="train")
n = len(ds)
ids = [f"loc{i}" for i in range(n)]
ds = ds.add_column("id", ids)

# A2I: query = audio, corpus = image
a2i_queries = ds.select_columns(["id", "audio"])
a2i_corpus = ds.select_columns(["id", "image"])
a2i_qrels = Dataset.from_dict({"query-id": ids, "corpus-id": ids, "score": [1] * n})

DatasetDict({"test": a2i_queries}).push_to_hub(
    "yaswanth169/ADVANCE-A2I", config_name="queries"
)
DatasetDict({"test": a2i_corpus}).push_to_hub(
    "yaswanth169/ADVANCE-A2I", config_name="corpus"
)
DatasetDict({"test": a2i_qrels}).push_to_hub(
    "yaswanth169/ADVANCE-A2I", config_name="qrels"
)

# I2A: query = image, corpus = audio (mirror of the above)
i2a_queries = ds.select_columns(["id", "image"])
i2a_corpus = ds.select_columns(["id", "audio"])
i2a_qrels = Dataset.from_dict({"query-id": ids, "corpus-id": ids, "score": [1] * n})

DatasetDict({"test": i2a_queries}).push_to_hub(
    "yaswanth169/ADVANCE-I2A", config_name="queries"
)
DatasetDict({"test": i2a_corpus}).push_to_hub(
    "yaswanth169/ADVANCE-I2A", config_name="corpus"
)
DatasetDict({"test": i2a_qrels}).push_to_hub(
    "yaswanth169/ADVANCE-I2A", config_name="qrels"
)
