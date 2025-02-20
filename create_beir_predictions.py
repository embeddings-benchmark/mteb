import json
from collections import defaultdict
import glob
from tqdm import tqdm

def read_trec_to_json(file: str) -> dict:
    """Read a TREC file and return a dictionary of the results in the format of MTEB"""
    # input is 1 Q0 43385013 3 5.157500 Anserini
    predictions = defaultdict(dict)
    with open(file, "r") as f:
        for line in f:
            parts = line.split()
            query_id = parts[0]
            doc_id = parts[2]
            score = parts[4]
            predictions[query_id][doc_id] = float(score)
    return predictions


#  for every file in beir-runs/bm25/*.trec, read it and convert it to a json file
for file in tqdm(list(glob.glob("beir-runs/bm25/*.trec"))):
    predictions = read_trec_to_json(file)
    with open(file.replace(".trec", ".json"), "w") as f:
        json.dump(predictions, f)
