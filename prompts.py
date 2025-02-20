import hashlib
import json

PROMPT_DICT = {
    "SciFact": """Claim: FILL_QUERY_HERE

A relevant passage would provide evidence that either **supports** or **refutes** this claim. A passage with any information on any related subpart should be relevant.""",

    "TRECCOVID": """FILL_QUERY_HERE If the article answers any part of the question it is relevant.""",

    "ArguAna": """FILL_QUERY_HERE\n\nGiven this claim, find documents that **refute** the claim""",

    "DBPedia": """FILL_QUERY_HERE

A relevant document would describe some connection to the entity above. It does not need to be directly relevant, only tangentially informational. Please mark as relevant any passages with even weak connections. I need to learn fast for my job, which means I need to understand each part individually.

Again remember, any connection means relevant even if indirect. So if it is not addressed, that is okay -- it does not need to be explicitly. 

Find me passages with any type of connection, including weak connections!!!!""",

    "FiQA2018": """FILL_QUERY_HERE Find a passage that would be a good answer from StackExchange.""",

    "NFCorpus": """Topic: FILL_QUERY_HERE

Given the above topic, I need to learn about all aspects of it. It does not need to be directly relevant, only tangentially informational. Please mark as relevant any passages with even weak connections. I need to learn fast for my job, which means I need to understand each part individually.

Again remember, any connection means relevant even if indirect. So if it is not addressed, that is okay -- it does not need to be explicitly. 

Find me passages with any type of connection, including weak connections!!!!""",

    "Touche2020": """FILL_QUERY_HERE **any** arguments for or against""",

    "SCIDOCS": """papers that could be cited in FILL_QUERY_HERE. Anything with even indirect relevance should be relevant. This includes papers in the same broader field of science""",

    "BrightRetrieval_aops": """Find different but similar math problems to FILL_QUERY_HERE\n\nA document is relevant if it uses the same class of functions and shares **any** overlapping techniques.""",

    "BrightRetrieval_theoremqa_questions": """Find a passage which uses the same mathematical process as this one: FILL_QUERY_HERE""",

    "BrightRetrieval_leetcode": """I want to find code that uses the same structure/approach to solve it as this one: FILL_QUERY_HERE

A passage with code that shares any similar approach or technique is relevant. I do not care about the goal or problem, just one that shares any part of the approach.""",

    "BrightRetrieval_pony": """I will use the programming language pony. Problem: FILL_QUERY_HERE

But to solve the problem above, I need to know things about pony. A passage is relevant if it contains docs that match **any** part (even basic parts) of the code I will have to write for the above program.""",

    "BrightRetrieval": """Can you find background information about the concepts used to answer the question:

FILL_QUERY_HERE

A passage is relevant if it contains background information about a **sub-concept** that someone might cite/link to when answering the above question."""

}

PROMPT_DICT["ClimateFEVER"] = PROMPT_DICT["SciFact"]
PROMPT_DICT["BrightRetrieval_theoremqa_theorems"] = PROMPT_DICT["BrightRetrieval_theoremqa_questions"]


def get_prompt(task_name, subtask_name: str = None):
    if subtask_name is not None:
        # if subtask is present, use that, otherwise use just the task name
        if f"{task_name}_{subtask_name}" in PROMPT_DICT:
            return PROMPT_DICT[f"{task_name}_{subtask_name}"]
        else: # default for subtask (e.g. BrightRetrieval)
            return PROMPT_DICT[task_name]
    elif task_name in PROMPT_DICT:
        # no subtask
        return PROMPT_DICT[task_name]
    else:
        return None


BEIR_DATASETS = {
    "ArguAna": "beir-runs/bm25/run.beir-v1.0.0-arguana-flat.json",
    # "CQADupstackRetrieval": "beir-runs/bm25/run.beir-v1.0.0-cqadupstack-flat.trec",
    "ClimateFEVER": "beir-runs/bm25/run.beir-v1.0.0-climate-fever-flat.json",
    "DBPedia": "beir-runs/bm25/run.beir-v1.0.0-dbpedia-entity-flat.json",
    "FEVER": "beir-runs/bm25/run.beir-v1.0.0-fever-flat.json",
    "FiQA2018": "beir-runs/bm25/run.beir-v1.0.0-fiqa-flat.json",
    "HotpotQA": "beir-runs/bm25/run.beir-v1.0.0-hotpotqa-flat.json",
    "NFCorpus": "beir-runs/bm25/run.beir-v1.0.0-nfcorpus-flat.json",
    "NQ": "beir-runs/bm25/run.beir-v1.0.0-nq-flat.json",
    "QuoraRetrieval": "beir-runs/bm25/run.beir-v1.0.0-quora-flat.json",
    "SCIDOCS": "beir-runs/bm25/run.beir-v1.0.0-scidocs-flat.json",
    "SciFact": "beir-runs/bm25/run.beir-v1.0.0-scifact-flat.json",
    "TRECCOVID": "beir-runs/bm25/run.beir-v1.0.0-trec-covid-flat.json",
    "Touche2020": "beir-runs/bm25/run.beir-v1.0.0-webis-touche2020-flat.json",
}


# make a prompt hash, so that we know what we used for a given run
# append to the output prompt_hash.txt file

# read previous hashes
previous_hashes = set()
with open("prompt_hash.txt", "r") as f:
    for line in f:
        prompt_name, hash = line.strip().split("===")
        previous_hashes.add(hash.strip())


f_out = open("prompt_hash.txt", "a")
for prompt_name, prompt in PROMPT_DICT.items():
    hash_prompt = hashlib.sha256(prompt.encode()).hexdigest()
    if hash_prompt not in previous_hashes:
        f_out.write(f"{prompt_name}=== {hash_prompt}\n")
f_out.close()


def validate_json(file_path: str) -> bool:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # assert there are string keys and that within that a dict of key -> float values
        for key in data:
            assert isinstance(key, str), f"Key is not a string: {key}"
            assert isinstance(data[key], dict), f"Data is not a dict: {data[key]}"
            for inner_key, inner_value in data[key].items():
                assert isinstance(inner_key, str), f"Inner key is not a string: {inner_key}"
                assert isinstance(inner_value, float), f"Inner value is not a float: {inner_value}"
        return True
    except Exception as e:
        print(e)
        return False