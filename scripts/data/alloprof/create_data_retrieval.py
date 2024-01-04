from collections import Counter
import json
import re

import datasets
import pandas as pd
from huggingface_hub import create_repo, upload_file, hf_hub_download
from huggingface_hub.utils._errors import HfHubHTTPError

########################
# Cleanup queries data #
########################

# load dataset
dl_path = hf_hub_download(
    repo_id="antoinelb7/alloprof",
    filename="data/alloprof.csv",
    repo_type="dataset",
    revision="0faa90fee1ad1a6e3e461d7be49abf71488e6687"
    )
alloprof_queries = pd.read_csv(dl_path)

# remove non-queries
alloprof_queries = alloprof_queries[alloprof_queries["is_query"]]

# remove nans in text
alloprof_queries = alloprof_queries[~alloprof_queries["text"].isna()]

# most data flagged as language "en" are actually french. We je remove english ones
# by matching specifig words
alloprof_queries = alloprof_queries[
    ~(
        (alloprof_queries["text"].str.lower().str.startswith("hi"))
        | (alloprof_queries["text"].str.lower().str.startswith("hello"))
        | (alloprof_queries["text"].str.lower().str.startswith("how"))
        | (alloprof_queries["text"].str.lower().str.startswith("i "))
    )
]

# only keep queries with french relevant documents
alloprof_queries = alloprof_queries[
    (~alloprof_queries["relevant"].isna()) & (alloprof_queries["relevant"].str.endswith("-fr"))
]

# remove queries with url in text because question relies on picture
alloprof_queries = alloprof_queries[~alloprof_queries["text"].str.contains("https://www.alloprof.qc.ca")]


# split multiple relevant docs and remove -fr suffix on id
def parse_relevant_ids(row):
    row = row.split(";")
    row = [r[:-3] for r in row if r.endswith("-fr")]
    return row


alloprof_queries["relevant"] = alloprof_queries["relevant"].apply(parse_relevant_ids)


# Parse the answer
def parse_answer(row):
    try:
        row = json.loads(row)
        text = []
        for i in row:
            if type(i["insert"]) is not dict:
                text.append(i["insert"])
        text = "".join(text)
    except:
        text = row
    return text.replace("&nbsp;", " ").replace("\u200b", "").replace("\xa0", "")


alloprof_queries["answer"] = alloprof_queries["answer"].apply(parse_answer)

# only keep useful columns
alloprof_queries = alloprof_queries[["id", "text", "answer", "relevant", "subject"]]

# remove duplicate queries (same text)
alloprof_queries = alloprof_queries.drop_duplicates(subset=["text"], keep="first")

##########################
# Cleanup documents data #
##########################

# load dataset
dl_path = hf_hub_download(
    repo_id="antoinelb7/alloprof",
    filename="data/pages/page-content-fr.json",
    repo_type="dataset",
    revision="0faa90fee1ad1a6e3e461d7be49abf71488e6687"
    )
alloprof_docs = pd.read_json(dl_path)

# Remove Nans in data
alloprof_docs = alloprof_docs[~alloprof_docs["data"].isna()]

# parse dataset
def parse_row(row):
    return [row["file"]["uuid"], row["file"]["title"], row["file"]["topic"]]


def get_text(row):
    text = []
    for s in row["file"]["sections"]:
        for m in s["modules"]:
            if m["type"] == "blocSpecial":
                if m["subtype"] in ["definition", "exemple"]:
                    for sm in m["submodules"]:
                        if sm["type"] == "text":
                            text.append(sm["text"])
            elif m["type"] == "text":
                text.append(m["text"])
    text = " ".join(text)
    text = re.sub("<[^<]+?>", "", text)
    text = text.replace("&nbsp;", " ").replace("\u200b", "")
    text = re.sub("\s{2,}", " ", text)

    return text


parsed_df = alloprof_docs["data"].apply(parse_row)
alloprof_docs[["uuid", "title", "topic"]] = parsed_df.tolist()
alloprof_docs["text"] = alloprof_docs["data"].apply(get_text)

# remove unnecessary columns
alloprof_docs = alloprof_docs[["uuid", "title", "topic", "text"]]

################
# Post Process #
################

# check that all relevant docs mentioned in queries are in docs dataset
relevants = alloprof_queries["relevant"].tolist()
relevants = {i for j in relevants for i in j}  # flatten list and get uniques
assert relevants.issubset(
    alloprof_docs["uuid"].tolist()
), "Some relevant document of queries are not present in the corpus"

# convert to Dataset
alloprof_queries = datasets.Dataset.from_pandas(alloprof_queries)
alloprof_docs = datasets.Dataset.from_pandas(alloprof_docs)

# identify duplicate documents
# (duplicates are actually error documents,
# such as "fiche en construction", " ", ...
duplicate_docs = Counter(alloprof_docs["text"])
duplicate_docs = {k:v for k,v in duplicate_docs.items() if v > 1}

# for each text that is in duplicate...
for dup_text in duplicate_docs:
    # ...get the ids of docs that have that text
    duplicate_ids = [d["uuid"] for d in alloprof_docs if d["text"] == dup_text]
    # ...delete all the documents that have these ids from the corpus dataset
    alloprof_docs = alloprof_docs.filter(lambda x: x["uuid"] not in duplicate_ids)
    # ...delete them from the relevant documents in queries
    alloprof_queries = alloprof_queries.map(lambda x: {"relevant": [i for i in x["relevant"] if i not in duplicate_ids]})

# remove the queries that have no remaining relevant documents
alloprof_queries = alloprof_queries.filter(lambda x: len(x["relevant"]) > 0)

####################
# Upload to HF Hub #
####################

# create HF repo
repo_id = "lyon-nlp/alloprof"
try:
    create_repo(repo_id, repo_type="dataset")
except HfHubHTTPError as e:
    print("HF repo already exist")

# save datasets as json
alloprof_queries.to_pandas().to_json("queries.json", orient="records")
alloprof_docs.to_pandas().to_json("documents.json", orient="records")

upload_file(path_or_fileobj="queries.json", path_in_repo="queries.json", repo_id=repo_id, repo_type="dataset")
upload_file(path_or_fileobj="documents.json", path_in_repo="documents.json", repo_id=repo_id, repo_type="dataset")
