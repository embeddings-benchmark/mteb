import re
import pandas as pd
from huggingface_hub import create_repo, upload_file

########################
# Cleanup queries data #
########################

# load dataset
alloprof_queries = pd.read_csv("alloprof/data/alloprof.csv")

# remove non-queries
alloprof_queries = alloprof_queries[alloprof_queries["is_query"]]

# remove nans in text
alloprof_queries = alloprof_queries[~alloprof_queries["text"].isna()]

# most data flagged as language "en" are actually french. We je remove english ones
# by matching specifig words
alloprof_queries = alloprof_queries[~(
    (alloprof_queries["text"].str.lower().str.startswith("hi"))\
    | (alloprof_queries["text"].str.lower().str.startswith("hello"))\
    | (alloprof_queries["text"].str.lower().str.startswith("how"))\
    | (alloprof_queries["text"].str.lower().str.startswith("i "))\
        )]

# only keep queries with french relevant documents
alloprof_queries = alloprof_queries[
    (~alloprof_queries["relevant"].isna())\
    & (alloprof_queries["relevant"].str.endswith("-fr"))
    ]

# remove queries with url in text because question relies on picture
alloprof_queries = alloprof_queries[
    ~alloprof_queries["text"].str.contains("https://www.alloprof.qc.ca")
    ]

# split multiple relevant docs and remove -fr suffix on id
def parse_relevant_ids(row):
    row = row.split(";")
    row = [r[:-3] for r in row if r.endswith("-fr")]
    return row

alloprof_queries["relevant"] = alloprof_queries["relevant"].apply(parse_relevant_ids)

# only keep useful columns
alloprof_queries = alloprof_queries[["id", "text", "answer", "relevant", "subject"]]

##########################
# Cleanup documents data #
##########################

alloprof_docs = pd.read_json("alloprof/data/pages/page-content-fr.json")

# Remove Nans in data
alloprof_docs = alloprof_docs[~alloprof_docs["data"].isna()]

# parse dataset
def parse_row(row):
    return [row['file']["uuid"], row['file']["title"], row['file']["topic"]]


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
    text = ' '.join(text)
    text = re.sub('<[^<]+?>', '', text)
    text = text.replace("&nbsp;", " ").replace("\u200b", "")
    text = re.sub('\s{2,}', ' ', text)

    return text

parsed_df = alloprof_docs["data"].apply(parse_row)
alloprof_docs[["uuid", "title", "topic"]] = parsed_df.tolist()
alloprof_docs["text"] = alloprof_docs["data"].apply(get_text)

# remove unnecessary columns
alloprof_docs = alloprof_docs[["uuid", "title", "topic", "text"]]

#################################
# Check sanity and upload to HF #
#################################

# check that all relevant docs mentioned in queries are in docs dataset
relevants = alloprof_queries["relevant"].tolist()
relevants = {i for j in relevants for i in j} # flatten list and get uniques
assert relevants.issubset(alloprof_docs["uuid"].tolist()),\
"Some relevant document of queries are not present in the corpus"

# create HF repo
repo_id = "lyon-nlp/alloprof"
create_repo(repo_id, repo_type="dataset")

# save datasets as json
alloprof_queries.to_json("queries.jsonl", orient="records", lines=True)
alloprof_docs.to_json("documents.jsonl", orient="records", lines=True)

upload_file(
    path_or_fileobj="queries.jsonl",
    path_in_repo="queries.jsonl",
    repo_id=repo_id,
    repo_type="dataset"
)
upload_file(
    path_or_fileobj="documents.jsonl",
    path_in_repo="documents.jsonl",
    repo_id=repo_id,
    repo_type="dataset"
)