import os

import pandas as pd


df_news = pd.read_csv("scripts/mind/data/MINDsmall_train/news.tsv", sep="\t", header=None)
df_news = df_news[[0, 3]]
df_news.columns = ["id", "text"]
df_news.index = df_news["id"]


df_behaviours = pd.read_csv("scripts/mind/data/MINDsmall_train/behaviors.tsv", sep="\t", header=None)
df_behaviours = df_behaviours[[0, 3, 4]]
df_behaviours.columns = ["id", "query", "data"]
df_behaviours.dropna(inplace=True)

# df_behaviours = df_behaviours.iloc[:10]


def proc_row(row):
    docs = row["data"].split()
    positives, negatives = [], []
    for doc in docs:
        idx, label = doc.split("-")
        if label == "1":
            positives.append(idx)
        elif label == "0":
            negatives.append(idx)
        else:
            raise Exception("Unknown label: {}".format(label))
    row["positive"] = [df_news.loc[idx]["text"] for idx in positives]
    row["negative"] = [df_news.loc[idx]["text"] for idx in negatives]
    queries = row["query"].split()
    row["query"] = [df_news.loc[idx]["text"] for idx in queries]
    return row


df_behaviours = df_behaviours.apply(proc_row, axis=1)
df_behaviours.drop(columns=["data"], inplace=True)
print(df_behaviours)

path = "mind"
os.makedirs(path, exist_ok=True)
df_behaviours.to_json(f"{path}/train.jsonl", orient="records", lines=True)
