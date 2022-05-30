import pandas as pd
import json
import glob
import os
from huggingface_hub import create_repo, upload_file
import matplotlib.pyplot as plt

# First, we scrape the data from the original CSV file
# python3 -m semeval_8_2022_ia_downloader.cli --links_file=semeval-2022_task8_train-data_batch.csv --dump_dir=train
# python3 -m semeval_8_2022_ia_downloader.cli --links_file=final_evaluation_data.csv --dump_dir=test

# Prepare the data
scraped_jsons = glob.glob("train/*/*.json")
articles = {}
for path in scraped_jsons:
    with open(path) as json_file:
        data = json.load(json_file)
        if len(data["text"]) > 0 or "get the full access" not in data["title"].lower():
            articles[path.split("/")[-1][:-5]] = data["text"]

df = pd.read_csv("semeval-2022_task8_train-data_batch.csv")
# df = pd.read_csv("final_evaluation_data.csv")

df["lang"] = df.apply(lambda row: row["url1_lang"] + "-" + row["url2_lang"], axis=1)
df = df[["lang", "pair_id", "Overall"]]
df[["sentence1", "sentence2"]] = df["pair_id"].str.split("_", expand=True)
df["sentence1"] = df["sentence1"].map(articles).str.strip()
df["sentence2"] = df["sentence2"].map(articles).str.strip()
print(len(df))
df = df.dropna()
print(len(df))

df.hist(column="Overall", bins=100)
plt.title("Score histogram (train)")
plt.savefig("hist-train.png")

df = df.rename(
    columns={
        "Overall": "score",
        "pair_id": "id",
    }
)

repo_name = "sts22-crosslingual-sts"
create_repo(repo_name, organization="mteb", repo_type="dataset")

# save to {lang}/{split}.jsonl
for lang in df["lang"].unique():
    lang_df = df[df["lang"] == lang]
    lang_df = lang_df.drop(columns=["lang"])
    print(lang, len(lang_df))
    lang1, lang2 = lang.split("-")
    if lang1 == lang2:
        lang = lang1
    os.makedirs(lang, exist_ok=True)
    save_path = lang + "/train.jsonl"
    lang_df.to_json(save_path, orient="records", lines=True)

    upload_file(path_or_fileobj=save_path, path_in_repo=save_path, repo_id="mteb/" + repo_name, repo_type="dataset")
    os.system(f"rm {save_path}")
