import os

from huggingface_hub import create_repo, upload_file

import pandas as pd
import requests


MAX_OUTPUT = 10000
NB_RESULTS = 100000
ORGANIZATION = "lyon-nlp/"
REPO_NAME = "clustering-hal-s2s"
SAVE_PATH = "test.jsonl"

df_papers = pd.DataFrame(columns=["hal_id","title","domain"])


start_index = 0
while start_index < NB_RESULTS:
    response = requests.request(
        "GET",
        f"https://api.archives-ouvertes.fr/search/?q=*:*&wt=json&fl=halId_s,title_s,level0_domain_s&fq=language_s:fr&fq=submittedDateY_i:[2000%20TO%20*]&rows={MAX_OUTPUT}&start={start_index}",
    )
    if "response" in response.json() :
        papers = response.json()["response"]["docs"]
        for paper in papers :
            if ("title_s" in paper) and ("level0_domain_s" in paper):
                paper_info = {
                    "hal_id": paper["halId_s"],
                    "title": paper["title_s"][0],
                    "domain": paper["level0_domain_s"][0]
                }
                df_papers = pd.concat([df_papers, pd.DataFrame([paper_info])], ignore_index=True)
    start_index += MAX_OUTPUT

df_papers = df_papers.drop_duplicates()
df_papers.to_json(SAVE_PATH, orient="records", lines=True)

create_repo(
    ORGANIZATION + REPO_NAME, 
    repo_type="dataset"
)

upload_file(
    path_or_fileobj=SAVE_PATH,
    path_in_repo=SAVE_PATH,
    repo_id=ORGANIZATION + REPO_NAME,
    repo_type="dataset"
)
os.system(f"rm {SAVE_PATH}")
