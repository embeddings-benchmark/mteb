import json
import pandas as pd
from huggingface_hub import create_repo, upload_file
from huggingface_hub.utils._errors import HfHubHTTPError

### Process articles ###
articles = pd.read_json("./syntec.json")
articles["content"] = articles["content"].str.replace("[En vigueur] ", "")
articles["content"] = articles["content"].str.replace("[En vigueur]\n", "")

### Process queries ###
queries = pd.read_excel("./Syntec.xlsx")
queries.drop(["Personne", "Réponse"], axis=1, inplace=True)

### Test ###
assert queries["Article"].isin(articles["id"]).all(),\
"Some queries's target documents are missing in documents corpus"

# create HF repo
repo_id = "lyon-nlp/mteb-fr-retrieval-syntec-s2p"
try:
    create_repo(repo_id, repo_type="dataset")
except HfHubHTTPError as e:
    print("HF repo already exist")

# save datasets as json
queries.to_json("queries.json", orient="records")
articles.to_json("documents.json", orient="records")

upload_file(path_or_fileobj="queries.json", path_in_repo="queries.json", repo_id=repo_id, repo_type="dataset")
upload_file(path_or_fileobj="documents.json", path_in_repo="documents.json", repo_id=repo_id, repo_type="dataset")
