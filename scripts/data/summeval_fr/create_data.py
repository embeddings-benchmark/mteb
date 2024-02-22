import json
import os 

from huggingface_hub import create_repo, upload_file

import datasets
import requests

# API key DeepL to set before running the script with the command 'export DEEPL_API_KEY=***'
DEEPL_API_KEY= os.environ.get('DEEPL_AUTH_KEY')
ORGANIZATION = "lyon-nlp/"
REPO_NAME = "summarization-summeval-fr-p2p"
SAVE_PATH = "test.json"
HEADERS = {
    'Authorization': f"DeepL-Auth-Key {DEEPL_API_KEY}",
    'Content-Type': 'application/x-www-form-urlencoded',
}


def translate_with_deepl(text: str) -> str:
    data = {
        'text': text,
        'target_lang': 'FR',
    }
    response = requests.post('https://api.deepl.com/v2/translate', headers=HEADERS, data=data)
    return response.json()['translations'][0]['text']


summeval = datasets.load_dataset("mteb/summeval")["test"]

trads = []

for line in summeval:
    trad = line
    trad["text"] = translate_with_deepl(line["text"])

    machine_summaries = []
    for machine_sum in line['machine_summaries']:
        machine_summaries.append(translate_with_deepl(machine_sum))
    trad['machine_summaries'] = machine_summaries

    human_summaries = []
    for human_sum in line['human_summaries']:
        human_summaries.append(translate_with_deepl(human_sum))
    trad['human_summaries'] = human_summaries
    trads.append(trad)

    with open(SAVE_PATH, "w", encoding='utf8') as final:
        json.dump(trads, final, ensure_ascii=False)


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
