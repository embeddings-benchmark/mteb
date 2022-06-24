import re
import gzip
import json 
from huggingface_hub import create_repo, upload_file

repo_name = "bucc-bitext-mining"
# create_repo(repo_name, organization="mteb", repo_type="dataset")

with open('bucc-data/zh-en/zh-en.training.zh','r') as f:
  sentence1=f.readlines()

with open('bucc-data/zh-en/zh-en.training.en','r') as f:
  sentence2=f.readlines()

with open('bucc-data/zh-en/zh-en.training.gold','r') as f:
  gold=f.readlines()

def process_sentence(x):
  x = re.sub('\n','',x).strip()
  return x.split('\t')[1]

def process_gold(x):
  id1, id2 = x.strip().split('\t')
  id1 = id1.split('-')[1]
  id2 = id2.split('-')[1]
  return int(id1), int(id2)

sentence1 = list(map(process_sentence, sentence1))
sentence2 = list(map(process_sentence, sentence2))
gold = list(map(process_gold, gold))

data = {'sentence1': sentence1, 'sentence2': sentence2, 'gold': gold}

with gzip.open('test.json.gz', 'wt', encoding='UTF-8') as zipfile:
    json.dump(data, zipfile)

upload_file(
    path_or_fileobj='test.json.gz', path_in_repo=f'zh-en/test.json.gz', repo_id=f'mteb/{repo_name}', repo_type='dataset'
)