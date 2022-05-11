import datasets
import jsonlines
import gzip
import numpy as np
import os
from tqdm import tqdm

np.random.seed(28042000)

d = datasets.load_dataset("sentence-transformers/reddit-title-body", data_files=['reddit_title_text_2021.jsonl.gz'])['train']

# d = d.select(range(1000))

def cluster_stats(labels):
    (unique, counts) = np.unique(labels, return_counts=True)
    for u,c in zip(unique, counts):
        print(u, c)

def get_text(record):
    return record['title'] + ' ' + record['body']

split_size = 50000
split_number = 10
indices = np.arange(len(d))

splits = []

# Coarse splits 50k
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [x+' '+y for (x,y) in zip(subset['title'],subset['body'])]
    splits.append({'sentences': text, 'labels': subset['subreddit']})

split_size = 10000
split_number = 40

# Coarse splits 25k
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [x+' '+y for (x,y) in zip(subset['title'],subset['body'])]
    splits.append({'sentences': text, 'labels': subset['subreddit']})

repository = f'reddit-clustering-p2p'
with jsonlines.open(f'{repository}/test.jsonl', 'w') as f_out:
    f_out.write_all(splits)
# Compress
with open(f'{repository}/test.jsonl', 'rb') as f_in:
    with gzip.open(f'{repository}/test.jsonl.gz', 'wb') as f_out:
        f_out.writelines(f_in)
# Remove uncompressed file
os.remove(f'{repository}/test.jsonl')