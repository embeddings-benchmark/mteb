import datasets
import jsonlines
import gzip
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

np.random.seed(28042000)

d = datasets.load_dataset('mteb/raw_arxiv')['train']
# d = d.select(range(1000))

main_categories = [
    'cs',
    'econ',
    'math',
    'eess',
    'q-bio',
    'q-fin',
    'stat',
    'astro-ph',
    'physics',
    'cond-mat',
    'quant-ph',
    'hep'
]

def cluster_stats(labels):
    (unique, counts) = np.unique(labels, return_counts=True)
    for u,c in zip(unique, counts):
        print(u, c)

def get_text(record, type='s2s'):
    if type == 's2s':
        return record['title']
    elif type == 'p2p':
        return record['title'] + ' ' + record['abstract']
    raise ValueError

def match_main_category(category):
    for main_cat in main_categories:
        if main_cat in category:
            return main_cat
    return 'other'

def main_category(category_string):
    category_list = category_string.split(' ')
    category_list = list(map(match_main_category, category_list))
    main_cat = Counter(category_list).most_common(1)[0][0]
    return main_cat

def sub_category(category_string):
    category_list = category_string.split(' ')
    return category_list[0]

main_cats = []
sub_cats = []
for paper in tqdm(d):
    main_cats.append(main_category(paper['categories']))
    sub_cats.append(sub_category(paper['categories']))

main_cats = np.array(main_cats)
sub_cats = np.array(sub_cats)

split_size = 25000
split_number = 10
indices = np.arange(len(main_cats))

splits = []

task_type = 'p2p'

# Coarse splits
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item, task_type) for item in subset]
    labels = main_cats[current_indices].tolist()
    splits.append({'sentences': text, 'labels': labels})

# Fine grained splits
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item, task_type) for item in subset]
    labels = sub_cats[current_indices].tolist()
    splits.append({'sentences': text, 'labels': labels})

# Fine grained splits inside sub categories
for main_cat in tqdm(main_categories):
    indices = np.argwhere(main_cats == main_cat).flatten()
    if len(indices) == 0:
        print(f'No papers in {main_cat}')
        continue
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item, task_type) for item in subset]
    labels = sub_cats[current_indices].tolist()
    splits.append({'sentences': text, 'labels': labels})

repository = 'arxiv-clustering-p2p'
with jsonlines.open(f'{repository}/test.jsonl', 'w') as f_out:
    f_out.write_all(splits)
# Compress
with open(f'{repository}/test.jsonl', 'rb') as f_in:
    with gzip.open(f'{repository}/test.jsonl.gz', 'wb') as f_out:
        f_out.writelines(f_in)
# Remove uncompressed file
os.remove(f'{repository}/test.jsonl')