---
language: en
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- mteb
datasets:
- s2orc
- flax-sentence-embeddings/stackexchange_xml
- ms_marco
- gooaq
- yahoo_answers_topics
- code_search_net
- search_qa
- eli5
- snli
- multi_nli
- wikihow
- natural_questions
- trivia_qa
- embedding-data/sentence-compression
- embedding-data/flickr30k-captions
- embedding-data/altlex
- embedding-data/simple-wiki
- embedding-data/QQP
- embedding-data/SPECTER
- embedding-data/PAQ_pairs
- embedding-data/WikiAnswers
pipeline_tag: sentence-similarity

model-index:
- name: all-MiniLM-L6-v2
  results:
  - dataset:
      config: default
      name: test_dataset
      revision: test
      split: test
      type: test
    metrics:
    - type: map_at_1
      value: 0.0
  - dataset:
      config: default
      name: MTEB BSARDRetrieval (default)
      revision: 5effa1b9b5fa3b0f9e12523e6e43e5f86a6e6d59
      split: test
      type: maastrichtlawtech/bsard
    metrics:
    - type: map_at_1
      value: 0.0
    - type: map_at_10
      value: 0.0
    - type: map_at_100
      value: 0.0
    - type: map_at_1000
      value: 0.006
    - type: map_at_3
      value: 0.0
    - type: map_at_5
      value: 0.0
    - type: mrr_at_1
      value: 0.0
    - type: mrr_at_10
      value: 0.0
    - type: mrr_at_100
      value: 0.0
    - type: mrr_at_1000
      value: 0.006
    - type: mrr_at_3
      value: 0.0
    - type: mrr_at_5
      value: 0.0
    - type: ndcg_at_1
      value: 0.0
    - type: ndcg_at_10
      value: 0.0
    - type: ndcg_at_100
      value: 0.0
    - type: ndcg_at_1000
      value: 0.21
    - type: ndcg_at_3
      value: 0.0
    - type: ndcg_at_5
      value: 0.0
    - type: precision_at_1
      value: 0.0
    - type: precision_at_10
      value: 0.0
    - type: precision_at_100
      value: 0.0
    - type: precision_at_1000
      value: 0.002
    - type: precision_at_3
      value: 0.0
    - type: precision_at_5
      value: 0.0
    - type: recall_at_1
      value: 0.0
    - type: recall_at_10
      value: 0.0
    - type: recall_at_100
      value: 0.0
    - type: recall_at_1000
      value: 1.802
    - type: recall_at_3
      value: 0.0
    - type: recall_at_5
      value: 0.0
    - type: main_score
      value: 0.0
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB BornholmBitextMining (default)
      revision: 3bc5cfb4ec514264fe2db5615fac9016f7251552
      split: test
      type: strombergnlp/bornholmsk_parallel
    metrics:
    - type: accuracy
      value: 36.0
    - type: f1
      value: 29.68132161955691
    - type: main_score
      value: 29.68132161955691
    - type: precision
      value: 27.690919913419915
    - type: recall
      value: 36.0
    task:
      type: BitextMining
  - dataset:
      config: ar
      name: MTEB STS22 (ar)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 5.006610360999117
    - type: cosine_spearman
      value: 22.63866797712348
    - type: manhattan_pearson
      value: 13.260328120447722
    - type: manhattan_spearman
      value: 22.340169287120716
    - type: euclidean_pearson
      value: 13.082283087945362
    - type: euclidean_spearman
      value: 22.63866797712348
    - type: main_score
      value: 22.63866797712348
    task:
      type: STS
  - dataset:
      config: de
      name: MTEB STS22 (de)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 26.596033966146116
    - type: cosine_spearman
      value: 31.044353994772354
    - type: manhattan_pearson
      value: 21.718468273577894
    - type: manhattan_spearman
      value: 31.197915595597696
    - type: euclidean_pearson
      value: 21.51728902500591
    - type: euclidean_spearman
      value: 31.044353994772354
    - type: main_score
      value: 31.044353994772354
    task:
      type: STS
  - dataset:
      config: de-en
      name: MTEB STS22 (de-en)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 47.54531236654512
    - type: cosine_spearman
      value: 44.038685024247606
    - type: manhattan_pearson
      value: 48.10217367438755
    - type: manhattan_spearman
      value: 44.4428504653391
    - type: euclidean_pearson
      value: 48.46975590869453
    - type: euclidean_spearman
      value: 44.038685024247606
    - type: main_score
      value: 44.038685024247606
    task:
      type: STS
  - dataset:
      config: de-fr
      name: MTEB STS22 (de-fr)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 26.472844763068938
    - type: cosine_spearman
      value: 30.067587482078228
    - type: manhattan_pearson
      value: 25.808959063835424
    - type: manhattan_spearman
      value: 27.996294873002153
    - type: euclidean_pearson
      value: 26.87230792075073
    - type: euclidean_spearman
      value: 30.067587482078228
    - type: main_score
      value: 30.067587482078228
    task:
      type: STS
  - dataset:
      config: de-pl
      name: MTEB STS22 (de-pl)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 7.026566971631159
    - type: cosine_spearman
      value: 4.9270565599404135
    - type: manhattan_pearson
      value: 9.01762174854638
    - type: manhattan_spearman
      value: 7.359790736410993
    - type: euclidean_pearson
      value: 6.729027056926625
    - type: euclidean_spearman
      value: 4.9270565599404135
    - type: main_score
      value: 4.9270565599404135
    task:
      type: STS
  - dataset:
      config: en
      name: MTEB STS22 (en)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 67.09882753030891
    - type: cosine_spearman
      value: 67.21465212910987
    - type: manhattan_pearson
      value: 68.41388868877884
    - type: manhattan_spearman
      value: 67.83615682571867
    - type: euclidean_pearson
      value: 68.21374069918403
    - type: euclidean_spearman
      value: 67.21465212910987
    - type: main_score
      value: 67.21465212910987
    task:
      type: STS
  - dataset:
      config: es
      name: MTEB STS22 (es)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 44.33815143022264
    - type: cosine_spearman
      value: 54.77772552456677
    - type: manhattan_pearson
      value: 49.29424073081744
    - type: manhattan_spearman
      value: 55.259696552690954
    - type: euclidean_pearson
      value: 48.483578263920634
    - type: euclidean_spearman
      value: 54.77772552456677
    - type: main_score
      value: 54.77772552456677
    task:
      type: STS
  - dataset:
      config: es-en
      name: MTEB STS22 (es-en)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 49.93601240112664
    - type: cosine_spearman
      value: 53.41895837272506
    - type: manhattan_pearson
      value: 49.86265183075983
    - type: manhattan_spearman
      value: 53.10065931046005
    - type: euclidean_pearson
      value: 50.16469746986203
    - type: euclidean_spearman
      value: 53.41895837272506
    - type: main_score
      value: 53.41895837272506
    task:
      type: STS
  - dataset:
      config: es-it
      name: MTEB STS22 (es-it)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 42.568537775842245
    - type: cosine_spearman
      value: 44.2699366594503
    - type: manhattan_pearson
      value: 43.954212787242284
    - type: manhattan_spearman
      value: 44.32159550471527
    - type: euclidean_pearson
      value: 43.569828137034264
    - type: euclidean_spearman
      value: 44.2699366594503
    - type: main_score
      value: 44.2699366594503
    task:
      type: STS
  - dataset:
      config: fr
      name: MTEB STS22 (fr)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 70.64344773137496
    - type: cosine_spearman
      value: 77.00398643056744
    - type: manhattan_pearson
      value: 71.64373853764818
    - type: manhattan_spearman
      value: 76.71158725879226
    - type: euclidean_pearson
      value: 71.58320199923101
    - type: euclidean_spearman
      value: 77.00398643056744
    - type: main_score
      value: 77.00398643056744
    task:
      type: STS
  - dataset:
      config: fr-pl
      name: MTEB STS22 (fr-pl)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 54.305559003968206
    - type: cosine_spearman
      value: 50.709255283710995
    - type: manhattan_pearson
      value: 52.33784187543789
    - type: manhattan_spearman
      value: 50.709255283710995
    - type: euclidean_pearson
      value: 53.00660084455784
    - type: euclidean_spearman
      value: 50.709255283710995
    - type: main_score
      value: 50.709255283710995
    task:
      type: STS
  - dataset:
      config: it
      name: MTEB STS22 (it)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 57.4312835830767
    - type: cosine_spearman
      value: 60.39610834515271
    - type: manhattan_pearson
      value: 57.83823485037898
    - type: manhattan_spearman
      value: 60.374938260317535
    - type: euclidean_pearson
      value: 57.81507077373551
    - type: euclidean_spearman
      value: 60.39610834515271
    - type: main_score
      value: 60.39610834515271
    task:
      type: STS
  - dataset:
      config: pl
      name: MTEB STS22 (pl)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 8.000336595206134
    - type: cosine_spearman
      value: 26.768906191975933
    - type: manhattan_pearson
      value: 1.588769366202155
    - type: manhattan_spearman
      value: 26.76300987426348
    - type: euclidean_pearson
      value: 1.4181188576056134
    - type: euclidean_spearman
      value: 26.768906191975933
    - type: main_score
      value: 26.768906191975933
    task:
      type: STS
  - dataset:
      config: pl-en
      name: MTEB STS22 (pl-en)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 35.08730015173829
    - type: cosine_spearman
      value: 32.79791295777814
    - type: manhattan_pearson
      value: 36.273935331272256
    - type: manhattan_spearman
      value: 35.88704294252439
    - type: euclidean_pearson
      value: 34.54132550386404
    - type: euclidean_spearman
      value: 32.79791295777814
    - type: main_score
      value: 32.79791295777814
    task:
      type: STS
  - dataset:
      config: ru
      name: MTEB STS22 (ru)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.03100716792233671
    - type: cosine_spearman
      value: 14.721380413194854
    - type: manhattan_pearson
      value: 5.7576102223040735
    - type: manhattan_spearman
      value: 15.08182690716095
    - type: euclidean_pearson
      value: 4.871526064730011
    - type: euclidean_spearman
      value: 14.721380413194854
    - type: main_score
      value: 14.721380413194854
    task:
      type: STS
  - dataset:
      config: tr
      name: MTEB STS22 (tr)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 20.597902459466386
    - type: cosine_spearman
      value: 33.694510807738595
    - type: manhattan_pearson
      value: 27.530294926210807
    - type: manhattan_spearman
      value: 33.74254435313719
    - type: euclidean_pearson
      value: 26.964862787540962
    - type: euclidean_spearman
      value: 33.694510807738595
    - type: main_score
      value: 33.694510807738595
    task:
      type: STS
  - dataset:
      config: zh
      name: MTEB STS22 (zh)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 23.127885111414432
    - type: cosine_spearman
      value: 44.92964024177277
    - type: manhattan_pearson
      value: 31.77656358573927
    - type: manhattan_spearman
      value: 44.964763982886375
    - type: euclidean_pearson
      value: 31.061639313469925
    - type: euclidean_spearman
      value: 44.92964024177277
    - type: main_score
      value: 44.92964024177277
    task:
      type: STS
  - dataset:
      config: zh-en
      name: MTEB STS22 (zh-en)
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 37.41111741585122
    - type: cosine_spearman
      value: 41.64399741744448
    - type: manhattan_pearson
      value: 35.71015224548175
    - type: manhattan_spearman
      value: 41.460551673456045
    - type: euclidean_pearson
      value: 36.83160927711053
    - type: euclidean_spearman
      value: 41.64399741744448
    - type: main_score
      value: 41.64399741744448
    task:
      type: STS
---


# all-MiniLM-L6-v2
This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

## Usage (Sentence-Transformers)
Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:
```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
```

## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)
```

## Evaluation Results

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/all-MiniLM-L6-v2)

------

## Background

The project aims to train sentence embedding models on very large sentence level datasets using a self-supervised 
contrastive learning objective. We used the pretrained [`nreimers/MiniLM-L6-H384-uncased`](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model and fine-tuned in on a 
1B sentence pairs dataset. We use a contrastive learning objective: given a sentence from the pair, the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in our dataset.

We developed this model during the 
[Community week using JAX/Flax for NLP & CV](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104), 
organized by Hugging Face. We developed this model as part of the project:
[Train the Best Sentence Embedding Model Ever with 1B Training Pairs](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354). We benefited from efficient hardware infrastructure to run the project: 7 TPUs v3-8, as well as intervention from Googles Flax, JAX, and Cloud team member about efficient deep learning frameworks.

## Intended uses

Our model is intended to be used as a sentence and short paragraph encoder. Given an input text, it outputs a vector which captures 
the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

By default, input text longer than 256 word pieces is truncated.


## Training procedure

### Pre-training 

We use the pretrained [`nreimers/MiniLM-L6-H384-uncased`](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model. Please refer to the model card for more detailed information about the pre-training procedure.

### Fine-tuning 

We fine-tune the model using a contrastive objective. Formally, we compute the cosine similarity from each possible sentence pairs from the batch.
We then apply the cross entropy loss by comparing with true pairs.

#### Hyper parameters

We trained our model on a TPU v3-8. We train the model during 100k steps using a batch size of 1024 (128 per TPU core).
We use a learning rate warm up of 500. The sequence length was limited to 128 tokens. We used the AdamW optimizer with
a 2e-5 learning rate. The full training script is accessible in this current repository: `train_script.py`.

#### Training data

We use the concatenation from multiple datasets to fine-tune our model. The total number of sentence pairs is above 1 billion sentences.
We sampled each dataset given a weighted probability which configuration is detailed in the `data_config.json` file.


| Dataset                                                  | Paper                                    | Number of training tuples  |
|--------------------------------------------------------|:----------------------------------------:|:--------------------------:|
| [Reddit comments (2015-2018)](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit) | [paper](https://arxiv.org/abs/1904.06472) | 726,484,430 |
| [S2ORC](https://github.com/allenai/s2orc) Citation pairs (Abstracts) | [paper](https://aclanthology.org/2020.acl-main.447/) | 116,288,806 |
| [WikiAnswers](https://github.com/afader/oqa#wikianswers-corpus) Duplicate question pairs | [paper](https://doi.org/10.1145/2623330.2623677) | 77,427,422 |
| [PAQ](https://github.com/facebookresearch/PAQ) (Question, Answer) pairs | [paper](https://arxiv.org/abs/2102.07033) | 64,371,441 |
| [S2ORC](https://github.com/allenai/s2orc) Citation pairs (Titles) | [paper](https://aclanthology.org/2020.acl-main.447/) | 52,603,982 |
| [S2ORC](https://github.com/allenai/s2orc) (Title, Abstract) | [paper](https://aclanthology.org/2020.acl-main.447/) | 41,769,185 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) (Title, Body) pairs  | - | 25,316,456 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) (Title+Body, Answer) pairs  | - | 21,396,559 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) (Title, Answer) pairs  | - | 21,396,559 |
| [MS MARCO](https://microsoft.github.io/msmarco/) triplets | [paper](https://doi.org/10.1145/3404835.3462804) | 9,144,553 |
| [GOOAQ: Open Question Answering with Diverse Answer Types](https://github.com/allenai/gooaq) | [paper](https://arxiv.org/pdf/2104.08727.pdf) | 3,012,496 |
| [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) (Title, Answer) | [paper](https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) | 1,198,260 |
| [Code Search](https://huggingface.co/datasets/code_search_net) | - | 1,151,414 |
| [COCO](https://cocodataset.org/#home) Image captions | [paper](https://link.springer.com/chapter/10.1007%2F978-3-319-10602-1_48) | 828,395|
| [SPECTER](https://github.com/allenai/specter) citation triplets | [paper](https://doi.org/10.18653/v1/2020.acl-main.207) | 684,100 |
| [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) (Question, Answer) | [paper](https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) | 681,164 |
| [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) (Title, Question) | [paper](https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) | 659,896 |
| [SearchQA](https://huggingface.co/datasets/search_qa) | [paper](https://arxiv.org/abs/1704.05179) | 582,261 |
| [Eli5](https://huggingface.co/datasets/eli5) | [paper](https://doi.org/10.18653/v1/p19-1346) | 325,475 |
| [Flickr 30k](https://shannon.cs.illinois.edu/DenotationGraph/) | [paper](https://transacl.org/ojs/index.php/tacl/article/view/229/33) | 317,695 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) Duplicate questions (titles) | | 304,525 |
| AllNLI ([SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) | [paper SNLI](https://doi.org/10.18653/v1/d15-1075), [paper MultiNLI](https://doi.org/10.18653/v1/n18-1101) | 277,230 | 
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) Duplicate questions (bodies) | | 250,519 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) Duplicate questions (titles+bodies) | | 250,460 |
| [Sentence Compression](https://github.com/google-research-datasets/sentence-compression) | [paper](https://www.aclweb.org/anthology/D13-1155/) | 180,000 |
| [Wikihow](https://github.com/pvl/wikihow_pairs_dataset) | [paper](https://arxiv.org/abs/1810.09305) | 128,542 |
| [Altlex](https://github.com/chridey/altlex/) | [paper](https://aclanthology.org/P16-1135.pdf) | 112,696 |
| [Quora Question Triplets](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | - | 103,663 |
| [Simple Wikipedia](https://cs.pomona.edu/~dkauchak/simplification/) | [paper](https://www.aclweb.org/anthology/P11-2117/) | 102,225 |
| [Natural Questions (NQ)](https://ai.google.com/research/NaturalQuestions) | [paper](https://transacl.org/ojs/index.php/tacl/article/view/1455) | 100,231 |
| [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) | [paper](https://aclanthology.org/P18-2124.pdf) | 87,599 |
| [TriviaQA](https://huggingface.co/datasets/trivia_qa) | - | 73,346 |
| **Total** | | **1,170,060,424** |