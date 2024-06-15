---
model-index:
- name: all-MiniLM-L6-v2
  results:
  - dataset:
      config: default
      name: MTEB BSARDRetrieval
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
      value: 6.0e-05
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
      value: 6.0e-05
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
      value: 0.0021
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
      value: 2.0e-05
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
      value: 0.01802
    - type: recall_at_3
      value: 0.0
    - type: recall_at_5
      value: 0.0
    - type: main_score
      value: 0.0
    - type: recall_at_100
      value: 0.0
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB BornholmBitextMining
      revision: 3bc5cfb4ec514264fe2db5615fac9016f7251552
      split: test
      type: strombergnlp/bornholmsk_parallel
    metrics:
    - type: accuracy
      value: 0.36
    - type: f1
      value: 0.2968132161955691
    - type: main_score
      value: 0.2968132161955691
    - type: precision
      value: 0.27690919913419915
    - type: recall
      value: 0.36
    - type: f1
      value: 0.2968132161955691
    task:
      type: BitextMining
tags:
- mteb
---
