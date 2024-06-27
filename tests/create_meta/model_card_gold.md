---
model-index:
- name: all-MiniLM-L6-v2
  results:
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
tags:
- mteb
---
