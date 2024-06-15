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
    task:
      type: Retrieval
  - dataset:
      config: ar
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.05006610360999117
    - type: cosine_spearman
      value: 0.2263866797712348
    - type: manhattan_pearson
      value: 0.13260328120447723
    - type: manhattan_spearman
      value: 0.22340169287120717
    - type: euclidean_pearson
      value: 0.13082283087945362
    - type: euclidean_spearman
      value: 0.2263866797712348
    - type: main_score
      value: 0.2263866797712348
    task:
      type: STS
  - dataset:
      config: de
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.26596033966146115
    - type: cosine_spearman
      value: 0.31044353994772356
    - type: manhattan_pearson
      value: 0.21718468273577896
    - type: manhattan_spearman
      value: 0.311979155955977
    - type: euclidean_pearson
      value: 0.2151728902500591
    - type: euclidean_spearman
      value: 0.31044353994772356
    - type: main_score
      value: 0.31044353994772356
    task:
      type: STS
  - dataset:
      config: de-en
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.47545312366545117
    - type: cosine_spearman
      value: 0.44038685024247604
    - type: manhattan_pearson
      value: 0.4810217367438755
    - type: manhattan_spearman
      value: 0.44442850465339095
    - type: euclidean_pearson
      value: 0.4846975590869453
    - type: euclidean_spearman
      value: 0.44038685024247604
    - type: main_score
      value: 0.44038685024247604
    task:
      type: STS
  - dataset:
      config: de-fr
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.26472844763068937
    - type: cosine_spearman
      value: 0.3006758748207823
    - type: manhattan_pearson
      value: 0.25808959063835424
    - type: manhattan_spearman
      value: 0.27996294873002153
    - type: euclidean_pearson
      value: 0.26872307920750726
    - type: euclidean_spearman
      value: 0.3006758748207823
    - type: main_score
      value: 0.3006758748207823
    task:
      type: STS
  - dataset:
      config: de-pl
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.0702656697163116
    - type: cosine_spearman
      value: 0.04927056559940413
    - type: manhattan_pearson
      value: 0.0901762174854638
    - type: manhattan_spearman
      value: 0.07359790736410993
    - type: euclidean_pearson
      value: 0.06729027056926624
    - type: euclidean_spearman
      value: 0.04927056559940413
    - type: main_score
      value: 0.04927056559940413
    task:
      type: STS
  - dataset:
      config: en
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.6709882753030891
    - type: cosine_spearman
      value: 0.6721465212910986
    - type: manhattan_pearson
      value: 0.6841388868877885
    - type: manhattan_spearman
      value: 0.6783615682571867
    - type: euclidean_pearson
      value: 0.6821374069918402
    - type: euclidean_spearman
      value: 0.6721465212910986
    - type: main_score
      value: 0.6721465212910986
    task:
      type: STS
  - dataset:
      config: es
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.4433815143022264
    - type: cosine_spearman
      value: 0.5477772552456677
    - type: manhattan_pearson
      value: 0.4929424073081744
    - type: manhattan_spearman
      value: 0.5525969655269095
    - type: euclidean_pearson
      value: 0.48483578263920635
    - type: euclidean_spearman
      value: 0.5477772552456677
    - type: main_score
      value: 0.5477772552456677
    task:
      type: STS
  - dataset:
      config: es-en
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.4993601240112664
    - type: cosine_spearman
      value: 0.5341895837272506
    - type: manhattan_pearson
      value: 0.4986265183075983
    - type: manhattan_spearman
      value: 0.5310065931046005
    - type: euclidean_pearson
      value: 0.5016469746986203
    - type: euclidean_spearman
      value: 0.5341895837272506
    - type: main_score
      value: 0.5341895837272506
    task:
      type: STS
  - dataset:
      config: es-it
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.42568537775842247
    - type: cosine_spearman
      value: 0.44269936659450304
    - type: manhattan_pearson
      value: 0.43954212787242286
    - type: manhattan_spearman
      value: 0.4432159550471527
    - type: euclidean_pearson
      value: 0.4356982813703426
    - type: euclidean_spearman
      value: 0.44269936659450304
    - type: main_score
      value: 0.44269936659450304
    task:
      type: STS
  - dataset:
      config: fr
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.7064344773137496
    - type: cosine_spearman
      value: 0.7700398643056744
    - type: manhattan_pearson
      value: 0.7164373853764818
    - type: manhattan_spearman
      value: 0.7671158725879226
    - type: euclidean_pearson
      value: 0.71583201999231
    - type: euclidean_spearman
      value: 0.7700398643056744
    - type: main_score
      value: 0.7700398643056744
    task:
      type: STS
  - dataset:
      config: fr-pl
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.5430555900396821
    - type: cosine_spearman
      value: 0.50709255283711
    - type: manhattan_pearson
      value: 0.5233784187543788
    - type: manhattan_spearman
      value: 0.50709255283711
    - type: euclidean_pearson
      value: 0.5300660084455784
    - type: euclidean_spearman
      value: 0.50709255283711
    - type: main_score
      value: 0.50709255283711
    task:
      type: STS
  - dataset:
      config: it
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.574312835830767
    - type: cosine_spearman
      value: 0.6039610834515271
    - type: manhattan_pearson
      value: 0.5783823485037898
    - type: manhattan_spearman
      value: 0.6037493826031753
    - type: euclidean_pearson
      value: 0.5781507077373551
    - type: euclidean_spearman
      value: 0.6039610834515271
    - type: main_score
      value: 0.6039610834515271
    task:
      type: STS
  - dataset:
      config: pl
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.08000336595206134
    - type: cosine_spearman
      value: 0.26768906191975933
    - type: manhattan_pearson
      value: 0.01588769366202155
    - type: manhattan_spearman
      value: 0.2676300987426348
    - type: euclidean_pearson
      value: 0.014181188576056134
    - type: euclidean_spearman
      value: 0.26768906191975933
    - type: main_score
      value: 0.26768906191975933
    task:
      type: STS
  - dataset:
      config: pl-en
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.3508730015173829
    - type: cosine_spearman
      value: 0.32797912957778136
    - type: manhattan_pearson
      value: 0.36273935331272256
    - type: manhattan_spearman
      value: 0.3588704294252439
    - type: euclidean_pearson
      value: 0.3454132550386404
    - type: euclidean_spearman
      value: 0.32797912957778136
    - type: main_score
      value: 0.32797912957778136
    task:
      type: STS
  - dataset:
      config: ru
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.0003100716792233671
    - type: cosine_spearman
      value: 0.14721380413194854
    - type: manhattan_pearson
      value: 0.057576102223040736
    - type: manhattan_spearman
      value: 0.1508182690716095
    - type: euclidean_pearson
      value: 0.04871526064730011
    - type: euclidean_spearman
      value: 0.14721380413194854
    - type: main_score
      value: 0.14721380413194854
    task:
      type: STS
  - dataset:
      config: tr
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.20597902459466386
    - type: cosine_spearman
      value: 0.3369451080773859
    - type: manhattan_pearson
      value: 0.2753029492621081
    - type: manhattan_spearman
      value: 0.3374254435313719
    - type: euclidean_pearson
      value: 0.2696486278754096
    - type: euclidean_spearman
      value: 0.3369451080773859
    - type: main_score
      value: 0.3369451080773859
    task:
      type: STS
  - dataset:
      config: zh
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.2312788511141443
    - type: cosine_spearman
      value: 0.4492964024177277
    - type: manhattan_pearson
      value: 0.3177656358573927
    - type: manhattan_spearman
      value: 0.44964763982886374
    - type: euclidean_pearson
      value: 0.31061639313469924
    - type: euclidean_spearman
      value: 0.4492964024177277
    - type: main_score
      value: 0.4492964024177277
    task:
      type: STS
  - dataset:
      config: zh-en
      name: MTEB STS22
      revision: 2de6ce8c1921b71a755b262c6b57fef195dd7906
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 0.37411117415851225
    - type: cosine_spearman
      value: 0.41643997417444484
    - type: manhattan_pearson
      value: 0.3571015224548175
    - type: manhattan_spearman
      value: 0.4146055167345605
    - type: euclidean_pearson
      value: 0.36831609277110533
    - type: euclidean_spearman
      value: 0.41643997417444484
    - type: main_score
      value: 0.41643997417444484
    task:
      type: STS
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
    task:
      type: BitextMining
tags:
- mteb
---
