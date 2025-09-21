---
model-index:
- name: sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (ar)
      type: mteb/sts22-crosslingual-sts
      config: ar
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.2263866797712348
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (de)
      type: mteb/sts22-crosslingual-sts
      config: de
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.31044353994772356
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (de-en)
      type: mteb/sts22-crosslingual-sts
      config: de-en
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.44038685024247604
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (de-fr)
      type: mteb/sts22-crosslingual-sts
      config: de-fr
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.3006758748207823
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (de-pl)
      type: mteb/sts22-crosslingual-sts
      config: de-pl
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.04927056559940413
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (en)
      type: mteb/sts22-crosslingual-sts
      config: en
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.6721465212910986
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (es)
      type: mteb/sts22-crosslingual-sts
      config: es
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.5477772552456677
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (es-en)
      type: mteb/sts22-crosslingual-sts
      config: es-en
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.5341895837272506
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (es-it)
      type: mteb/sts22-crosslingual-sts
      config: es-it
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.44269936659450304
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (fr)
      type: mteb/sts22-crosslingual-sts
      config: fr
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.7700398643056744
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (fr-pl)
      type: mteb/sts22-crosslingual-sts
      config: fr-pl
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.50709255283711
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (it)
      type: mteb/sts22-crosslingual-sts
      config: it
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.6039610834515271
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (pl)
      type: mteb/sts22-crosslingual-sts
      config: pl
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.26768906191975933
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (pl-en)
      type: mteb/sts22-crosslingual-sts
      config: pl-en
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.32797912957778136
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (ru)
      type: mteb/sts22-crosslingual-sts
      config: ru
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.14721380413194854
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (tr)
      type: mteb/sts22-crosslingual-sts
      config: tr
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.3369451080773859
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (zh)
      type: mteb/sts22-crosslingual-sts
      config: zh
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.4492964024177277
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: sentence-similarity
      name: STS
    dataset:
      name: STS22 (zh-en)
      type: mteb/sts22-crosslingual-sts
      config: zh-en
      split: test
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
    metrics:
    - type: cosine_spearman
      value: 0.41643997417444484
      name: cosine_spearman
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: text-retrieval
      name: Retrieval
    dataset:
      name: BSARDRetrieval (default)
      type: mteb/BSARDRetrieval
      config: default
      split: test
      revision: 8c492add6a14ac188f2debdaf6cbdfb406fd6be3
    metrics:
    - type: recall_at_100
      value: 0.0
      name: recall_at_100
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
  - task:
      type: translation
      name: BitextMining
    dataset:
      name: BornholmBitextMining (default)
      type: mteb/BornholmBitextMining
      config: default
      split: test
      revision: 5b02048bd75e79275aa91a1fce6cdfd3f4a391cb
    metrics:
    - type: f1
      value: 0.2968132161955691
      name: f1
    source:
      url: https://github.com/embeddings-benchmark/mteb/
      name: MTEB
tags:
- mteb
---
