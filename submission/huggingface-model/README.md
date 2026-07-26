---
license: apache-2.0
library_name: mteb
pipeline_tag: feature-extraction
tags:
  - sentence-transformers
  - text-embeddings-inference
  - mteb
  - router
  - code
base_model:
  - codefuse-ai/F2LLM-v2-330M
  - codefuse-ai/C2LLM-7B
---

# CoREB task-type router: F2LLM-v2-330M + C2LLM-7B

This is a deterministic, training-free MTEB router for code retrieval:

- MTEB `Retrieval` tasks route to
  [`codefuse-ai/F2LLM-v2-330M`](https://huggingface.co/codefuse-ai/F2LLM-v2-330M)
  at revision `e8ef9a8eb907a9dffdd9442424a967ba73e70d31`.
- MTEB `Reranking` tasks route to
  [`codefuse-ai/C2LLM-7B`](https://huggingface.co/codefuse-ai/C2LLM-7B)
  at revision `c1dc16d6d64eb962c783bfb36a6d9c2f24a86dca`.

The router reads only the coarse public `TaskMetadata.type` field. It does not
inspect the benchmark name, examples, labels, or scores. Evaluation keeps only
one child model resident at a time.

## Important disclosure

The composition was selected after comparing public CoREB results and is
therefore leaderboard-adapted. CoREB exposes only a test split. The submitted
result should be interpreted as the performance of this transparent routing
policy, not as evidence of zero-shot model selection.

## Embedding dimensions

The output dimension depends on the route:

| MTEB task type | Model | Dimension |
|---|---|---:|
| Retrieval | F2LLM-v2-330M | 896 |
| Reranking | C2LLM-7B | 3,584 |

Each individual evaluation task has a fixed output dimension. Current MTEB
`ModelMeta` has no representation for task-dependent dimensions, so the model
implementation leaves `embed_dim` unset.

## Reproduction

The MTEB model implementation pins both child model revisions. Evaluate the six
CoREB v1 tasks with MTEB 2.18.6, CUDA, bfloat16, and FlashAttention 2. The
submission includes per-task MTEB JSON results, an environment record, and the
complete run log.
