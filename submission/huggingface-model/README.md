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

| MTEB task type | Model | Dimension | Configured max tokens |
|---|---|---:|---:|
| Retrieval | F2LLM-v2-330M | 896 | 8,192 |
| Reranking | C2LLM-7B | 3,584 | 2,048 |

Each individual evaluation task has a fixed output dimension. Current MTEB
`ModelMeta` has no representation for task-dependent dimensions, so the model
implementation leaves `embed_dim` unset.

## CoREB v1 results

| Task | Main score |
|---|---:|
| CorebC2CReranking | 0.42383 |
| CorebC2CRetrieval | 0.54383 |
| CorebC2TReranking | 0.95685 |
| CorebC2TRetrieval | 0.96794 |
| CorebT2CReranking | 0.28824 |
| CorebT2CRetrieval | 0.43093 |
| **Macro mean** | **0.6019367** |

Retrieval was evaluated through the router with MTEB 2.18.6 on an NVIDIA L4.
The reranking scores are the existing official C2LLM-7B results at the exact
pinned revision, copied without modification because the router delegates
those tasks and forwards their encode context unchanged.

## Reproduction

The MTEB model implementation pins both child model revisions and delegates to
each child's official MTEB loader configuration. In particular, the F2LLM
Retrieval route uses its configured bfloat16/FlashAttention 2 implementation,
while the C2LLM Reranking route keeps C2LLM's official adapter defaults.
Evaluate all six CoREB v1 tasks with MTEB 2.18.6 and CUDA. The submission
includes per-task MTEB JSON results, an environment record, and the complete
run log.
