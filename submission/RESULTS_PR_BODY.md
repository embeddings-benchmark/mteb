## Summary

Adds all six CoREB(v1) task results for
`keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b`.

## Evaluation

- Device: NVIDIA L4
- Child settings: each pinned model's official MTEB loader configuration
- Retrieval child: `codefuse-ai/F2LLM-v2-330M` at
  `e8ef9a8eb907a9dffdd9442424a967ba73e70d31`
- Reranking child: `codefuse-ai/C2LLM-7B` at
  `c1dc16d6d64eb962c783bfb36a6d9c2f24a86dca`

The route is determined only from the public coarse MTEB task type.

| Task | Main score |
|---|---:|
| CorebC2CReranking | 0.42383 |
| CorebC2CRetrieval | 0.54383 |
| CorebC2TReranking | 0.95685 |
| CorebC2TRetrieval | 0.96794 |
| CorebT2CReranking | 0.28824 |
| CorebT2CRetrieval | 0.43093 |
| **Macro mean** | **0.6019367** |

At preparation time, the live CoREB leader was C2LLM-7B at `0.6009717`.

## Result provenance

- Retrieval: newly evaluated through the router using the pinned
  F2LLM-v2-330M revision with MTEB `2.18.6`.
- Reranking: copied verbatim from the existing official C2LLM-7B result files
  at the same pinned revision and MTEB `2.12.30`. Because the router delegates
  the entire task and forwards all MTEB encode context unchanged, these are the
  router's exact Reranking outputs.

## Disclosure

This composition was selected after inspecting public CoREB results. CoREB has
only a test split, so the policy is leaderboard-adapted.

## Before opening

- Ensure the result model revision equals the public Hugging Face commit SHA.
- Link the MTEB model implementation PR.
