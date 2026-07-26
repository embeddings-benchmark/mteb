## Summary

Adds all six CoREB(v1) task results for
`keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b`.

## Evaluation

- MTEB: `2.18.6`
- Device: NVIDIA L4
- Child settings: each pinned model's official MTEB loader configuration
- Retrieval child: `codefuse-ai/F2LLM-v2-330M` at
  `e8ef9a8eb907a9dffdd9442424a967ba73e70d31`
- Reranking child: `codefuse-ai/C2LLM-7B` at
  `c1dc16d6d64eb962c783bfb36a6d9c2f24a86dca`

The route is determined only from the public coarse MTEB task type.

## Disclosure

This composition was selected after inspecting public CoREB results. CoREB has
only a test split, so the policy is leaderboard-adapted.

## Before opening

- Insert the final per-task table and macro mean.
- Ensure the result model revision equals the public Hugging Face commit SHA.
- Link the MTEB model implementation PR.
