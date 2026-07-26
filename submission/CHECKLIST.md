# Submission checklist

1. [x] Create the public Hugging Face model-card/config repository from
   `submission/huggingface-model`.

   ```bash
   hf auth login
   hf repo create \
     keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b \
     --repo-type model
   hf upload \
     keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b \
     submission/huggingface-model \
     .
   ```

2. [x] Record its immutable commit SHA:
   `a3f180e8d5a1c85aa16f3d37027498d1af780074`.
3. [x] Set the MTEB `ModelMeta` to that immutable revision.
4. Run the two model loading checks from `submission/MTEB_PR_BODY.md`.
5. Re-key the completed MTEB cache and prepare the manual results branch:

   ```bash
   PYTHONPATH=. python submission/finalize_results.py \
     /path/to/mteb-cache a3f180e8d5a1c85aa16f3d37027498d1af780074 \
     --reranking-source \
     /path/to/gce-20260726-102117/artifacts/mteb-cache/results/keonkim__coreb-task-type-router-f2llmv2-330m-c2llm-7b/router-v1 \
     --prepare-submission
   ```

   The script preserves the source cache, validates all six task files, and
   writes `model_meta.json` from the submission implementation. It merges the
   newly evaluated MTEB 2.18.6 Reranking results and run settings into the
   Retrieval cache. The exact official C2LLM-7B result directory remains
   accepted as a compatibility fallback.
6. Check every result JSON contains the expected task, split, score, model name,
   revision, MTEB version, and no error.
7. Open the MTEB implementation PR first.
8. Open the results PR and link the implementation PR.
9. Keep the leaderboard-adaptation and task-dependent-dimension disclosures in
    both PRs.
