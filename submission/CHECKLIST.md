# Submission checklist

1. Create the public Hugging Face model-card/config repository from
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

2. Record its immutable commit SHA.
3. Replace `TODO_AFTER_HF_UPLOAD` in the MTEB `ModelMeta`.
4. Run the two model loading checks from `submission/MTEB_PR_BODY.md`.
5. Re-key the completed MTEB cache and prepare the manual results branch:

   ```bash
   PYTHONPATH=. python submission/finalize_results.py \
     /path/to/mteb-cache HUGGING_FACE_COMMIT_SHA \
     --prepare-submission
   ```

   The script preserves the source cache, validates all six task files, and
   writes `model_meta.json` from the submission implementation.
6. Check every result JSON contains the expected task, split, score, model name,
   revision, MTEB version, and no error.
7. Open the MTEB implementation PR first.
8. Open the results PR and link the implementation PR.
9. Keep the leaderboard-adaptation and task-dependent-dimension disclosures in
    both PRs.
