# New Model PR Guidelines

Ensure the PR description includes the evidence and context reviewers need to evaluate correctness and quality.

## ModelMeta Completeness

- The PR must include a `ModelMeta` object with all fields filled out to the extent possible.
- Required fields: `name`, `loader`, `languages`, `open_weights`, `revision`, `release_date`, `n_parameters`, `memory_usage_mb`, `embed_dim`, `license`, `max_tokens`, `reference`, `similarity_fn_name`, `framework`.
- Flag the PR if any required field is missing or set to a placeholder value without explanation.
- The model must be public: either available via API or with publicly downloadable weights.

## Model Loading

- The PR must confirm the model loads correctly via:
  - `mteb.get_model(model_name, revision)`
  - `mteb.get_model_meta(model_name, revision)`
- Flag the PR if there is no indication that loading was verified.

## Mock Run Results

- The PR must include the output of `mteb mock-run -m your_model_name` (or the equivalent Python call).
- The resulting `mteb_mock_run_results.md` file should be committed with the PR.
- Flag the PR if mock run results are missing or if `all_passed` is not confirmed.

## Reproduction Results

- If the model has an associated paper or published benchmark results, the PR must reproduce at least one benchmark score and compare it against the paper.
- Results must be presented as a table, not as prose.
- Columns: score source (e.g. "PR" and "Paper").
- Rows: one row per benchmark or task.
- Flag the PR if results are described only in paragraph form, or if the table is transposed (benchmarks as columns).
- If the model has no associated paper or published results, the PR must include a note explaining this.

Example of correct format:

| Benchmark | PR | Paper |
|---|---|---|
| MTEB English | 64.2 | 64.5 |
| BEIR | 51.3 | 51.0 |

## AI-Generated Boilerplate

- Flag PR descriptions that contain noise that appears AI-generated rather than human-written.
- Common patterns to flag:
  - Bullet lists of CI checks that passed (e.g. "Ruff, typos, mypy checks pass")
  - Generic validation summaries (e.g. "all tests pass")
  - Filler sections such as "Validation", "Testing Done", or "Checklist" filled with auto-generated text that adds no information for reviewers
  - Verbose restatements of what the code does without any insight
- The PR description should explain the model, its origin, and why it belongs in MTEB.
