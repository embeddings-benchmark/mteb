# New Dataset PR Guidelines

Ensure the PR description includes the evidence and context reviewers need to evaluate correctness and quality.

## Gap in MTEB

- The PR description must explain why this dataset fills an existing gap in MTEB (e.g. new language, domain, task type, or modality not yet covered).
- A generic statement that the dataset is "useful" is not sufficient — the gap must be stated concretely. A tagged issue is acceptable.

## Descriptive Statistics

- The PR description must include descriptive statistics for the dataset directly (e.g. number of samples per split, average text length, label distribution, language breakdown).
- A reference to the dataset card is not sufficient — the stats must appear in the PR itself.

## Dataset Size

- Flag the PR if the dataset appears oversized for its task type without justification (e.g. more than ~2048 examples for binary classification).
- The author should note whether the size was deliberately reduced.

## Results Table Format

- If reproduction results from a paper are included, they must be presented as a table, not as prose.
- Columns: score source (e.g. "PR" and "Paper").
- Rows: one row per model.
- Flag the PR if results are described only in paragraph form, or if the table is transposed (models as columns).
- If the dataset has no associated paper, or no equivalent model exists in MTEB to compare against, the PR must include a note explaining this.

Example of correct format:

| Model | PR | Paper |
|---|---|---|
| model-a | 42.1 | 42.3 |
| model-b | 38.7 | 39.0 |

## Model Results

- The PR must report scores for at least two models:
  - `mteb/baseline-random-encoder` (random baseline)
  - A small real model, e.g. `intfloat/multilingual-e5-small`
  - These can be run with: `mteb run -m {model_name} -t {task_name}`
- The random baseline confirms performance is neither trivially high (suggesting the task is too easy or data is leaked) nor at chance level (suggesting the task or metric is misconfigured).
- Flag the PR if either model result is missing.

## AI-Generated Boilerplate

- Flag PR descriptions that contain noise that appears AI-generated rather than human-written.
- Common patterns to flag:
  - Bullet lists of CI checks that passed (e.g. "Ruff, typos, mypy, media decoding, ID/qrel integrity checks pass")
  - Generic validation summaries (e.g. "6,724 practical tests pass")
  - Filler sections such as "Validation", "Testing Done", or "Checklist" filled with auto-generated text that adds no information for reviewers
  - Verbose restatements of what the code does without any insight
- The PR description should explain the dataset, its origin, and why it belongs in MTEB.

## Data Upload Script

- If the Hugging Face dataset is hosted under a username that matches the PR author's GitHub username, check whether a corresponding upload script exists under `scripts/data/`.
- If no such script is present, ask the author to add one showing how the data was created and uploaded to Hugging Face.
