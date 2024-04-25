# Welcome to MMTEB! 👋

The Massive Multilingual Text Embedding Benchmark (MMTEB) is a community-led extension of [MTEB](https://arxiv.org/abs/2210.07316) to cover embedding tasks for a massive number of languages.

## Background 

The Massive Text Embedding Benchmark (MTEB) is intended to evaluate the quality of document embeddings. When it was initially introduced, the benchmark consisted of 8 embedding tasks and 58 different datasets. Since then, MTEB has been subject to multiple community contributions as well as benchmark extensions over specific languages such as [SEB](https://openreview.net/pdf/f5f1953a9c798ec61bb050e62bc7a94037fd4fab.pdf), [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) and [MTEB-French](https://github.com/Lyon-NLP/mteb-french). However, we want even wider coverage and thus announce the community-led extension of MTEB, where we seek to expand coverage of MTEB to as many languages as possible.

## Contributing to MMTEB

Everyone can join and contribute to this initiative from:
- 10th of April 2024 to 15th of May 2024 for adding new datasets
- 15th of May to 30th of May for running models

Win some SWAG, and become a co-author of our upcoming paper. We aim to publish the results of our findings at a top conference such as EMNLP, NeurIPS, etc. We have identified four ways to contribute:

### 🗃️ 1: Contribute a new dataset

For this segment, you open a PR in the MTEB repository where you create an implementation (subclass) of a task using a new language dataset uploaded to huggingface. Read more about how to add a dataset [here](../adding_a_dataset.md) and check out [one of the previous additions](https://github.com/embeddings-benchmark/mteb/pull/247) for an example.

### 🖥️ 2: Contribute a new task

MTEB currently consists of 8 embedding tasks including tasks such as STS, retrieval, reranking, and more. If you feel like there is a category of tasks that is not yet covered, we would welcome contributions of these as well.

### 🔍 3: Contribute new scores

Once we have the datasets, we want to evaluate models on them. We welcome evaluation scores for models, which will be added to the leaderboard.

### 🔓 4: Review PRs

We welcome reviews of PRs adding new datasets. If you wish to review PRs of a specific language feel free to contact members of the MTEB team.

## Authorship

We follow a similar approach as in the [SeaCrowd Project](https://github.com/SEACrowd#contributing-to-seacrowd) and use a point-based system to determine co-authorships. 

To be considered a co-author, at least 10 contribution points are required. The position of contributors in the author list is determined by the score they acquire, higher scores will appear first.

To monitor how many points you have obtained, the contribution point tracking is now live at [this sheet](points.md) and we recommend updating the score along with your PR. Past contributions also count. 

Everyone with sufficient points will also be added to the MTEB GitHub and Huggingface repository as a contributor.

# Contribution point guideline
The contribution points are computed using the following table:

> **Note**: The purpose of the point system is not to barrier collaboration, but to reward contributions. We might adjust the point requirement lower to accommodate more co-authorship if needed.


| Contribution type   | Demand              | Points  | Description                                                                                                       |
| ------------------- | ------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| New dataset         | As many as possible | 2+bonus | The first dataset for a language x task gains 4 bonus points. |
| New task            | If relevant         | 10      | Task 2.                                                                                                           |
| Dataset annotations | On demand           | 1       | Adding missing dataset annotations to existing datasets.                                                          |
| Bug fixes            | On demand           | 1-10    | Points depends the effect of code changes. If you want to find issues related to the MMTEB you can find them [here](https://github.com/embeddings-benchmark/mteb/milestone/1), issues marked with "help-wanted" or "good-first-issue" are great places to start. |
| Running Models      | On demand           | 1       | Task 3.                                   |
| Review PR           | On demand           | 2       | Task 4.                                   |

For the purpose of counting points, a language is defined by its [ISO 639-3](https://en.wikipedia.org/wiki/ISO_639-3) code, however, we encourage dialects or written language variants. All programming languages are considered one language.

Team submissions are free to distribute points among the members as they like.

## Communication Channels

We will communicate via this GitHub repository. Please feel free to open issues or discussions and `Watch` the repository to be notified of any changes.

# Acknowledgments

We thank [Contextual AI](https://contextual.ai/) for sponsoring the compute for this project.
