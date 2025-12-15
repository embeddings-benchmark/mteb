# CHANGELOG

<!-- version list -->

## Unreleased


## v2.3.11 (2025-12-12)

### Bug Fixes

- Make `PIL` optional ([#3713](https://github.com/embeddings-benchmark/mteb/pull/3713),
  [`80fef47`](https://github.com/embeddings-benchmark/mteb/commit/80fef471f0bdb883a7578ca95406689a7ff6f03c))


## v2.3.10 (2025-12-10)

### Bug Fixes

- Add optional "per language table"
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Check emptiness before further analysis
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- External links in hf space ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Set column width for task and language tables
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

### Features

- Add per-language table creation ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Enhance language view condition to include 'all' option in per-language table
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Enhance per-language table functionality
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Enhance per-language table functionality with support flag and styling improvements
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Refacto language filtering support
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Simplify per-language table styling by rounding values for wide tables
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Update per-language table to support 'all' option and improve styling for wide tables
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

### Refactoring

- Simplify per-language table creation by removing unnecessary comments and print statements
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))

- Update button display options in per-language table styling
  ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617),
  [`46612af`](https://github.com/embeddings-benchmark/mteb/commit/46612af421721a3c21029fe6793a207077d28477))


## v2.3.9 (2025-12-08)

### Bug Fixes

- Add baseline encoders ([#3701](https://github.com/embeddings-benchmark/mteb/pull/3701),
  [`fd395fd`](https://github.com/embeddings-benchmark/mteb/commit/fd395fd80e3192b6c3ef4771c6c9b05ae477b3a4))


## v2.3.8 (2025-12-08)

### Bug Fixes

- Add hamming score to multilabel classification
  ([#3700](https://github.com/embeddings-benchmark/mteb/pull/3700),
  [`bc2e24e`](https://github.com/embeddings-benchmark/mteb/commit/bc2e24e09376cd5a21954da3681ac49a5919ff06))

### Testing

- Add mixed performance case for hamming_score
  ([#3700](https://github.com/embeddings-benchmark/mteb/pull/3700),
  [`bc2e24e`](https://github.com/embeddings-benchmark/mteb/commit/bc2e24e09376cd5a21954da3681ac49a5919ff06))


## v2.3.7 (2025-12-08)

### Bug Fixes

- Don't pass embed dim to `openai/text-embedding-ada-002`
  ([#3689](https://github.com/embeddings-benchmark/mteb/pull/3689),
  [`25af7b9`](https://github.com/embeddings-benchmark/mteb/commit/25af7b9221af9ba319019cb785c1bb60dd55832a))


## v2.3.6 (2025-12-08)

### Bug Fixes

- Correct public_training_code syntax and initialize training_datasets as an empty set
  ([`a39c011`](https://github.com/embeddings-benchmark/mteb/commit/a39c0119f99c7bd6f6de62180f253b7d5fdf3223))

- Update import path for ModelMeta in kowshik24_models.py
  ([`a39c011`](https://github.com/embeddings-benchmark/mteb/commit/a39c0119f99c7bd6f6de62180f253b7d5fdf3223))


## v2.3.5 (2025-12-07)

### Bug Fixes

- Change git clone results depth to 1
  ([#3682](https://github.com/embeddings-benchmark/mteb/pull/3682),
  [`874ddf2`](https://github.com/embeddings-benchmark/mteb/commit/874ddf2efd23b90016e9ef2e8f1756194998d6a1))

- Resolve search wrapper warning ([#3681](https://github.com/embeddings-benchmark/mteb/pull/3681),
  [`46fb44c`](https://github.com/embeddings-benchmark/mteb/commit/46fb44cab716aed701f6df60c6051ae49cecca35))


## v2.3.4 (2025-12-05)

### Bug Fixes

- `get_model` now correctly assumed `SentenceTransformer`
  ([#3673](https://github.com/embeddings-benchmark/mteb/pull/3673),
  [`21cf638`](https://github.com/embeddings-benchmark/mteb/commit/21cf6382dee64661bb2aee0b3c85fa9cf7591821))

- `get_model` now correctly assumed `SentenceTransformer` if
  ([#3673](https://github.com/embeddings-benchmark/mteb/pull/3673),
  [`21cf638`](https://github.com/embeddings-benchmark/mteb/commit/21cf6382dee64661bb2aee0b3c85fa9cf7591821))


## v2.3.3 (2025-12-05)

### Bug Fixes

- `get_model` now correctly assumed `SentenceTransformer` if
  ([`968e498`](https://github.com/embeddings-benchmark/mteb/commit/968e498b0d7ea4f96659757255d1eea7e477055c))

- Cohere import error ([#3665](https://github.com/embeddings-benchmark/mteb/pull/3665),
  [`745ad84`](https://github.com/embeddings-benchmark/mteb/commit/745ad84755827737d0a804a3498d04ad1a95d431))

- GoogleTextEmbeddingModel were given multiple `model_name`
  ([#3653](https://github.com/embeddings-benchmark/mteb/pull/3653),
  [`94979ca`](https://github.com/embeddings-benchmark/mteb/commit/94979ca42284547c26426069279cceb191ec766a))

- Linq-Embed-Mistral loader does not take `instruction_template` as a kwargs
  ([#3653](https://github.com/embeddings-benchmark/mteb/pull/3653),
  [`94979ca`](https://github.com/embeddings-benchmark/mteb/commit/94979ca42284547c26426069279cceb191ec766a))

- Reduce the number of decimals for the number of parameters
  ([#3668](https://github.com/embeddings-benchmark/mteb/pull/3668),
  [`397e0dd`](https://github.com/embeddings-benchmark/mteb/commit/397e0dd9b96dd382a8322c91c07baad65a3d0617))

- Remove "Unknown" for int on leaderboard causing them to be unsortable
  ([#3653](https://github.com/embeddings-benchmark/mteb/pull/3653),
  [`94979ca`](https://github.com/embeddings-benchmark/mteb/commit/94979ca42284547c26426069279cceb191ec766a))

- Remove docker system prune command
  ([#3671](https://github.com/embeddings-benchmark/mteb/pull/3671),
  [`6d92cbc`](https://github.com/embeddings-benchmark/mteb/commit/6d92cbced52bbacffb707642b16895825d805266))

### Continuous Integration

- Remove unnecessary items on disk ([#3664](https://github.com/embeddings-benchmark/mteb/pull/3664),
  [`2c51e3a`](https://github.com/embeddings-benchmark/mteb/commit/2c51e3a6122878547e63b5df781c2936f57fc3ea))

- Update broken links in pull request template
  ([#3656](https://github.com/embeddings-benchmark/mteb/pull/3656),
  [`095803c`](https://github.com/embeddings-benchmark/mteb/commit/095803c5c4951d36959d0411ad548a84c2a367b0))

### Documentation

- Add direct image links to the readme
  ([#3667](https://github.com/embeddings-benchmark/mteb/pull/3667),
  [`8e2929d`](https://github.com/embeddings-benchmark/mteb/commit/8e2929d4194d7dd7d496618bea22da54f50099eb))

- Minor grammar fixes ([#3657](https://github.com/embeddings-benchmark/mteb/pull/3657),
  [`663bb87`](https://github.com/embeddings-benchmark/mteb/commit/663bb87080215672ca88572cf962fdd959b69dfa))


## v2.3.2 (2025-12-03)

### Bug Fixes

- Bump gradio to v6 ([#3629](https://github.com/embeddings-benchmark/mteb/pull/3629),
  [`a882295`](https://github.com/embeddings-benchmark/mteb/commit/a8822953bd6ccc95e27395e873cb8a63b35e04e5))

- Fix display for task information and improve UI for benchmark filtering
  ([#3629](https://github.com/embeddings-benchmark/mteb/pull/3629),
  [`a882295`](https://github.com/embeddings-benchmark/mteb/commit/a8822953bd6ccc95e27395e873cb8a63b35e04e5))


## v2.3.1 (2025-12-03)

### Bug Fixes

- `colpali_training_set` & updated `JinaVDR` and `ViDoRe` tasks annotation
  ([#3636](https://github.com/embeddings-benchmark/mteb/pull/3636),
  [`1ce74c2`](https://github.com/embeddings-benchmark/mteb/commit/1ce74c22b89d1b48941258611862fb98d289c096))

- Add flag to run public only tasks
  ([#3563](https://github.com/embeddings-benchmark/mteb/pull/3563),
  [`f905b68`](https://github.com/embeddings-benchmark/mteb/commit/f905b68a79765ddb3ba2ce09895171a684ce7584))

- Remove unused import
  ([`1ecd892`](https://github.com/embeddings-benchmark/mteb/commit/1ecd8921cb22035ab1e578ad8fa9f626837ea8b0))

- Remove useless check
  ([`1ecd892`](https://github.com/embeddings-benchmark/mteb/commit/1ecd8921cb22035ab1e578ad8fa9f626837ea8b0))

- Ruff format
  ([`1ecd892`](https://github.com/embeddings-benchmark/mteb/commit/1ecd8921cb22035ab1e578ad8fa9f626837ea8b0))

- Use only one class which inherits from CrossEncoderWrapper
  ([`1ecd892`](https://github.com/embeddings-benchmark/mteb/commit/1ecd8921cb22035ab1e578ad8fa9f626837ea8b0))

- **colqwen**: Require transformers>=4.57 and refresh metadata, set revision
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

### Chores

- **colqwen**: Remove unused methods and fix lint errors
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

- **colqwen**: Set release date for tomoro colqwen3 8b
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

- **colqwen**: Update encoding progress message
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

- **colqwen**: Update model revisions for colqwen models
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

### Documentation

- **colqwen**: Update train data annotation
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

### Features

- **colqwen3**: Add fused image-text encoding path
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

- **colqwen3**: Add wrapper and model metadata
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

- **colqwen3**: Update ColQwen3Wrapper to use bfloat16 and enhance similarity scoring
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

### Refactoring

- **colqwen**: Reorder wrappers and metadata definitions for clarity
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))

- **colqwen**: Unify encode method with get_fused_embeddings
  ([#3627](https://github.com/embeddings-benchmark/mteb/pull/3627),
  [`71ac96c`](https://github.com/embeddings-benchmark/mteb/commit/71ac96ccef0ebdee06e71e621e55eb0845370bdc))


## v2.3.0 (2025-11-28)

### Bug Fixes

- Updated metadata on model memory ([#3624](https://github.com/embeddings-benchmark/mteb/pull/3624),
  [`73168c6`](https://github.com/embeddings-benchmark/mteb/commit/73168c621ee859dbe23c40e12ddd6928d2f340f9))

### Continuous Integration

- Add HF_TOKEN to dataset loading and merge CI
  ([#3622](https://github.com/embeddings-benchmark/mteb/pull/3622),
  [`4ffef40`](https://github.com/embeddings-benchmark/mteb/commit/4ffef408b0990c9cf4ee7ed7ba74175e21426b08))

- Update action versions ([#3623](https://github.com/embeddings-benchmark/mteb/pull/3623),
  [`bcf4e82`](https://github.com/embeddings-benchmark/mteb/commit/bcf4e82fed2eca1972d96850717600350b25fc04))

### Documentation

- Update "speeding up"-section to include bumping version
  ([#3634](https://github.com/embeddings-benchmark/mteb/pull/3634),
  [`392186f`](https://github.com/embeddings-benchmark/mteb/commit/392186f574af609702897860a6fb6c0159dba7db))

### Features

- Add search encoder backend ([#3492](https://github.com/embeddings-benchmark/mteb/pull/3492),
  [`4ed7ef4`](https://github.com/embeddings-benchmark/mteb/commit/4ed7ef49ab0c36c79dcf8c658a47b0dac4595839))


## v2.2.2 (2025-11-25)

### Bug Fixes

- Vidore loading ([#3618](https://github.com/embeddings-benchmark/mteb/pull/3618),
  [`ca8e7c4`](https://github.com/embeddings-benchmark/mteb/commit/ca8e7c47110d5466604fb99e068ad7dc31288b49))


## v2.2.1 (2025-11-25)

### Bug Fixes

- Avoiding stating warning if what is logged is not a warning
  ([#3619](https://github.com/embeddings-benchmark/mteb/pull/3619),
  [`b0d6c7b`](https://github.com/embeddings-benchmark/mteb/commit/b0d6c7b5e840beb8bd0b3923cc0e56cf87b26248))


## v2.2.0 (2025-11-25)

### Features

- Make STS and PairClassification asymmetric
  ([#3568](https://github.com/embeddings-benchmark/mteb/pull/3568),
  [`5010468`](https://github.com/embeddings-benchmark/mteb/commit/5010468f3c72ccd45b02d959eb9941ff9aa543c5))


## v2.1.19 (2025-11-25)

### Bug Fixes

- Cache language filtering ([#3612](https://github.com/embeddings-benchmark/mteb/pull/3612),
  [`f75bfc4`](https://github.com/embeddings-benchmark/mteb/commit/f75bfc46dab6445acfd405a6b95837a2d9e63b45))


## v2.1.18 (2025-11-24)

### Bug Fixes

- Correcting the cohere lstrip bug in `cohere`
  ([#3610](https://github.com/embeddings-benchmark/mteb/pull/3610),
  [`398b31b`](https://github.com/embeddings-benchmark/mteb/commit/398b31b4c09fe2f328588a08f760af559be12812))


## v2.1.17 (2025-11-20)

### Bug Fixes

- Improve messages for running missing splits
  ([#3596](https://github.com/embeddings-benchmark/mteb/pull/3596),
  [`5d7b78b`](https://github.com/embeddings-benchmark/mteb/commit/5d7b78bd84437f14f01149d86e362b1016d4e88f))


## v2.1.16 (2025-11-20)

### Bug Fixes

- Bump gradio version to fix links on leaderboard
  ([#3591](https://github.com/embeddings-benchmark/mteb/pull/3591),
  [`09021df`](https://github.com/embeddings-benchmark/mteb/commit/09021df642d3f526801dda26b95cff1741f304ef))

- Fix adapted from points to the models itself
  ([#3566](https://github.com/embeddings-benchmark/mteb/pull/3566),
  [`4636b24`](https://github.com/embeddings-benchmark/mteb/commit/4636b24c5450b0179589094429658e9d9dc386ac))

- Issues on cache hits ([#3558](https://github.com/embeddings-benchmark/mteb/pull/3558),
  [`9d7d4df`](https://github.com/embeddings-benchmark/mteb/commit/9d7d4dfbf128e754d12efc5cc71639766696e49c))

- Overwrite / ignore existing results if not mergeable
  ([#3558](https://github.com/embeddings-benchmark/mteb/pull/3558),
  [`9d7d4df`](https://github.com/embeddings-benchmark/mteb/commit/9d7d4dfbf128e754d12efc5cc71639766696e49c))

- Typo for attn_implementation kwargs in jasper models
  ([#3592](https://github.com/embeddings-benchmark/mteb/pull/3592),
  [`0d33bd3`](https://github.com/embeddings-benchmark/mteb/commit/0d33bd38b06cc4d689426cd2cbd72280941dbb9a))


## v2.1.15 (2025-11-19)

### Bug Fixes

- Add adapted_from and convert public_training_code/data to URLs
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Add required metadata fields (memory_usage_mb, open_weights, public_training_code,
  public_training_data)
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Correct release_date to November (2025-11-16)
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Move model to model_implementations and clean up per @Samoed feedback
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Remove old files and revert __init__.py per @Samoed feedback
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Remove old model file from wrong location
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Remove stray root file
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Restore file with proper newlines (encoding fix)
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Restore proper line breaks in model file
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Update release_date to 2025-11-15 (actual training date)
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Update revision to commit hash and add training datasets
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Update training_datasets format per @Samoed feedback
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Use sentence_transformers_loader per MTEB guide
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

- Utilize `max_seq_length` ([#3588](https://github.com/embeddings-benchmark/mteb/pull/3588),
  [`f7b481e`](https://github.com/embeddings-benchmark/mteb/commit/f7b481e4e3c51cbb445c171ff4718b0a409afe1c))

- **lint**: Correct module docstring in model file
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))

### Features

- Add training dataset info to ModelMeta
  ([`723fd98`](https://github.com/embeddings-benchmark/mteb/commit/723fd9808be2dce238f1ddef2af7b5d28adbcd3a))


## v2.1.14 (2025-11-16)

### Bug Fixes

- Benchmark references links ([#3560](https://github.com/embeddings-benchmark/mteb/pull/3560),
  [`07f1e6e`](https://github.com/embeddings-benchmark/mteb/commit/07f1e6e67c52668558f29a3e3bb7e9583e8e280d))

- External links in hf space ([#3560](https://github.com/embeddings-benchmark/mteb/pull/3560),
  [`07f1e6e`](https://github.com/embeddings-benchmark/mteb/commit/07f1e6e67c52668558f29a3e3bb7e9583e8e280d))


## v2.1.13 (2025-11-15)

### Bug Fixes

- Set default input_type for VoyageMultiModalModelWrapper
  ([#3567](https://github.com/embeddings-benchmark/mteb/pull/3567),
  [`711e7cb`](https://github.com/embeddings-benchmark/mteb/commit/711e7cbfdd5625ddb70f5bc8a773b7fa8c740ef6))


## v2.1.12 (2025-11-15)

### Bug Fixes

- Fix adapted from points to the models itself
  ([#3565](https://github.com/embeddings-benchmark/mteb/pull/3565),
  [`08b8ec7`](https://github.com/embeddings-benchmark/mteb/commit/08b8ec7426a0dc442b923b626c65d2f941e82dbb))


## v2.1.11 (2025-11-14)

### Bug Fixes

- Add jasper token compression model
  ([#3557](https://github.com/embeddings-benchmark/mteb/pull/3557),
  [`64459d1`](https://github.com/embeddings-benchmark/mteb/commit/64459d127fd259d2a88254946204e8586ae809ab))

- MTEB-NL switches to v2 datasets with new prompts
  ([#3555](https://github.com/embeddings-benchmark/mteb/pull/3555),
  [`26e36cd`](https://github.com/embeddings-benchmark/mteb/commit/26e36cda9d4c7c44d069d7c89a9e27b6fa62ec6e))


## v2.1.10 (2025-11-13)

### Bug Fixes

- Resolve hash randomization in retrieval task ID generation
  ([#3553](https://github.com/embeddings-benchmark/mteb/pull/3553),
  [`0c4f099`](https://github.com/embeddings-benchmark/mteb/commit/0c4f099b2252e4cfa394d841a6c5207e20dfbd2e))


## v2.1.9 (2025-11-13)

### Bug Fixes

- Add VisualDocumentRetrieval to previous benchmark names
  ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))

- Added leaderboard Vidore V3 ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))

- Remove JinaVisualDocumentBenchmark
  ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))

- Update JinaVisualDocumentBenchmark summary table creation method
  ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))

- Update VISUAL_DOCUMENT_RETRIEVAL to use VidoreBenchmark
  ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))

### Features

- Update summary table for ViDoRe V3 to reflect Document Understanding tasks
  ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))

### Refactoring

- Update leaderboard references ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542),
  [`ab390ce`](https://github.com/embeddings-benchmark/mteb/commit/ab390cec5f05a5baf720b25b25b562057e6eccf8))


## v2.1.8 (2025-11-13)

### Bug Fixes

- Pass encode kwargs in all dataloaders
  ([#3548](https://github.com/embeddings-benchmark/mteb/pull/3548),
  [`eaec6cb`](https://github.com/embeddings-benchmark/mteb/commit/eaec6cb0571783416c76a3475f6e21bc02869ea5))

- Update dataset revisions for Vidore3 retrieval classes + remove custom load_data methods
  ([`1a02edb`](https://github.com/embeddings-benchmark/mteb/commit/1a02edbb7d50ec9f02c258267de299a731ea10ed))

### Continuous Integration

- Fix false positive check in typos
  ([#3540](https://github.com/embeddings-benchmark/mteb/pull/3540),
  [`b2b9599`](https://github.com/embeddings-benchmark/mteb/commit/b2b959936afa2a658fcf4797f17715535a1d28ac))

### Documentation

- Convert all descriptions to singe line
  ([#3544](https://github.com/embeddings-benchmark/mteb/pull/3544),
  [`fe83e27`](https://github.com/embeddings-benchmark/mteb/commit/fe83e27968adff648de9520e187164ea5f906935))

### Features

- Added descriptive statistics for public ViDoRe V3 datasets
  ([`1a02edb`](https://github.com/embeddings-benchmark/mteb/commit/1a02edbb7d50ec9f02c258267de299a731ea10ed))

- Better leaderboard
  ([`1a02edb`](https://github.com/embeddings-benchmark/mteb/commit/1a02edbb7d50ec9f02c258267de299a731ea10ed))

- Enhance ViDoRe V3 benchmark with detailed description and update task domains
  ([`1a02edb`](https://github.com/embeddings-benchmark/mteb/commit/1a02edbb7d50ec9f02c258267de299a731ea10ed))

### Refactoring

- Update Vidore3 retrieval classes and paths for improved organization
  ([`1a02edb`](https://github.com/embeddings-benchmark/mteb/commit/1a02edbb7d50ec9f02c258267de299a731ea10ed))


## v2.1.7 (2025-11-07)

### Bug Fixes

- MTEB-NL prompts ([#3516](https://github.com/embeddings-benchmark/mteb/pull/3516),
  [`8f3f806`](https://github.com/embeddings-benchmark/mteb/commit/8f3f8067734257ff082fb5ba4596dfc3fbc8fe6f))


## v2.1.6 (2025-11-06)

### Bug Fixes

- Add support for python 3.14 ([#3450](https://github.com/embeddings-benchmark/mteb/pull/3450),
  [`632a83a`](https://github.com/embeddings-benchmark/mteb/commit/632a83a50dd6bb98b1e73035565230031d716937))

- Model revision
  ([`28f9c54`](https://github.com/embeddings-benchmark/mteb/commit/28f9c54fdf046e80c17158698ee0fae6d4c885de))

- Recover ESCIReranking for train data
  ([`28f9c54`](https://github.com/embeddings-benchmark/mteb/commit/28f9c54fdf046e80c17158698ee0fae6d4c885de))

- Recover the original auto-generated kalm_training_data
  ([`28f9c54`](https://github.com/embeddings-benchmark/mteb/commit/28f9c54fdf046e80c17158698ee0fae6d4c885de))

- Restore docs logo files
  ([`28f9c54`](https://github.com/embeddings-benchmark/mteb/commit/28f9c54fdf046e80c17158698ee0fae6d4c885de))


## v2.1.5 (2025-11-04)

### Bug Fixes

- Materialize corpus id to speed up evaluation
  ([#3518](https://github.com/embeddings-benchmark/mteb/pull/3518),
  [`08f76a6`](https://github.com/embeddings-benchmark/mteb/commit/08f76a6c183d231ed5a9ce903ae8a894a0373479))


## v2.1.4 (2025-10-30)

### Bug Fixes

- Reupload winogrande ([#3513](https://github.com/embeddings-benchmark/mteb/pull/3513),
  [`fe43f73`](https://github.com/embeddings-benchmark/mteb/commit/fe43f735f2ac51532a3e9a63a0ec132159171ca8))


## v2.1.3 (2025-10-29)

### Bug Fixes

- Aggregated task evaluation ([#3510](https://github.com/embeddings-benchmark/mteb/pull/3510),
  [`5eae04c`](https://github.com/embeddings-benchmark/mteb/commit/5eae04ce0746ea87b77f96ffd9cadcffceca1150))


## v2.1.2 (2025-10-29)

### Bug Fixes

- Remove `set_float32_matmul_precision`
  ([#3509](https://github.com/embeddings-benchmark/mteb/pull/3509),
  [`ce07dfd`](https://github.com/embeddings-benchmark/mteb/commit/ce07dfdb6c7c7425f12c1a074d672049c9a4da64))


## v2.1.1 (2025-10-27)

### Bug Fixes

- `top_k` document selection in two stage reranking
  ([#3486](https://github.com/embeddings-benchmark/mteb/pull/3486),
  [`16ae6ff`](https://github.com/embeddings-benchmark/mteb/commit/16ae6ff9cdc44cb1e2ce9dfa73155ec50cf77dd3))

- Add prompts to hardnegative tasks
  ([#3469](https://github.com/embeddings-benchmark/mteb/pull/3469),
  [`7b7bdd0`](https://github.com/embeddings-benchmark/mteb/commit/7b7bdd021dac853623dadc50136512ef85a16db8))

- Qrels selection ([#3479](https://github.com/embeddings-benchmark/mteb/pull/3479),
  [`31c8329`](https://github.com/embeddings-benchmark/mteb/commit/31c8329817e2c1f6e52a90166738d53c683d7ca5))

- Qrels selection negative scores ([#3479](https://github.com/embeddings-benchmark/mteb/pull/3479),
  [`31c8329`](https://github.com/embeddings-benchmark/mteb/commit/31c8329817e2c1f6e52a90166738d53c683d7ca5))

- Release CI ([#3493](https://github.com/embeddings-benchmark/mteb/pull/3493),
  [`21223ed`](https://github.com/embeddings-benchmark/mteb/commit/21223edd870601105567a64803e9728af080ba03))

- Rollback to semantic release ([#3502](https://github.com/embeddings-benchmark/mteb/pull/3502),
  [`1325328`](https://github.com/embeddings-benchmark/mteb/commit/13253287ec7b2ea472afcb665bcabe0a10205f2a))

- Simplify release ([#3494](https://github.com/embeddings-benchmark/mteb/pull/3494),
  [`4484112`](https://github.com/embeddings-benchmark/mteb/commit/4484112ceaade5326e4844926f8b5fa6e8cdd197))

- Task metadata was not passed in Jina implementation
  ([#3485](https://github.com/embeddings-benchmark/mteb/pull/3485),
  [`ea1bac1`](https://github.com/embeddings-benchmark/mteb/commit/ea1bac1a650e4f254537b87c2193217b93dd53eb))

- Update action ([#3448](https://github.com/embeddings-benchmark/mteb/pull/3448),
  [`b649e6f`](https://github.com/embeddings-benchmark/mteb/commit/b649e6f7207fcd8955b255af93e5e22ce20c65a3))

- Verify languages during filtering
  ([#3472](https://github.com/embeddings-benchmark/mteb/pull/3472),
  [`799b869`](https://github.com/embeddings-benchmark/mteb/commit/799b869747c9dd45c37e6f9c0836d366336e3e84))

### Continuous Integration

- New release workflow ([#3448](https://github.com/embeddings-benchmark/mteb/pull/3448),
  [`b649e6f`](https://github.com/embeddings-benchmark/mteb/commit/b649e6f7207fcd8955b255af93e5e22ce20c65a3))

### Documentation

- Fix broken links ([#3483](https://github.com/embeddings-benchmark/mteb/pull/3483),
  [`dfd516a`](https://github.com/embeddings-benchmark/mteb/commit/dfd516aa33abfaca37b72b2c55eb29fbdc550242))

- Update links in readme ([#3484](https://github.com/embeddings-benchmark/mteb/pull/3484),
  [`b0b0e7d`](https://github.com/embeddings-benchmark/mteb/commit/b0b0e7d2744886e8ded4b3ca58e183c78995e133))


## v2.1.0 (2025-10-21)

### Bug Fixes

- Add more impots to root `__init__`
  ([#3458](https://github.com/embeddings-benchmark/mteb/pull/3458),
  [`f8c07ff`](https://github.com/embeddings-benchmark/mteb/commit/f8c07ff43e42f79aa9a148b2921e43f0b2396e17))

- License pyproject ([#3457](https://github.com/embeddings-benchmark/mteb/pull/3457),
  [`6fbe482`](https://github.com/embeddings-benchmark/mteb/commit/6fbe482fc1217f6083f09c0ed3d069b7ea18414e))

### Documentation

- Ignore overview docs ([#3456](https://github.com/embeddings-benchmark/mteb/pull/3456),
  [`5e903b1`](https://github.com/embeddings-benchmark/mteb/commit/5e903b145316041056daad4598a0d94d573e8670))

### Features

- Add mteb nl ([#3464](https://github.com/embeddings-benchmark/mteb/pull/3464),
  [`4f9f157`](https://github.com/embeddings-benchmark/mteb/commit/4f9f157df75d98dc4da95ac6615769564e6940f0))


## v2.0.5 (2025-10-21)

### Bug Fixes

- Vdr category ([#3465](https://github.com/embeddings-benchmark/mteb/pull/3465),
  [`4b384bb`](https://github.com/embeddings-benchmark/mteb/commit/4b384bb4d27b6adf29f3c1dc4d1ca8e914190270))


## v2.0.4 (2025-10-20)

### Bug Fixes

- Roll back setting OMP_NUM_THREADS for clustering
  ([#3444](https://github.com/embeddings-benchmark/mteb/pull/3444),
  [`38e7bc7`](https://github.com/embeddings-benchmark/mteb/commit/38e7bc776d2372b9de33128905d1f8aa737f709b))

### Documentation

- Don't shorten embedding size ([#3455](https://github.com/embeddings-benchmark/mteb/pull/3455),
  [`be20185`](https://github.com/embeddings-benchmark/mteb/commit/be20185035c0f9fb77b07041265eb50e010306c5))

- Update readme header ([#3443](https://github.com/embeddings-benchmark/mteb/pull/3443),
  [`0a6fe95`](https://github.com/embeddings-benchmark/mteb/commit/0a6fe957f4c9b8806fb6ddb2d48930ecc3df19ef))

### Testing

- Disable flaky test ([#3444](https://github.com/embeddings-benchmark/mteb/pull/3444),
  [`38e7bc7`](https://github.com/embeddings-benchmark/mteb/commit/38e7bc776d2372b9de33128905d1f8aa737f709b))


## v2.0.3 (2025-10-20)

### Bug Fixes

- Speedup retrieval computation ([#3454](https://github.com/embeddings-benchmark/mteb/pull/3454),
  [`01f3a19`](https://github.com/embeddings-benchmark/mteb/commit/01f3a19064aa800db3515d9959feed5478cdb91d))


## v2.0.2 (2025-10-20)

### Bug Fixes

- Add citations to models ([#3439](https://github.com/embeddings-benchmark/mteb/pull/3439),
  [`a04d78b`](https://github.com/embeddings-benchmark/mteb/commit/a04d78bf0e7cc859079d18376745dc1e5f02d935))

- Add citations to models (#3435) ([#3439](https://github.com/embeddings-benchmark/mteb/pull/3439),
  [`a04d78b`](https://github.com/embeddings-benchmark/mteb/commit/a04d78bf0e7cc859079d18376745dc1e5f02d935))

### Continuous Integration

- Updating docs ci ([#3445](https://github.com/embeddings-benchmark/mteb/pull/3445),
  [`3af1aa0`](https://github.com/embeddings-benchmark/mteb/commit/3af1aa0af9f77e72b29cfe228ca2f8fe6157898d))

### Features

- Updating to v2 ([#3445](https://github.com/embeddings-benchmark/mteb/pull/3445),
  [`3af1aa0`](https://github.com/embeddings-benchmark/mteb/commit/3af1aa0af9f77e72b29cfe228ca2f8fe6157898d))


## v2.0.1 (2025-10-20)

### Bug Fixes

- : rename TaskMetadata.py to resolve class/module ambiguity
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- [v2] Add mock dialog retrieal task
  ([#2824](https://github.com/embeddings-benchmark/mteb/pull/2824),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add @classmethod for @field_validators in TaskMetadata
  ([#3100](https://github.com/embeddings-benchmark/mteb/pull/3100),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add `ModelMeta` license & custom validations
  ([#2293](https://github.com/embeddings-benchmark/mteb/pull/2293),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add `trust_remote_code` to MIRACLRetrieval
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add `trust_remote_code` to MIRACLRetrieval
  ([#2344](https://github.com/embeddings-benchmark/mteb/pull/2344),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add adapted_from to Cmedqaretrieval
  ([#2806](https://github.com/embeddings-benchmark/mteb/pull/2806),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add adapted_from to Cmedqaretrieval
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add annotation models for stella zh
  ([#2277](https://github.com/embeddings-benchmark/mteb/pull/2277),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add annotation models for stella zh
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add annotations for Voyage exp ([#2144](https://github.com/embeddings-benchmark/mteb/pull/2144),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add beta version of RTEB related benchmarks
  ([#3048](https://github.com/embeddings-benchmark/mteb/pull/3048),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add check for code lora
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add citation to models ([#3403](https://github.com/embeddings-benchmark/mteb/pull/3403),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add citations to models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add conflicting dependencies to toml
  ([#3191](https://github.com/embeddings-benchmark/mteb/pull/3191),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add datasets in CodeRAG-Bench ([#1595](https://github.com/embeddings-benchmark/mteb/pull/1595),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add dedicated display for RTEB benchmark results
  ([#3089](https://github.com/embeddings-benchmark/mteb/pull/3089),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add docs and .load_results to ResultsCache
  ([#2833](https://github.com/embeddings-benchmark/mteb/pull/2833),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add Encodechka benchmark ([#2561](https://github.com/embeddings-benchmark/mteb/pull/2561),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add missing training sets for qzhou
  ([#3023](https://github.com/embeddings-benchmark/mteb/pull/3023),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add mixedbread
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add model specific dependencies in pyproject.toml
  ([#2424](https://github.com/embeddings-benchmark/mteb/pull/2424),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add ModelMeta rubert-mini-frida, BERTA
  ([#2330](https://github.com/embeddings-benchmark/mteb/pull/2330),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add more training data annotations
  ([#2178](https://github.com/embeddings-benchmark/mteb/pull/2178),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add nb_sbert model ([#2339](https://github.com/embeddings-benchmark/mteb/pull/2339),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add new benchmark beRuSciBench along with AbsTaskTextRegression
  ([#2716](https://github.com/embeddings-benchmark/mteb/pull/2716),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add option to remove benchmark from leaderboard
  ([#2417](https://github.com/embeddings-benchmark/mteb/pull/2417),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add option to remove leaderboard from leaderboard
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add prompt for MIRACLRetrievalHardNegatives
  ([#3266](https://github.com/embeddings-benchmark/mteb/pull/3266),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add retry and token counting in Cohere models
  ([#3253](https://github.com/embeddings-benchmark/mteb/pull/3253),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add rteb submission references and improve descriptions.
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add submission references for RTEB
  ([#3233](https://github.com/embeddings-benchmark/mteb/pull/3233),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add train and test split for both datasets
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add Training data annotations ([#2173](https://github.com/embeddings-benchmark/mteb/pull/2173),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add training data annotations to uderver-bloom models
  ([#2210](https://github.com/embeddings-benchmark/mteb/pull/2210),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add training data annotations to uderver-bloom models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add training data for Bilingual Embeddings
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add VDR Multilingual Dataset ([#2408](https://github.com/embeddings-benchmark/mteb/pull/2408),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add version check for `embeddinggemma-300m`
  ([#3189](https://github.com/embeddings-benchmark/mteb/pull/3189),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add VN-MTEB benchmark and Leaderboard
  ([#2995](https://github.com/embeddings-benchmark/mteb/pull/2995),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add voyage quantization models ([#3092](https://github.com/embeddings-benchmark/mteb/pull/3092),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add WebFAQ bitext mining tasks ([#2326](https://github.com/embeddings-benchmark/mteb/pull/2326),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add WebSSL models ([#2604](https://github.com/embeddings-benchmark/mteb/pull/2604),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added b1ade_models.py ([#2340](https://github.com/embeddings-benchmark/mteb/pull/2340),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added dataframe utilities to BenchmarkResults
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added docs for `mteb.evaluate` ([#2831](https://github.com/embeddings-benchmark/mteb/pull/2831),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added docs for `mteb.evaluate`
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added Filter Modality ([#2262](https://github.com/embeddings-benchmark/mteb/pull/2262),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added training data annotation for MMLW models
  ([#2188](https://github.com/embeddings-benchmark/mteb/pull/2188),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added training data annotations to MXBAI
  ([#2185](https://github.com/embeddings-benchmark/mteb/pull/2185),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added training data for sentence-croissant
  ([#2189](https://github.com/embeddings-benchmark/mteb/pull/2189),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Adding client arg to init method of OpenAI models wrapper
  ([#2803](https://github.com/embeddings-benchmark/mteb/pull/2803),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Adds family of NeuML/pubmedbert-base-embedding models
  ([#2443](https://github.com/embeddings-benchmark/mteb/pull/2443),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Allow closed datasets ([#3059](https://github.com/embeddings-benchmark/mteb/pull/3059),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Allow model to output torch.tensor
  ([#2234](https://github.com/embeddings-benchmark/mteb/pull/2234),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Allow model to output torch.tensor
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Alphabetical ordering of tasks in dropdowns
  ([#2191](https://github.com/embeddings-benchmark/mteb/pull/2191),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Apply review comments
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Apply review suggestions
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Attribute model_name
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- AttributeError in ColPaliEngineWrapper similarity method
  ([#3177](https://github.com/embeddings-benchmark/mteb/pull/3177),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- B1ade ([#2386](https://github.com/embeddings-benchmark/mteb/pull/2386),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Bug in voyage implementation ([#2304](https://github.com/embeddings-benchmark/mteb/pull/2304),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- CachedEmbeddingWrapper issues in both documentation and code
  ([#2779](https://github.com/embeddings-benchmark/mteb/pull/2779),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- CacheWrapper per task ([#2467](https://github.com/embeddings-benchmark/mteb/pull/2467),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Change `passage` prompt to `document`
  ([#2912](https://github.com/embeddings-benchmark/mteb/pull/2912),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Change language for task SlovakMovieReviewSentimentClassification
  ([#3296](https://github.com/embeddings-benchmark/mteb/pull/3296),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Changes requested in PR 2443
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Compute missing data and create issue where not possible
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Convert brightretrieval to two tasks
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Correct logic for filtering public tasks in ModelResult class
  ([#3230](https://github.com/embeddings-benchmark/mteb/pull/3230),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Correct metadata for ArguAna dataset
  ([#3202](https://github.com/embeddings-benchmark/mteb/pull/3202),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Correct task category
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Correctly pass trust remote code to Miracl
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Datasets loading ([#3421](https://github.com/embeddings-benchmark/mteb/pull/3421),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Delete kwargs for similarity score in ColPaliEngineWrapper for method behavior
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Duplicate mteb multilingual variables
  ([#3080](https://github.com/embeddings-benchmark/mteb/pull/3080),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- E5 instruct now listed as sbert compatible
  ([#2475](https://github.com/embeddings-benchmark/mteb/pull/2475),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Edit ack & sponsors ([#3187](https://github.com/embeddings-benchmark/mteb/pull/3187),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Enable NPY ruleset for v2
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Ensure bright uses the correct revision
  ([#2812](https://github.com/embeddings-benchmark/mteb/pull/2812),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Ensure BrightRetrieval is valid to run
  ([#2334](https://github.com/embeddings-benchmark/mteb/pull/2334),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Ensure BrightRetrieval is valid to run
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Ensure MIRACL pass trust_remote_code
  ([#2346](https://github.com/embeddings-benchmark/mteb/pull/2346),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Ensure that there are always relevant docs attached to query
  ([#3058](https://github.com/embeddings-benchmark/mteb/pull/3058),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Ensure that there are always relevant docs attached to query
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Error in computing missing splits when checking if cache is hit
  ([#3130](https://github.com/embeddings-benchmark/mteb/pull/3130),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Error in computing missing splits when checking if cache is hit
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fix bug in voyage implementation
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fixed commit hash for pubmed_bert model2vec models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fixed leaderboard crash ([#2221](https://github.com/embeddings-benchmark/mteb/pull/2221),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Format seed_1_6_embedding_models.py with Ruff
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Formatting and init imports
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Further specified macro-language code for Norwegian
  ([#3228](https://github.com/embeddings-benchmark/mteb/pull/3228),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Further specified macro-language code for Norwegian
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Improving validate_task_to_prompt_name logs and error messages
  ([#3079](https://github.com/embeddings-benchmark/mteb/pull/3079),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Incorrect annotations for Mistral-based embedding models
  ([#2157](https://github.com/embeddings-benchmark/mteb/pull/2157),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Incorrect revision for SNLRetrieval
  ([#3033](https://github.com/embeddings-benchmark/mteb/pull/3033),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Jasper models embeddings having nan values
  ([#2481](https://github.com/embeddings-benchmark/mteb/pull/2481),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Jinav4 revision
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Major updates to docs + make mieb dep optional
  ([#2397](https://github.com/embeddings-benchmark/mteb/pull/2397),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Major updates to documentation
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Make lint
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Make sentence_transformers an optional import
  ([#3167](https://github.com/embeddings-benchmark/mteb/pull/3167),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Make sentence_transformers an optional import
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Make torchvision optional ([#2399](https://github.com/embeddings-benchmark/mteb/pull/2399),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Make torchvision optional
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Me5 trainind data config to include xquad dataset
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- ME5_TRAINING_DATA format
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Minor fixes to docstring and code
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Minor style changes
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Missing import and formatting
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- More training data annotations ([#2220](https://github.com/embeddings-benchmark/mteb/pull/2220),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Move zero-shot percentage calculation to the end of summary
  ([#3231](https://github.com/embeddings-benchmark/mteb/pull/3231),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Only import SparseEncoder once sentence-transformer version have been checked
  ([#2940](https://github.com/embeddings-benchmark/mteb/pull/2940),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Only import SparseEncoder once sentence-transformer version have been checked
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Only pin model name and rank ([#3263](https://github.com/embeddings-benchmark/mteb/pull/3263),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Only pin model name and rank
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Open_clip package validation ([#3073](https://github.com/embeddings-benchmark/mteb/pull/3073),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Perform citation and code formatting
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Pin datasets version ([#2892](https://github.com/embeddings-benchmark/mteb/pull/2892),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Pin gradio dependency to ensure leaderboards works
  ([#2387](https://github.com/embeddings-benchmark/mteb/pull/2387),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Prevent EOS token truncation ([#3218](https://github.com/embeddings-benchmark/mteb/pull/3218),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Prevent incorrectly passing "selector_state" to `get_benchmark`
  ([#2939](https://github.com/embeddings-benchmark/mteb/pull/2939),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Prompt validation for tasks with `-`
  ([#2846](https://github.com/embeddings-benchmark/mteb/pull/2846),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Recompute descriptive stats to match v2 format
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Reduce logging and Warnings ([#2349](https://github.com/embeddings-benchmark/mteb/pull/2349),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Refactor `languages.py` [v2] ([#2813](https://github.com/embeddings-benchmark/mteb/pull/2813),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Refactor models modules
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Refactor split create_tables into static Benchmark methods
  ([#3126](https://github.com/embeddings-benchmark/mteb/pull/3126),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove encode_corpus/query
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove encode_corpus/query from `gme_v_models`
  ([#2388](https://github.com/embeddings-benchmark/mteb/pull/2388),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove links from models with a reference
  ([#3426](https://github.com/embeddings-benchmark/mteb/pull/3426),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove links from models with a reference
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove SummaryRetrieval as a type
  ([#1915](https://github.com/embeddings-benchmark/mteb/pull/1915),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove syntax warnings occuring in python 3.12
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove SyntaxWarnings in py312 ([#2325](https://github.com/embeddings-benchmark/mteb/pull/2325),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove trust_remote_code option
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove untested task selection ([#2821](https://github.com/embeddings-benchmark/mteb/pull/2821),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Remove unused 'is_public' attribute from TaskResult
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Removed missing dataset for MTEB(Multilingual) and bumped version
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Rename evaluators rename to snakecase
  ([#2979](https://github.com/embeddings-benchmark/mteb/pull/2979),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Rename evaluators rename to snakecase
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Rename TaskMetadata.py to resolve class/module ambiguity
  ([#2829](https://github.com/embeddings-benchmark/mteb/pull/2829),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Renaming Zeroshot -> ZeroShot ([#2395](https://github.com/embeddings-benchmark/mteb/pull/2395),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Renaming Zeroshot -> ZeroShot
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Replace relative imports from parent modules with absolute imports
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Replace with passage ([#2934](https://github.com/embeddings-benchmark/mteb/pull/2934),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Repllama models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Resolve conflicting dependencies ([#2323](https://github.com/embeddings-benchmark/mteb/pull/2323),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Resolve conflicting dependencies
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Resolve flash-attention dependency issue
  ([#3265](https://github.com/embeddings-benchmark/mteb/pull/3265),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Resolve flash-attention dependency issue
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Restructure evaluators into folders to match supported modalities
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Reuploaded previously unavailable SNL datasets
  ([#2819](https://github.com/embeddings-benchmark/mteb/pull/2819),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Reuploaded previously unavailable SNL datasets
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Revert rename and add to description
  ([#1918](https://github.com/embeddings-benchmark/mteb/pull/1918),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Run `ruff check` on all files during ci
  ([#3086](https://github.com/embeddings-benchmark/mteb/pull/3086),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Run `ruff check` on all files during ci
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Save results also when raise_error=False
  ([#3138](https://github.com/embeddings-benchmark/mteb/pull/3138),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Seperate model implementations and encoder specifications
  ([#2879](https://github.com/embeddings-benchmark/mteb/pull/2879),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Specify dependencies for nomic models and ensure that they can load
  ([#2748](https://github.com/embeddings-benchmark/mteb/pull/2748),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Specify dependencies for nomic models and ensure that they can load
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Specify revision for opensearch ([#2919](https://github.com/embeddings-benchmark/mteb/pull/2919),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Uniform batch size
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update AmazonPolarityClassification license
  ([#2402](https://github.com/embeddings-benchmark/mteb/pull/2402),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update Caltech101 datasets to latest revision [v1]
  ([#2778](https://github.com/embeddings-benchmark/mteb/pull/2778),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update Caltech101 datasets to latest revision [v2]
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update code task
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update colpali engine models ([#2905](https://github.com/embeddings-benchmark/mteb/pull/2905),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update error message
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update max length
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update model selection for the leaderboard
  ([#2855](https://github.com/embeddings-benchmark/mteb/pull/2855),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update model selection for the leaderboard
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update mteb task according to feedback
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update MTEB(Scandinavian) to use new DanFEVER
  ([#2180](https://github.com/embeddings-benchmark/mteb/pull/2180),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update mteb.get_tasks with an exclude_aggregate parameter to exclude aggregate tasks
  ([#2536](https://github.com/embeddings-benchmark/mteb/pull/2536),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update NVIDIA-Embed training data
  ([#2143](https://github.com/embeddings-benchmark/mteb/pull/2143),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update NVIDIA-Embed training data
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update package requirements in JinaWrapper for einops and flash_attn
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update requirements in JinaWrapper
  ([#2548](https://github.com/embeddings-benchmark/mteb/pull/2548),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update ResultsCache
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update revision for qzhou models ([#3069](https://github.com/embeddings-benchmark/mteb/pull/3069),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update RTEB summary columns ([#3226](https://github.com/embeddings-benchmark/mteb/pull/3226),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update ru models annotation ([#2181](https://github.com/embeddings-benchmark/mteb/pull/2181),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update training dataset info of Seed-1.6-embedding model
  ([#2857](https://github.com/embeddings-benchmark/mteb/pull/2857),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update training datasets and revision for jina models
  ([#2179](https://github.com/embeddings-benchmark/mteb/pull/2179),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update voyage name to include Org.
  ([#2322](https://github.com/embeddings-benchmark/mteb/pull/2322),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Updated citation for mteb(scandinavian)
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Updated revision for jina-embeddings-v4
  ([#3046](https://github.com/embeddings-benchmark/mteb/pull/3046),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Updating the default batch size calculation in the voyage models
  ([#3091](https://github.com/embeddings-benchmark/mteb/pull/3091),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Use official judged queries for TREC-DL 2019/2020 datasets
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Validate lang code in ModelMeta ([#2499](https://github.com/embeddings-benchmark/mteb/pull/2499),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- **models**: Correct eos token handling in `BMRetrieverWrapper`
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- **models**: Ensure prompt_type is passed to format_instruction
  ([#3216](https://github.com/embeddings-benchmark/mteb/pull/3216),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- **models**: Prevent EOS token truncation for BMRetriever
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

### Chores

- Add 'Patent retrieval' subtype to TaskMetadata
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fix colpali_models similarity handle device
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

### Continuous Integration

- Add pre-commit hook ([#2194](https://github.com/embeddings-benchmark/mteb/pull/2194),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add stale workflow ([#3066](https://github.com/embeddings-benchmark/mteb/pull/3066),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Bump semantic release
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Cache `~/.cache/huggingface` ([#2464](https://github.com/embeddings-benchmark/mteb/pull/2464),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Cache ~/.cache/huggingface
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Dataset check on new PR ([#3103](https://github.com/embeddings-benchmark/mteb/pull/3103),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fix config error for semantic release
  ([#2800](https://github.com/embeddings-benchmark/mteb/pull/2800),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Reduce parallel runs for when checking if a dataset exists
  ([#3035](https://github.com/embeddings-benchmark/mteb/pull/3035),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Refactor TaskMetadata eval langs test
  ([#2501](https://github.com/embeddings-benchmark/mteb/pull/2501),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Run dataset loading only when pushing to main
  ([#2480](https://github.com/embeddings-benchmark/mteb/pull/2480),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Temporarily limit pytrec version to "pytrec-eval-terrier>=0.5.6, <0.5.8" to prevent errors
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Updating rerun delays to prevent false positives errors
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

### Documentation

- Add contribution section to the docs
  ([#3396](https://github.com/embeddings-benchmark/mteb/pull/3396),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add MIEB citation in benchmarks ([#2544](https://github.com/embeddings-benchmark/mteb/pull/2544),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add sort to domains for task metadata
  ([#1922](https://github.com/embeddings-benchmark/mteb/pull/1922),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added documentation for push_to_hub
  ([#2747](https://github.com/embeddings-benchmark/mteb/pull/2747),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added documentation for push_to_hub
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fix some typos in `docs/usage/usage.md`
  ([#2835](https://github.com/embeddings-benchmark/mteb/pull/2835),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fix typos
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fix typos in `docs/adding_a_benchmark.md`
  ([#3344](https://github.com/embeddings-benchmark/mteb/pull/3344),
  [`721e8a3`](https://github.com/embeddings-benchmark/mteb/commit/721e8a34743c050179ee5e9093ed24ee7ab37fba))

- Typos; Standardize spacing; Chronological order
  ([#2436](https://github.com/embeddings-benchmark/mteb/pull/2436),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update AbsTaskPairClassification to correct path
  ([#3437](https://github.com/embeddings-benchmark/mteb/pull/3437),
  [`a329381`](https://github.com/embeddings-benchmark/mteb/commit/a3293811f40629a20a71d2c6474099c344749629))

- Update adding benchmark documentation
  ([#3229](https://github.com/embeddings-benchmark/mteb/pull/3229),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update adding_a_dataset.md ([#2947](https://github.com/embeddings-benchmark/mteb/pull/2947),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update adding_a_dataset.md
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update description of EURLex ([#2231](https://github.com/embeddings-benchmark/mteb/pull/2231),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update docs to use mteb.evaluate for two stage retrieval
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Updated citation for mteb(scandinavian)
  ([#1914](https://github.com/embeddings-benchmark/mteb/pull/1914),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

### Features

- Add beir ([#1933](https://github.com/embeddings-benchmark/mteb/pull/1933),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add faiss cache backend ([#3402](https://github.com/embeddings-benchmark/mteb/pull/3402),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add KaLM_Embedding_X_0605 in kalm_models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add MIEB and MIEB-lite as benchmarks
  ([#2035](https://github.com/embeddings-benchmark/mteb/pull/2035),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add Qodo-Embed-1-7B model metadata and rename existing model
  ([#2146](https://github.com/embeddings-benchmark/mteb/pull/2146),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add Qodo-Embed-1-7B model metadata and rename existing model
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add swedish cpc patent classifications to mteb
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add TREC Deep Learning 2019 and 2020 retrieval tasks
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add TRECDL retrieval tasks to English registry
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Add TRECDL2019 and TRECDL2020 descriptive stats
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added dataframe utilities to BenchmarkResults
  ([#2542](https://github.com/embeddings-benchmark/mteb/pull/2542),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added date for tasks
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added description for jinavdr
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added descriptions per dataset
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added jinavdr benchmark
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added max pixel argument for jina models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Added pubmedbert model2vec models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Adjust jina model for new mteb code
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Better reference and fixed comments
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- CacheWrapper per task
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Code cleanup
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Corrected bibtex
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Corrected query numbers
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Create TRECDLRetrieval
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fixed comments
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fixed licenses and added bibtex
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Fixed missing metadata and bibtex
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Implement BMREtrieverWrapper based on InstructSentenceTransformerWrapper
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Made jinav4 compatible with vidore benchmark
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Make colpali run with jinavdr
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Merge MIEB into main 🎉 ([#1944](https://github.com/embeddings-benchmark/mteb/pull/1944),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Officially include RTEB in the leaderboard
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Removed print
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Rename TRECDLRetrieval.py to trecdl_retrieval.py
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Score calculation on cpu
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- UI Overhaul ([#2549](https://github.com/embeddings-benchmark/mteb/pull/2549),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Unify text and image embeddings for all tasks
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Update training datasets and revision for jina models
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- Updating to v2
  ([`c20226f`](https://github.com/embeddings-benchmark/mteb/commit/c20226f945749947913058c749956e266ff96f7d))

- **retrieval**: Add DAPFAM patent retrieval tasks (+18 variants)
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

### Refactoring

- Update training datasets for bmretriever
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

- **models**: Refactor tokenizer setup in `InstructSentenceTransformerWrapper`
  ([`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))

### Testing

- Disable flaky test ([#3442](https://github.com/embeddings-benchmark/mteb/pull/3442),
  [`29e605f`](https://github.com/embeddings-benchmark/mteb/commit/29e605f7f022c47c96567cc9dfcf653d7d270088))

- Fix dataset availability test ([#2141](https://github.com/embeddings-benchmark/mteb/pull/2141),
  [`c07c289`](https://github.com/embeddings-benchmark/mteb/commit/c07c289b0bf1975e4a128ba10414d9a815bebd70))


## v1.39.7 (2025-10-08)

### Bug Fixes

- Change language for task SlovakMovieReviewSentimentClassification
  ([#3296](https://github.com/embeddings-benchmark/mteb/pull/3296),
  [`0a902a3`](https://github.com/embeddings-benchmark/mteb/commit/0a902a3c6a68cd0dc7c116239c2eb8edb081c320))


## v1.39.6 (2025-10-07)

### Bug Fixes

- Add prompt for MIRACLRetrievalHardNegatives
  ([#3266](https://github.com/embeddings-benchmark/mteb/pull/3266),
  [`9b6f320`](https://github.com/embeddings-benchmark/mteb/commit/9b6f320828fa9419e991a3da819c92ead96da109))


## v1.39.5 (2025-10-07)

### Bug Fixes

- Add retry and token counting in Cohere models
  ([#3253](https://github.com/embeddings-benchmark/mteb/pull/3253),
  [`e81c94f`](https://github.com/embeddings-benchmark/mteb/commit/e81c94f6a04bdcfc9905ab09d4328903005ad847))


## v1.39.4 (2025-10-06)

### Bug Fixes

- Only pin model name and rank ([#3265](https://github.com/embeddings-benchmark/mteb/pull/3265),
  [`1e29385`](https://github.com/embeddings-benchmark/mteb/commit/1e29385ec891a9ef58220e0232db6559aa1be278))

- Resolve flash-attention dependency issue
  ([#3265](https://github.com/embeddings-benchmark/mteb/pull/3265),
  [`1e29385`](https://github.com/embeddings-benchmark/mteb/commit/1e29385ec891a9ef58220e0232db6559aa1be278))


## v1.39.3 (2025-10-06)

### Bug Fixes

- Move zero-shot percentage calculation to the end of summary
  ([#3231](https://github.com/embeddings-benchmark/mteb/pull/3231),
  [`65829bd`](https://github.com/embeddings-benchmark/mteb/commit/65829bdc4c09f6e8260bc711aa661338cc3aa063))

- Only pin model name and rank ([#3263](https://github.com/embeddings-benchmark/mteb/pull/3263),
  [`58a81a9`](https://github.com/embeddings-benchmark/mteb/commit/58a81a9cfbe23999ad75ee77c9ffcd4a2552f7f8))


## v1.39.2 (2025-10-02)

### Bug Fixes

- Further specified macro-language code for Norwegian
  ([#3228](https://github.com/embeddings-benchmark/mteb/pull/3228),
  [`a2f7488`](https://github.com/embeddings-benchmark/mteb/commit/a2f748802918a1a0317d483704c657c574dea7a7))

### Documentation

- Update adding benchmark documentation
  ([#3229](https://github.com/embeddings-benchmark/mteb/pull/3229),
  [`50aa4ac`](https://github.com/embeddings-benchmark/mteb/commit/50aa4acb08be810b97b3d3751f2f3d2e8aef8226))


## v1.39.1 (2025-10-01)

### Bug Fixes

- Add rteb submission references and improve descriptions.
  ([#3233](https://github.com/embeddings-benchmark/mteb/pull/3233),
  [`600c290`](https://github.com/embeddings-benchmark/mteb/commit/600c2905e183051a07b5fcb46673f430842e33f8))

- Add submission references for RTEB
  ([#3233](https://github.com/embeddings-benchmark/mteb/pull/3233),
  [`600c290`](https://github.com/embeddings-benchmark/mteb/commit/600c2905e183051a07b5fcb46673f430842e33f8))


## v1.39.0 (2025-10-01)

### Bug Fixes

- Add train and test split for both datasets
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- AttributeError in ColPaliEngineWrapper similarity method
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Correct logic for filtering public tasks in ModelResult class
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Delete kwargs for similarity score in ColPaliEngineWrapper for method behavior
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Formatting and init imports ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Perform citation and code formatting
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Prevent EOS token truncation ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Refactor split create_tables into static Benchmark methods
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Remove unused 'is_public' attribute from TaskResult
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Update mteb task according to feedback
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Update RTEB summary columns ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- **models**: Correct eos token handling in `BMRetrieverWrapper`
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- **models**: Ensure prompt_type is passed to format_instruction
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- **models**: Prevent EOS token truncation for BMRetriever
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

### Chores

- Fix colpali_models similarity handle device
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

### Features

- Add swedish cpc patent classifications to mteb
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

- Officially include RTEB in the leaderboard
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))

### Refactoring

- **models**: Refactor tokenizer setup in `InstructSentenceTransformerWrapper`
  ([#3222](https://github.com/embeddings-benchmark/mteb/pull/3222),
  [`11f9c1d`](https://github.com/embeddings-benchmark/mteb/commit/11f9c1db7a76e24887b9882857789b9f08a21782))


## v1.38.61 (2025-09-29)

### Bug Fixes

- Refactor split create_tables into static Benchmark methods
  ([#3126](https://github.com/embeddings-benchmark/mteb/pull/3126),
  [`cb03bd4`](https://github.com/embeddings-benchmark/mteb/commit/cb03bd4edbdc626401f5877f7630df2ec3c6fc46))


## v1.38.60 (2025-09-27)

### Bug Fixes

- Prevent EOS token truncation ([#3218](https://github.com/embeddings-benchmark/mteb/pull/3218),
  [`f58ac2b`](https://github.com/embeddings-benchmark/mteb/commit/f58ac2bb21b3c3cef7787335025f435adf2b8855))

- **models**: Correct eos token handling in `BMRetrieverWrapper`
  ([#3218](https://github.com/embeddings-benchmark/mteb/pull/3218),
  [`f58ac2b`](https://github.com/embeddings-benchmark/mteb/commit/f58ac2bb21b3c3cef7787335025f435adf2b8855))

- **models**: Prevent EOS token truncation for BMRetriever
  ([#3218](https://github.com/embeddings-benchmark/mteb/pull/3218),
  [`f58ac2b`](https://github.com/embeddings-benchmark/mteb/commit/f58ac2bb21b3c3cef7787335025f435adf2b8855))

### Refactoring

- **models**: Refactor tokenizer setup in `InstructSentenceTransformerWrapper`
  ([#3218](https://github.com/embeddings-benchmark/mteb/pull/3218),
  [`f58ac2b`](https://github.com/embeddings-benchmark/mteb/commit/f58ac2bb21b3c3cef7787335025f435adf2b8855))


## v1.38.59 (2025-09-27)

### Bug Fixes

- Add train and test split for both datasets
  ([#3072](https://github.com/embeddings-benchmark/mteb/pull/3072),
  [`e863bc1`](https://github.com/embeddings-benchmark/mteb/commit/e863bc1db388361996409ebb3950f560ae7b0ff5))

- AttributeError in ColPaliEngineWrapper similarity method
  ([#3177](https://github.com/embeddings-benchmark/mteb/pull/3177),
  [`8c180d4`](https://github.com/embeddings-benchmark/mteb/commit/8c180d4ed5a9d437c7740e746cf960602790be29))

- Delete kwargs for similarity score in ColPaliEngineWrapper for method behavior
  ([#3177](https://github.com/embeddings-benchmark/mteb/pull/3177),
  [`8c180d4`](https://github.com/embeddings-benchmark/mteb/commit/8c180d4ed5a9d437c7740e746cf960602790be29))

- Formatting and init imports ([#3072](https://github.com/embeddings-benchmark/mteb/pull/3072),
  [`e863bc1`](https://github.com/embeddings-benchmark/mteb/commit/e863bc1db388361996409ebb3950f560ae7b0ff5))

- Perform citation and code formatting
  ([#3072](https://github.com/embeddings-benchmark/mteb/pull/3072),
  [`e863bc1`](https://github.com/embeddings-benchmark/mteb/commit/e863bc1db388361996409ebb3950f560ae7b0ff5))

- Update mteb task according to feedback
  ([#3072](https://github.com/embeddings-benchmark/mteb/pull/3072),
  [`e863bc1`](https://github.com/embeddings-benchmark/mteb/commit/e863bc1db388361996409ebb3950f560ae7b0ff5))

### Chores

- Fix colpali_models similarity handle device
  ([#3177](https://github.com/embeddings-benchmark/mteb/pull/3177),
  [`8c180d4`](https://github.com/embeddings-benchmark/mteb/commit/8c180d4ed5a9d437c7740e746cf960602790be29))

### Features

- Add swedish cpc patent classifications to mteb
  ([#3072](https://github.com/embeddings-benchmark/mteb/pull/3072),
  [`e863bc1`](https://github.com/embeddings-benchmark/mteb/commit/e863bc1db388361996409ebb3950f560ae7b0ff5))


## v1.38.58 (2025-09-27)

### Bug Fixes

- Remove trust_remote_code option
  ([`6718290`](https://github.com/embeddings-benchmark/mteb/commit/6718290e644326939a02fe4b2b3bff5b04c6268b))

- **models**: Ensure prompt_type is passed to format_instruction
  ([#3216](https://github.com/embeddings-benchmark/mteb/pull/3216),
  [`82d9e29`](https://github.com/embeddings-benchmark/mteb/commit/82d9e297eb08a1d9c1f6b1cf854bc316ac978fe5))

### Features

- Implement BMREtrieverWrapper based on InstructSentenceTransformerWrapper
  ([`6718290`](https://github.com/embeddings-benchmark/mteb/commit/6718290e644326939a02fe4b2b3bff5b04c6268b))

### Refactoring

- Update training datasets for bmretriever
  ([`6718290`](https://github.com/embeddings-benchmark/mteb/commit/6718290e644326939a02fe4b2b3bff5b04c6268b))


## v1.38.57 (2025-09-21)

### Bug Fixes

- Correct metadata for ArguAna dataset
  ([#3202](https://github.com/embeddings-benchmark/mteb/pull/3202),
  [`90e9f43`](https://github.com/embeddings-benchmark/mteb/commit/90e9f4308bda75726caac0147fe0167893dc18ab))


## v1.38.56 (2025-09-18)

### Bug Fixes

- Add conflicting dependencies to toml
  ([#3191](https://github.com/embeddings-benchmark/mteb/pull/3191),
  [`0cc6802`](https://github.com/embeddings-benchmark/mteb/commit/0cc680242309e9a04339f7a82bff78a997f6cfb1))


## v1.38.55 (2025-09-18)

### Bug Fixes

- Add version check for `embeddinggemma-300m`
  ([#3189](https://github.com/embeddings-benchmark/mteb/pull/3189),
  [`2093798`](https://github.com/embeddings-benchmark/mteb/commit/2093798b41e55f747b852808b0234c848ecc3251))

- Edit ack & sponsors ([#3187](https://github.com/embeddings-benchmark/mteb/pull/3187),
  [`57ffd43`](https://github.com/embeddings-benchmark/mteb/commit/57ffd431aa6eeb1c654793563c29c2b4c002d1d9))

### Chores

- Add 'Patent retrieval' subtype to TaskMetadata
  ([#2946](https://github.com/embeddings-benchmark/mteb/pull/2946),
  [`8f8ed49`](https://github.com/embeddings-benchmark/mteb/commit/8f8ed4973ac065649daeb92a70513e98d093a3ec))

### Features

- **retrieval**: Add DAPFAM patent retrieval tasks (+18 variants)
  ([#2946](https://github.com/embeddings-benchmark/mteb/pull/2946),
  [`8f8ed49`](https://github.com/embeddings-benchmark/mteb/commit/8f8ed4973ac065649daeb92a70513e98d093a3ec))


## v1.38.54 (2025-09-08)

### Bug Fixes

- Add dedicated display for RTEB benchmark results
  ([#3089](https://github.com/embeddings-benchmark/mteb/pull/3089),
  [`53f49ec`](https://github.com/embeddings-benchmark/mteb/commit/53f49ecc4d4c50eb6ac45fe9832208693bb2502d))


## v1.38.53 (2025-09-03)

### Bug Fixes

- Add voyage quantization models ([#3092](https://github.com/embeddings-benchmark/mteb/pull/3092),
  [`9c7804c`](https://github.com/embeddings-benchmark/mteb/commit/9c7804cc786e22be0043f0d8abe09be4e56eedb3))

### Continuous Integration

- Dataset check on new PR ([#3103](https://github.com/embeddings-benchmark/mteb/pull/3103),
  [`6e8eba1`](https://github.com/embeddings-benchmark/mteb/commit/6e8eba15ac7bff94f9a99929b395f76e703d99b2))


## v1.38.52 (2025-09-01)

### Bug Fixes

- Allow closed datasets ([#3059](https://github.com/embeddings-benchmark/mteb/pull/3059),
  [`5844cc7`](https://github.com/embeddings-benchmark/mteb/commit/5844cc7f7b0df6a6c79d65bf8d78c65a5b7de259))


## v1.38.51 (2025-09-01)

### Bug Fixes

- Add @classmethod for @field_validators in TaskMetadata
  ([#3100](https://github.com/embeddings-benchmark/mteb/pull/3100),
  [`4012517`](https://github.com/embeddings-benchmark/mteb/commit/40125175024c82f1efa6e00f272c8bf4a1caf211))


## v1.38.50 (2025-09-01)

### Bug Fixes

- Updating the default batch size calculation in the voyage models
  ([#3091](https://github.com/embeddings-benchmark/mteb/pull/3091),
  [`5851c7a`](https://github.com/embeddings-benchmark/mteb/commit/5851c7a7d22b41927a57457b347daa38c462578a))


## v1.38.49 (2025-08-28)

### Bug Fixes

- Duplicate mteb multilingual variables
  ([#3080](https://github.com/embeddings-benchmark/mteb/pull/3080),
  [`27be671`](https://github.com/embeddings-benchmark/mteb/commit/27be67143393fc6eebe86bf247184d6a37e47bab))

- Improving validate_task_to_prompt_name logs and error messages
  ([#3079](https://github.com/embeddings-benchmark/mteb/pull/3079),
  [`139fc73`](https://github.com/embeddings-benchmark/mteb/commit/139fc737fc012b3d3ae69c700a37f717747ccc24))


## v1.38.48 (2025-08-27)

### Bug Fixes

- Run `ruff check` on all files during ci
  ([#3086](https://github.com/embeddings-benchmark/mteb/pull/3086),
  [`b46b633`](https://github.com/embeddings-benchmark/mteb/commit/b46b633689a800e8147d49ef2fdcebc898f41337))


## v1.38.47 (2025-08-27)

### Bug Fixes

- Add beta version of RTEB related benchmarks
  ([#3048](https://github.com/embeddings-benchmark/mteb/pull/3048),
  [`1541318`](https://github.com/embeddings-benchmark/mteb/commit/1541318c02cd963c7b78778b0a52cfbb03f048ac))


## v1.38.46 (2025-08-25)

### Bug Fixes

- Update revision for qzhou models ([#3069](https://github.com/embeddings-benchmark/mteb/pull/3069),
  [`63a0c60`](https://github.com/embeddings-benchmark/mteb/commit/63a0c6075f56224a040bb61af235ab4cb7d436c9))


## v1.38.45 (2025-08-25)

### Bug Fixes

- Open_clip package validation ([#3073](https://github.com/embeddings-benchmark/mteb/pull/3073),
  [`1f9641a`](https://github.com/embeddings-benchmark/mteb/commit/1f9641aeffb038baa9547b9d0b68c3a22ddf0b18))

### Continuous Integration

- Add stale workflow ([#3066](https://github.com/embeddings-benchmark/mteb/pull/3066),
  [`df719cc`](https://github.com/embeddings-benchmark/mteb/commit/df719cc05308068f47436ba0967a8f59e93340e8))


## v1.38.44 (2025-08-22)

### Bug Fixes

- Ensure that there are always relevant docs attached to query
  ([#3058](https://github.com/embeddings-benchmark/mteb/pull/3058),
  [`9c27f71`](https://github.com/embeddings-benchmark/mteb/commit/9c27f71e44612f190756d41f1fcffeb817b0f3e3))

### Features

- Added date for tasks ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Added description for jinavdr ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Added descriptions per dataset ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Added jinavdr benchmark ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Added max pixel argument for jina models
  ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Adjust jina model for new mteb code
  ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Better reference and fixed comments
  ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Code cleanup ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Corrected bibtex ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Corrected query numbers ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Fixed comments ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Fixed licenses and added bibtex ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Fixed missing metadata and bibtex
  ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Made jinav4 compatible with vidore benchmark
  ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Make colpali run with jinavdr ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Removed print ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))

- Score calculation on cpu ([#2942](https://github.com/embeddings-benchmark/mteb/pull/2942),
  [`cf3e1bb`](https://github.com/embeddings-benchmark/mteb/commit/cf3e1bbe62b53abeec7a932373c52e7d852f97cd))


## v1.38.43 (2025-08-20)

### Bug Fixes

- Add VN-MTEB benchmark and Leaderboard
  ([#2995](https://github.com/embeddings-benchmark/mteb/pull/2995),
  [`0a6e855`](https://github.com/embeddings-benchmark/mteb/commit/0a6e855ccb6d8a86831b64e023829beedca61a3f))

### Continuous Integration

- Temporarily limit pytrec version to "pytrec-eval-terrier>=0.5.6, <0.5.8" to prevent errors
  ([`6fa6efa`](https://github.com/embeddings-benchmark/mteb/commit/6fa6efa4ac52acc4f5487cf1985a6eb33ebe9709))


## v1.38.42 (2025-08-18)

### Bug Fixes

- Jinav4 revision ([#3046](https://github.com/embeddings-benchmark/mteb/pull/3046),
  [`c58b319`](https://github.com/embeddings-benchmark/mteb/commit/c58b319c60b0ce472caeac07a128fe4630fdd033))

- Updated revision for jina-embeddings-v4
  ([#3046](https://github.com/embeddings-benchmark/mteb/pull/3046),
  [`c58b319`](https://github.com/embeddings-benchmark/mteb/commit/c58b319c60b0ce472caeac07a128fe4630fdd033))

### Continuous Integration

- Reduce parallel runs for when checking if a dataset exists
  ([#3035](https://github.com/embeddings-benchmark/mteb/pull/3035),
  [`4aaf47e`](https://github.com/embeddings-benchmark/mteb/commit/4aaf47e16883218a2753f78e40ccce1dbade83e1))

- Updating rerun delays to prevent false positives errors
  ([`e476dc3`](https://github.com/embeddings-benchmark/mteb/commit/e476dc3ec084956f033437f7125c9449e08cf1b5))


## v1.38.41 (2025-08-17)

### Bug Fixes

- Incorrect revision for SNLRetrieval
  ([#3033](https://github.com/embeddings-benchmark/mteb/pull/3033),
  [`5c65913`](https://github.com/embeddings-benchmark/mteb/commit/5c659134996d1b1b0c53ed0589c8cfe4960f13bc))

### Continuous Integration

- Updating rerun delays to prevent false positives errors
  ([`e124b56`](https://github.com/embeddings-benchmark/mteb/commit/e124b56ea87aaf1a1a444f13c27ad2e0798da88a))


## v1.38.40 (2025-08-16)

### Bug Fixes

- Add missing training sets for qzhou
  ([#3023](https://github.com/embeddings-benchmark/mteb/pull/3023),
  [`20bc80c`](https://github.com/embeddings-benchmark/mteb/commit/20bc80c2c2b7e7f99654e3525db1f70425eb4478))


## v1.38.39 (2025-08-03)

### Bug Fixes

- Add new benchmark beRuSciBench along with AbsTaskTextRegression
  ([#2716](https://github.com/embeddings-benchmark/mteb/pull/2716),
  [`36df9ca`](https://github.com/embeddings-benchmark/mteb/commit/36df9ca6d20b450e48b58700ee4988fa95db9515))


## v1.38.38 (2025-07-25)

### Bug Fixes

- Only import SparseEncoder once sentence-transformer version have been checked
  ([#2940](https://github.com/embeddings-benchmark/mteb/pull/2940),
  [`79a43af`](https://github.com/embeddings-benchmark/mteb/commit/79a43af0a5f6df0dcb2f42b1877f1205693024b6))

- Prevent incorrectly passing "selector_state" to `get_benchmark`
  ([#2939](https://github.com/embeddings-benchmark/mteb/pull/2939),
  [`8496ec2`](https://github.com/embeddings-benchmark/mteb/commit/8496ec217578b4cf3bc17cb5f681d43d0884f389))

- Replace with passage ([#2934](https://github.com/embeddings-benchmark/mteb/pull/2934),
  [`5ed6c90`](https://github.com/embeddings-benchmark/mteb/commit/5ed6c909f8d451ea63e4f664d39d80544d2c37d8))

### Continuous Integration

- Bump semantic release
  ([`4ef8571`](https://github.com/embeddings-benchmark/mteb/commit/4ef85716fdc0efe0bd9e4e81a4e331f32cf2060b))

### Documentation

- Update adding_a_dataset.md ([#2947](https://github.com/embeddings-benchmark/mteb/pull/2947),
  [`a78debf`](https://github.com/embeddings-benchmark/mteb/commit/a78debff104aca1c64045da5d0033aca40fb89c8))


## v1.38.37 (2025-07-21)

### Bug Fixes

- Specify revision for opensearch ([#2919](https://github.com/embeddings-benchmark/mteb/pull/2919),
  [`0ac0231`](https://github.com/embeddings-benchmark/mteb/commit/0ac0231c7465aff5f2ebddb1d7624b6d3100930e))


## v1.38.36 (2025-07-20)

### Bug Fixes

- Change `passage` prompt to `document`
  ([#2912](https://github.com/embeddings-benchmark/mteb/pull/2912),
  [`a298fa9`](https://github.com/embeddings-benchmark/mteb/commit/a298fa95d544036efbe6f06af878e10e0e5cf8f9))


## v1.38.35 (2025-07-16)

### Bug Fixes

- Apply review suggestions ([#2893](https://github.com/embeddings-benchmark/mteb/pull/2893),
  [`17be7e5`](https://github.com/embeddings-benchmark/mteb/commit/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6))

- Uniform batch size ([#2893](https://github.com/embeddings-benchmark/mteb/pull/2893),
  [`17be7e5`](https://github.com/embeddings-benchmark/mteb/commit/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6))

- Update code task ([#2893](https://github.com/embeddings-benchmark/mteb/pull/2893),
  [`17be7e5`](https://github.com/embeddings-benchmark/mteb/commit/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6))

- Update colpali engine models ([#2905](https://github.com/embeddings-benchmark/mteb/pull/2905),
  [`9864e2a`](https://github.com/embeddings-benchmark/mteb/commit/9864e2a0fafff094f628adb519108b6d2983e3f4))

- Update error message ([#2893](https://github.com/embeddings-benchmark/mteb/pull/2893),
  [`17be7e5`](https://github.com/embeddings-benchmark/mteb/commit/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6))

- Update max length ([#2893](https://github.com/embeddings-benchmark/mteb/pull/2893),
  [`17be7e5`](https://github.com/embeddings-benchmark/mteb/commit/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6))

### Features

- Add KaLM_Embedding_X_0605 in kalm_models
  ([#2889](https://github.com/embeddings-benchmark/mteb/pull/2889),
  [`9ecac21`](https://github.com/embeddings-benchmark/mteb/commit/9ecac2104bec034a33dfb2043d2ae5fe3becd62a))

- Unify text and image embeddings for all tasks
  ([#2893](https://github.com/embeddings-benchmark/mteb/pull/2893),
  [`17be7e5`](https://github.com/embeddings-benchmark/mteb/commit/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6))


## v1.38.34 (2025-07-10)

### Bug Fixes

- Pin datasets version ([#2892](https://github.com/embeddings-benchmark/mteb/pull/2892),
  [`00c95cf`](https://github.com/embeddings-benchmark/mteb/commit/00c95cff6846a03a478ee43a0b8a69d2846db325))

### Features

- Add KaLM_Embedding_X_0605 in kalm_models
  ([#2853](https://github.com/embeddings-benchmark/mteb/pull/2853),
  [`b67bd04`](https://github.com/embeddings-benchmark/mteb/commit/b67bd043fe7575e91f08c16a16e318fc4baaa1d6))


## v1.38.33 (2025-06-27)

### Bug Fixes

- Add check for code lora
  ([`f1d560a`](https://github.com/embeddings-benchmark/mteb/commit/f1d560af3f86b2b16962e0480cfe241bc59f94fd))

- Apply review comments
  ([`f1d560a`](https://github.com/embeddings-benchmark/mteb/commit/f1d560af3f86b2b16962e0480cfe241bc59f94fd))

- Prompt validation for tasks with `-`
  ([#2846](https://github.com/embeddings-benchmark/mteb/pull/2846),
  [`430357c`](https://github.com/embeddings-benchmark/mteb/commit/430357cdff0cc719da2a1a7c4df65016ba7dfcce))


## v1.38.32 (2025-06-25)

### Bug Fixes

- Update training dataset info of Seed-1.6-embedding model
  ([#2857](https://github.com/embeddings-benchmark/mteb/pull/2857),
  [`a8214e2`](https://github.com/embeddings-benchmark/mteb/commit/a8214e2ed7111340f1d213c43a7829a9ffe83da0))


## v1.38.31 (2025-06-25)

### Bug Fixes

- Format seed_1_6_embedding_models.py with Ruff
  ([`8851bf0`](https://github.com/embeddings-benchmark/mteb/commit/8851bf0a6a261c74dae10f8deb82a840864779df))

- Update model selection for the leaderboard
  ([#2855](https://github.com/embeddings-benchmark/mteb/pull/2855),
  [`9a800d3`](https://github.com/embeddings-benchmark/mteb/commit/9a800d32bd3d84ff220d702e78f6d51ae0e85017))

### Documentation

- Fix some typos in `docs/usage/usage.md`
  ([#2835](https://github.com/embeddings-benchmark/mteb/pull/2835),
  [`774a942`](https://github.com/embeddings-benchmark/mteb/commit/774a9429adc13da3b40e65e63ec32b37a89b1337))


## v1.38.30 (2025-06-16)

### Bug Fixes

- Reuploaded previously unavailable SNL datasets
  ([#2819](https://github.com/embeddings-benchmark/mteb/pull/2819),
  [`c790269`](https://github.com/embeddings-benchmark/mteb/commit/c7902698d76071e8bb21d3b8ec226422c88a6088))


## v1.38.29 (2025-06-11)

### Bug Fixes

- Adding client arg to init method of OpenAI models wrapper
  ([#2803](https://github.com/embeddings-benchmark/mteb/pull/2803),
  [`873ee76`](https://github.com/embeddings-benchmark/mteb/commit/873ee7612e2d655846aae1d0987e18872a613dc1))

- Ensure bright uses the correct revision
  ([#2812](https://github.com/embeddings-benchmark/mteb/pull/2812),
  [`56dc620`](https://github.com/embeddings-benchmark/mteb/commit/56dc62072760f7b9651e24a36bbd12e2b255e06c))


## v1.38.28 (2025-06-10)

### Bug Fixes

- Add adapted_from to Cmedqaretrieval
  ([#2806](https://github.com/embeddings-benchmark/mteb/pull/2806),
  [`fef1837`](https://github.com/embeddings-benchmark/mteb/commit/fef1837e19c00855a59b43979334e72fc9c49674))

### Continuous Integration

- Fix config error for semantic release
  ([#2800](https://github.com/embeddings-benchmark/mteb/pull/2800),
  [`3d8dd9e`](https://github.com/embeddings-benchmark/mteb/commit/3d8dd9e2d35e7a3340f848bbd69e97a3cda45d26))


## v1.38.27 (2025-06-05)

### Bug Fixes

- CachedEmbeddingWrapper issues in both documentation and code
  ([#2779](https://github.com/embeddings-benchmark/mteb/pull/2779),
  [`f7656d5`](https://github.com/embeddings-benchmark/mteb/commit/f7656d50c8be7bb233deab76a305f36bd2b01cc3))


## v1.38.26 (2025-06-05)

### Bug Fixes

- Update Caltech101 datasets to latest revision [v1]
  ([#2778](https://github.com/embeddings-benchmark/mteb/pull/2778),
  [`40f0841`](https://github.com/embeddings-benchmark/mteb/commit/40f08419c8ab7495c3b40e43340dc36d39baa20b))

- Update Caltech101 datasets to latest revision [v2]
  ([#2778](https://github.com/embeddings-benchmark/mteb/pull/2778),
  [`40f0841`](https://github.com/embeddings-benchmark/mteb/commit/40f08419c8ab7495c3b40e43340dc36d39baa20b))


## v1.38.25 (2025-06-05)

### Bug Fixes

- Update giga embeddings ([#2774](https://github.com/embeddings-benchmark/mteb/pull/2774),
  [`5b71e34`](https://github.com/embeddings-benchmark/mteb/commit/5b71e34dfae1232e92ca0d7c8851a1fe3ed15c1e))

### Continuous Integration

- Add new prefixes to releases ([#2766](https://github.com/embeddings-benchmark/mteb/pull/2766),
  [`755a6eb`](https://github.com/embeddings-benchmark/mteb/commit/755a6eb76650a887307547ce7ce199fa62ec12a3))


## v1.38.24 (2025-06-05)

### Bug Fixes

- Add xet support ([#2603](https://github.com/embeddings-benchmark/mteb/pull/2603),
  [`5ffcd63`](https://github.com/embeddings-benchmark/mteb/commit/5ffcd6381c50166c2ddc85d5a6c74654139e17d1))

### Documentation

- Leaderboard simplifications ([#2764](https://github.com/embeddings-benchmark/mteb/pull/2764),
  [`33fddfe`](https://github.com/embeddings-benchmark/mteb/commit/33fddfe89b963bba33e029eef5e9295a5a32e05d))


## v1.38.23 (2025-06-03)

### Bug Fixes

- Add `cadet-embed-base-v1` ([#2727](https://github.com/embeddings-benchmark/mteb/pull/2727),
  [`39a391d`](https://github.com/embeddings-benchmark/mteb/commit/39a391de2fa18380ff38e89c827ab29a7d17aade))

### Continuous Integration

- Delete cache in Model loading test only when model is loaded
  ([#2761](https://github.com/embeddings-benchmark/mteb/pull/2761),
  [`9827ec8`](https://github.com/embeddings-benchmark/mteb/commit/9827ec8d3972538e64d8b67c8b07d3906bda1a87))


## v1.38.22 (2025-06-02)

### Bug Fixes

- Update caltech101 ([#2759](https://github.com/embeddings-benchmark/mteb/pull/2759),
  [`1651f60`](https://github.com/embeddings-benchmark/mteb/commit/1651f60afeed767eba0fd0aae895108080301fee))

- Update Caltech101 to different source
  ([#2759](https://github.com/embeddings-benchmark/mteb/pull/2759),
  [`1651f60`](https://github.com/embeddings-benchmark/mteb/commit/1651f60afeed767eba0fd0aae895108080301fee))

### Documentation

- Updated description of FEVER ([#2759](https://github.com/embeddings-benchmark/mteb/pull/2759),
  [`1651f60`](https://github.com/embeddings-benchmark/mteb/commit/1651f60afeed767eba0fd0aae895108080301fee))

- Updated description of FEVER ([#2745](https://github.com/embeddings-benchmark/mteb/pull/2745),
  [`82f0bb9`](https://github.com/embeddings-benchmark/mteb/commit/82f0bb9b106d27743f3a121faaf072d81c5da2d8))


## v1.38.21 (2025-05-29)

### Bug Fixes

- Correct embedding dimension for bge-m3
  ([#2738](https://github.com/embeddings-benchmark/mteb/pull/2738),
  [`d5ccf10`](https://github.com/embeddings-benchmark/mteb/commit/d5ccf108686254d98c3589b06deaeb3e186ac1b5))


## v1.38.20 (2025-05-28)

### Bug Fixes

- Add colpali models family ([#2721](https://github.com/embeddings-benchmark/mteb/pull/2721),
  [`6303839`](https://github.com/embeddings-benchmark/mteb/commit/630383955232e04575d7e2cb4d32008a05036c55))


## v1.38.19 (2025-05-27)

### Bug Fixes

- Rename display name of VDR ([#2734](https://github.com/embeddings-benchmark/mteb/pull/2734),
  [`b0988e2`](https://github.com/embeddings-benchmark/mteb/commit/b0988e2d20b44745b205409bf6a70732d8542d19))


## v1.38.18 (2025-05-27)

### Bug Fixes

- Promote Persian benchmark to v1 ([#2707](https://github.com/embeddings-benchmark/mteb/pull/2707),
  [`1098109`](https://github.com/embeddings-benchmark/mteb/commit/10981090685d283a89acc40c0de9dc83db10366d))


## v1.38.17 (2025-05-27)

### Bug Fixes

- `IndicQARetrieval` loader ([#2729](https://github.com/embeddings-benchmark/mteb/pull/2729),
  [`c3b66d9`](https://github.com/embeddings-benchmark/mteb/commit/c3b66d96438893082e2e7fe43a9469b11604b83a))


## v1.38.16 (2025-05-26)

### Bug Fixes

- Add vidore v2 benchmarks ([#2713](https://github.com/embeddings-benchmark/mteb/pull/2713),
  [`175de94`](https://github.com/embeddings-benchmark/mteb/commit/175de9447965d91be93599fa9b4cc0ed1910648f))


## v1.38.15 (2025-05-26)

### Bug Fixes

- Ara and ben classification dataset cleaning
  ([#2632](https://github.com/embeddings-benchmark/mteb/pull/2632),
  [`4093099`](https://github.com/embeddings-benchmark/mteb/commit/40930991dc700a41d36285e87a9b1a5dd2933cf1))

- Update Seed1.5-Embedding API ([#2724](https://github.com/embeddings-benchmark/mteb/pull/2724),
  [`1da660e`](https://github.com/embeddings-benchmark/mteb/commit/1da660eca1651a23c67cdbddfccf0d0a3f50775f))


## v1.38.14 (2025-05-23)

### Bug Fixes

- Added potion-multilingual-128M ([#2717](https://github.com/embeddings-benchmark/mteb/pull/2717),
  [`08b72c9`](https://github.com/embeddings-benchmark/mteb/commit/08b72c909887c4c4f53dddf6b29cfb923a9b76d4))

### Documentation

- Fix number of tasks for eng, v2 in docs
  ([#2720](https://github.com/embeddings-benchmark/mteb/pull/2720),
  [`7586624`](https://github.com/embeddings-benchmark/mteb/commit/75866242f7e5d23738562b233d250a46e8f5eaa6))


## v1.38.13 (2025-05-22)

### Bug Fixes

- Integrate `lightonai/GTE-ModernColBERT-v1`
  ([#2708](https://github.com/embeddings-benchmark/mteb/pull/2708),
  [`2b13659`](https://github.com/embeddings-benchmark/mteb/commit/2b13659ccce91140a1c74817fdfc5e0d200f2fa6))


## v1.38.12 (2025-05-21)

### Bug Fixes

- Rename gemini-embedding-exp-03-07 to gemini-embedding-001
  ([#2711](https://github.com/embeddings-benchmark/mteb/pull/2711),
  [`0c0ad05`](https://github.com/embeddings-benchmark/mteb/commit/0c0ad053fc510223e78b06adea04efc82afc661f))


## v1.38.11 (2025-05-19)

### Bug Fixes

- Remove models from the leaderboard
  ([#2705](https://github.com/embeddings-benchmark/mteb/pull/2705),
  [`78080cd`](https://github.com/embeddings-benchmark/mteb/commit/78080cd8aff671845dda6ffadd276607eb4d91b8))

### Documentation

- Updated the PR template and improved submission docs
  ([#2704](https://github.com/embeddings-benchmark/mteb/pull/2704),
  [`835f6e6`](https://github.com/embeddings-benchmark/mteb/commit/835f6e67d1a27b5d6c0ed4e7e178e5444fbdd071))


## v1.38.10 (2025-05-19)

### Bug Fixes

- Ensure that optional dependencies are compatible and if not state it
  ([#2706](https://github.com/embeddings-benchmark/mteb/pull/2706),
  [`7222458`](https://github.com/embeddings-benchmark/mteb/commit/72224581553e2d4c753be61a0f7cda1c565c34ca))

- Only install mteb into site packages
  ([#2618](https://github.com/embeddings-benchmark/mteb/pull/2618),
  [`1c803a1`](https://github.com/embeddings-benchmark/mteb/commit/1c803a135f2201d6a466d55e111e5924631caa2a))


## v1.38.9 (2025-05-09)

### Bug Fixes

- `MTEB(Code, v1)` languages ([#2679](https://github.com/embeddings-benchmark/mteb/pull/2679),
  [`40ce571`](https://github.com/embeddings-benchmark/mteb/commit/40ce5716f6d586c9c159601cec624c4aba4569cc))


## v1.38.8 (2025-05-09)

### Bug Fixes

- Allow empty string for openai models
  ([#2676](https://github.com/embeddings-benchmark/mteb/pull/2676),
  [`6f0b08d`](https://github.com/embeddings-benchmark/mteb/commit/6f0b08d63d997f6ced98b4817aafc1414496a2ff))

- Allow empty string in openai models
  ([#2676](https://github.com/embeddings-benchmark/mteb/pull/2676),
  [`6f0b08d`](https://github.com/embeddings-benchmark/mteb/commit/6f0b08d63d997f6ced98b4817aafc1414496a2ff))


## v1.38.7 (2025-05-07)

### Bug Fixes

- Update datasets wich can't be loaded with `datasets>=3.0`
  ([#2661](https://github.com/embeddings-benchmark/mteb/pull/2661),
  [`1ba6716`](https://github.com/embeddings-benchmark/mteb/commit/1ba671665919ce50532c8e38b06271c370b0d969))


## v1.38.6 (2025-05-06)

### Bug Fixes

- SIB200 machine translated > human translated
  ([#2665](https://github.com/embeddings-benchmark/mteb/pull/2665),
  [`ebdf0ca`](https://github.com/embeddings-benchmark/mteb/commit/ebdf0cafb61a997ed66becb10cd0546f82dffbf3))


## v1.38.5 (2025-05-05)

### Bug Fixes

- Update VisualSTS Aggregate task modalities
  ([#2597](https://github.com/embeddings-benchmark/mteb/pull/2597),
  [`671dc04`](https://github.com/embeddings-benchmark/mteb/commit/671dc04c7116c1da3531e93a43dbf62dd5eb6206))


## v1.38.4 (2025-05-02)

### Bug Fixes

- Removed missing dataset for MTEB(Multilingual) and bumped version
  ([`f063638`](https://github.com/embeddings-benchmark/mteb/commit/f063638aece517e038951b371820f0a60d91a219))


## v1.38.3 (2025-05-01)

### Bug Fixes

- Add WebSSL models ([#2604](https://github.com/embeddings-benchmark/mteb/pull/2604),
  [`afb72ac`](https://github.com/embeddings-benchmark/mteb/commit/afb72ac1e6d96cb63439b9a7032bf4558934b9f4))


## v1.38.2 (2025-04-27)

### Bug Fixes

- Add Encodechka benchmark ([#2561](https://github.com/embeddings-benchmark/mteb/pull/2561),
  [`0737e78`](https://github.com/embeddings-benchmark/mteb/commit/0737e78c0c9a4c18fb604613c32f78791ad44156))


## v1.38.1 (2025-04-20)

### Bug Fixes

- Jasper models embeddings having nan values
  ([#2481](https://github.com/embeddings-benchmark/mteb/pull/2481),
  [`f7072d5`](https://github.com/embeddings-benchmark/mteb/commit/f7072d51f1cbfbf4c4accdaabc2b6a85cfbf3a51))


## v1.38.0 (2025-04-17)

### Features

- UI Overhaul ([#2549](https://github.com/embeddings-benchmark/mteb/pull/2549),
  [`0ab947b`](https://github.com/embeddings-benchmark/mteb/commit/0ab947bd5f73dc4fdae3f672d5dd599b6b6598b3))


## v1.37.0 (2025-04-16)

### Bug Fixes

- Added dataframe utilities to BenchmarkResults
  ([#2542](https://github.com/embeddings-benchmark/mteb/pull/2542),
  [`8fe5742`](https://github.com/embeddings-benchmark/mteb/commit/8fe5742b150abb0bfa40d7c7e208cfcd58669be4))

- Me5 trainind data config to include xquad dataset
  ([#2552](https://github.com/embeddings-benchmark/mteb/pull/2552),
  [`1f82b59`](https://github.com/embeddings-benchmark/mteb/commit/1f82b596d79549bbfcf01c884ec7af670e11023a))

- ME5_TRAINING_DATA format ([#2552](https://github.com/embeddings-benchmark/mteb/pull/2552),
  [`1f82b59`](https://github.com/embeddings-benchmark/mteb/commit/1f82b596d79549bbfcf01c884ec7af670e11023a))

### Features

- Added dataframe utilities to BenchmarkResults
  ([#2542](https://github.com/embeddings-benchmark/mteb/pull/2542),
  [`8fe5742`](https://github.com/embeddings-benchmark/mteb/commit/8fe5742b150abb0bfa40d7c7e208cfcd58669be4))


## v1.36.41 (2025-04-15)

### Bug Fixes

- Update package requirements in JinaWrapper for einops and flash_attn
  ([#2548](https://github.com/embeddings-benchmark/mteb/pull/2548),
  [`caa6e70`](https://github.com/embeddings-benchmark/mteb/commit/caa6e7020ffedf91c4c2de9254ca531d79b8b97d))

- Update requirements in JinaWrapper
  ([#2548](https://github.com/embeddings-benchmark/mteb/pull/2548),
  [`caa6e70`](https://github.com/embeddings-benchmark/mteb/commit/caa6e7020ffedf91c4c2de9254ca531d79b8b97d))


## v1.36.40 (2025-04-15)

### Bug Fixes

- CacheWrapper per task ([#2467](https://github.com/embeddings-benchmark/mteb/pull/2467),
  [`67881c4`](https://github.com/embeddings-benchmark/mteb/commit/67881c470dc6ee1be4373d11fba2a446d8a09caf))

### Documentation

- Add MIEB citation in benchmarks ([#2544](https://github.com/embeddings-benchmark/mteb/pull/2544),
  [`99c22b5`](https://github.com/embeddings-benchmark/mteb/commit/99c22b5268ae886072203114cac0fbc74bac537b))

### Features

- CacheWrapper per task ([#2467](https://github.com/embeddings-benchmark/mteb/pull/2467),
  [`67881c4`](https://github.com/embeddings-benchmark/mteb/commit/67881c470dc6ee1be4373d11fba2a446d8a09caf))


## v1.36.39 (2025-04-14)

### Bug Fixes

- Update mteb.get_tasks with an exclude_aggregate parameter to exclude aggregate tasks
  ([#2536](https://github.com/embeddings-benchmark/mteb/pull/2536),
  [`c52690d`](https://github.com/embeddings-benchmark/mteb/commit/c52690d565e7acca794f4226a042c81eb731253f))


## v1.36.38 (2025-04-08)

### Bug Fixes

- Validate lang code in ModelMeta ([#2499](https://github.com/embeddings-benchmark/mteb/pull/2499),
  [`2d15895`](https://github.com/embeddings-benchmark/mteb/commit/2d15895ab4f1ebcf37025c610085058abd5497a5))

### Continuous Integration

- Refactor TaskMetadata eval langs test
  ([#2501](https://github.com/embeddings-benchmark/mteb/pull/2501),
  [`cb2825c`](https://github.com/embeddings-benchmark/mteb/commit/cb2825ce21a6308fe51fbda5384f6d134b1e3cb1))


## v1.36.37 (2025-04-04)

### Bug Fixes

- Ignore datasets not available in tests
  ([#2484](https://github.com/embeddings-benchmark/mteb/pull/2484),
  [`8d87f41`](https://github.com/embeddings-benchmark/mteb/commit/8d87f41aab8c4ef0f7d9de1054d2faeec539575b))


## v1.36.36 (2025-04-04)

### Bug Fixes

- Add prompt to NanoDBPedia ([#2486](https://github.com/embeddings-benchmark/mteb/pull/2486),
  [`7d4302e`](https://github.com/embeddings-benchmark/mteb/commit/7d4302e2022e4202f3bcdc6c31cf097d2209b5a7))

### Continuous Integration

- Run dataset loading only when pushing to main
  ([#2480](https://github.com/embeddings-benchmark/mteb/pull/2480),
  [`17b53b4`](https://github.com/embeddings-benchmark/mteb/commit/17b53b4f586a62a0b675082fabea09be813e27df))


## v1.36.35 (2025-04-02)

### Bug Fixes

- E5 instruct now listed as sbert compatible
  ([#2475](https://github.com/embeddings-benchmark/mteb/pull/2475),
  [`6c8c8d2`](https://github.com/embeddings-benchmark/mteb/commit/6c8c8d240d710cdf662d0b8d095d85966affaeb5))


## v1.36.34 (2025-04-01)

### Bug Fixes

- Add nb_sbert model ([#2339](https://github.com/embeddings-benchmark/mteb/pull/2339),
  [`c617598`](https://github.com/embeddings-benchmark/mteb/commit/c61759807fa8b32d5969598c127139ff38f96062))

- Adds family of NeuML/pubmedbert-base-embedding models
  ([#2443](https://github.com/embeddings-benchmark/mteb/pull/2443),
  [`f293d8b`](https://github.com/embeddings-benchmark/mteb/commit/f293d8bf65470b0e431c39f18082fe3845fe233e))

- Attribute model_name ([#2443](https://github.com/embeddings-benchmark/mteb/pull/2443),
  [`f293d8b`](https://github.com/embeddings-benchmark/mteb/commit/f293d8bf65470b0e431c39f18082fe3845fe233e))

- Changes requested in PR 2443 ([#2443](https://github.com/embeddings-benchmark/mteb/pull/2443),
  [`f293d8b`](https://github.com/embeddings-benchmark/mteb/commit/f293d8bf65470b0e431c39f18082fe3845fe233e))

- Fixed commit hash for pubmed_bert model2vec models
  ([#2443](https://github.com/embeddings-benchmark/mteb/pull/2443),
  [`f293d8b`](https://github.com/embeddings-benchmark/mteb/commit/f293d8bf65470b0e431c39f18082fe3845fe233e))

- Make lint ([#2339](https://github.com/embeddings-benchmark/mteb/pull/2339),
  [`c617598`](https://github.com/embeddings-benchmark/mteb/commit/c61759807fa8b32d5969598c127139ff38f96062))

### Continuous Integration

- Cache `~/.cache/huggingface` ([#2464](https://github.com/embeddings-benchmark/mteb/pull/2464),
  [`d11934f`](https://github.com/embeddings-benchmark/mteb/commit/d11934fd03655011527045afca02adacac0a8d0d))

- Cache ~/.cache/huggingface ([#2464](https://github.com/embeddings-benchmark/mteb/pull/2464),
  [`d11934f`](https://github.com/embeddings-benchmark/mteb/commit/d11934fd03655011527045afca02adacac0a8d0d))

### Features

- Added pubmedbert model2vec models
  ([#2443](https://github.com/embeddings-benchmark/mteb/pull/2443),
  [`f293d8b`](https://github.com/embeddings-benchmark/mteb/commit/f293d8bf65470b0e431c39f18082fe3845fe233e))


## v1.36.33 (2025-03-26)

### Bug Fixes

- Add model specific dependencies in pyproject.toml
  ([#2424](https://github.com/embeddings-benchmark/mteb/pull/2424),
  [`8a024be`](https://github.com/embeddings-benchmark/mteb/commit/8a024be2e69a4f42c8cc281a5af21e08d69ff0ff))

### Documentation

- Typos; Standardize spacing; Chronological order
  ([#2436](https://github.com/embeddings-benchmark/mteb/pull/2436),
  [`0db0a20`](https://github.com/embeddings-benchmark/mteb/commit/0db0a20d48d422d3f3db25d50b73ada2525ccc19))


## v1.36.32 (2025-03-23)

### Bug Fixes

- Add VDR Multilingual Dataset ([#2408](https://github.com/embeddings-benchmark/mteb/pull/2408),
  [`9d9b0b4`](https://github.com/embeddings-benchmark/mteb/commit/9d9b0b4329f29351b4c6dc596845205a97893744))


## v1.36.31 (2025-03-23)

### Bug Fixes

- Add option to remove benchmark from leaderboard
  ([#2417](https://github.com/embeddings-benchmark/mteb/pull/2417),
  [`e8faf3f`](https://github.com/embeddings-benchmark/mteb/commit/e8faf3fb1c132d17caba4da5b6f570739f005a91))

- Add option to remove leaderboard from leaderboard
  ([#2417](https://github.com/embeddings-benchmark/mteb/pull/2417),
  [`e8faf3f`](https://github.com/embeddings-benchmark/mteb/commit/e8faf3fb1c132d17caba4da5b6f570739f005a91))


## v1.36.30 (2025-03-22)

### Bug Fixes

- Add validation to model_name in `ModelMeta`
  ([#2404](https://github.com/embeddings-benchmark/mteb/pull/2404),
  [`095851f`](https://github.com/embeddings-benchmark/mteb/commit/095851f66fbb87560fd677b04332b57293c3da61))


## v1.36.29 (2025-03-22)

### Bug Fixes

- Major updates to docs + make mieb dep optional
  ([#2397](https://github.com/embeddings-benchmark/mteb/pull/2397),
  [`cae1575`](https://github.com/embeddings-benchmark/mteb/commit/cae157558e984c8dd6472e0b65df786b8cd217ae))

- Major updates to documentation ([#2397](https://github.com/embeddings-benchmark/mteb/pull/2397),
  [`cae1575`](https://github.com/embeddings-benchmark/mteb/commit/cae157558e984c8dd6472e0b65df786b8cd217ae))

- Make torchvision optional ([#2397](https://github.com/embeddings-benchmark/mteb/pull/2397),
  [`cae1575`](https://github.com/embeddings-benchmark/mteb/commit/cae157558e984c8dd6472e0b65df786b8cd217ae))

- Minor style changes ([#2397](https://github.com/embeddings-benchmark/mteb/pull/2397),
  [`cae1575`](https://github.com/embeddings-benchmark/mteb/commit/cae157558e984c8dd6472e0b65df786b8cd217ae))

- Minor style changes ([#2396](https://github.com/embeddings-benchmark/mteb/pull/2396),
  [`8be95b7`](https://github.com/embeddings-benchmark/mteb/commit/8be95b78c6de0023d0429bb0af4ffc4b202584cf))

- Renaming Zeroshot -> ZeroShot ([#2397](https://github.com/embeddings-benchmark/mteb/pull/2397),
  [`cae1575`](https://github.com/embeddings-benchmark/mteb/commit/cae157558e984c8dd6472e0b65df786b8cd217ae))

- Renaming Zeroshot -> ZeroShot ([#2396](https://github.com/embeddings-benchmark/mteb/pull/2396),
  [`8be95b7`](https://github.com/embeddings-benchmark/mteb/commit/8be95b78c6de0023d0429bb0af4ffc4b202584cf))


## v1.36.28 (2025-03-20)

### Bug Fixes

- Update AmazonPolarityClassification license
  ([#2402](https://github.com/embeddings-benchmark/mteb/pull/2402),
  [`cf84a79`](https://github.com/embeddings-benchmark/mteb/commit/cf84a79085c1499235c519d5fb913019870c060e))


## v1.36.27 (2025-03-20)

### Bug Fixes

- Renaming Zeroshot -> ZeroShot ([#2395](https://github.com/embeddings-benchmark/mteb/pull/2395),
  [`e7b04a6`](https://github.com/embeddings-benchmark/mteb/commit/e7b04a67c0c4010753c5f4bd0efa0d3e8aa21865))


## v1.36.26 (2025-03-18)

### Bug Fixes

- Convert brightretrieval to two tasks
  ([#2334](https://github.com/embeddings-benchmark/mteb/pull/2334),
  [`cf26764`](https://github.com/embeddings-benchmark/mteb/commit/cf26764f62efbf62c6bdac3bc545a75166de8d85))

- Ensure BrightRetrieval is valid to run
  ([#2334](https://github.com/embeddings-benchmark/mteb/pull/2334),
  [`cf26764`](https://github.com/embeddings-benchmark/mteb/commit/cf26764f62efbf62c6bdac3bc545a75166de8d85))


## v1.36.25 (2025-03-17)

### Bug Fixes

- Pin gradio dependency to ensure leaderboards works
  ([#2387](https://github.com/embeddings-benchmark/mteb/pull/2387),
  [`43b5b69`](https://github.com/embeddings-benchmark/mteb/commit/43b5b69fe85c6ed99e0843fc1bac465391c1d471))


## v1.36.24 (2025-03-17)

### Bug Fixes

- Added b1ade_models.py ([#2386](https://github.com/embeddings-benchmark/mteb/pull/2386),
  [`60c0a75`](https://github.com/embeddings-benchmark/mteb/commit/60c0a750454a2d920158a6d135ac2950bb8f8412))

- B1ade ([#2386](https://github.com/embeddings-benchmark/mteb/pull/2386),
  [`60c0a75`](https://github.com/embeddings-benchmark/mteb/commit/60c0a750454a2d920158a6d135ac2950bb8f8412))

- Missing import and formatting ([#2386](https://github.com/embeddings-benchmark/mteb/pull/2386),
  [`60c0a75`](https://github.com/embeddings-benchmark/mteb/commit/60c0a750454a2d920158a6d135ac2950bb8f8412))


## v1.36.23 (2025-03-17)

### Bug Fixes

- Reduce logging and Warnings ([#2349](https://github.com/embeddings-benchmark/mteb/pull/2349),
  [`99eb94b`](https://github.com/embeddings-benchmark/mteb/commit/99eb94beed6156fe1efa6af87925de31cbc7077f))


## v1.36.22 (2025-03-13)

### Bug Fixes

- Add `trust_remote_code` to MIRACLRetrieval
  ([#2346](https://github.com/embeddings-benchmark/mteb/pull/2346),
  [`75961a0`](https://github.com/embeddings-benchmark/mteb/commit/75961a0d0b47fc2305cf0cdd6bc9dbe54db489e9))

- Add `trust_remote_code` to MIRACLRetrieval
  ([`fc329ba`](https://github.com/embeddings-benchmark/mteb/commit/fc329ba869e7773ff70c349a662860a0f068e125))

- Correctly pass trust remote code to Miracl
  ([#2346](https://github.com/embeddings-benchmark/mteb/pull/2346),
  [`75961a0`](https://github.com/embeddings-benchmark/mteb/commit/75961a0d0b47fc2305cf0cdd6bc9dbe54db489e9))

- Correctly pass trust remote code to Miracl
  ([`d6d8552`](https://github.com/embeddings-benchmark/mteb/commit/d6d8552c664ec530eb4ea87fa4d35d8a57c8bfef))

- Ensure MIRACL pass trust_remote_code
  ([#2346](https://github.com/embeddings-benchmark/mteb/pull/2346),
  [`75961a0`](https://github.com/embeddings-benchmark/mteb/commit/75961a0d0b47fc2305cf0cdd6bc9dbe54db489e9))


## v1.36.21 (2025-03-13)

### Bug Fixes

- Add `trust_remote_code` to MIRACLRetrieval
  ([#2344](https://github.com/embeddings-benchmark/mteb/pull/2344),
  [`2d45653`](https://github.com/embeddings-benchmark/mteb/commit/2d4565308329e64d56db5486302791e1bf59da30))


## v1.36.20 (2025-03-12)

### Bug Fixes

- Add WebFAQ bitext mining tasks ([#2326](https://github.com/embeddings-benchmark/mteb/pull/2326),
  [`04cfe4d`](https://github.com/embeddings-benchmark/mteb/commit/04cfe4df42fb60804d61945ddef8bcfbaac9e065))


## v1.36.19 (2025-03-11)

### Bug Fixes

- Add ModelMeta rubert-mini-frida, BERTA
  ([#2330](https://github.com/embeddings-benchmark/mteb/pull/2330),
  [`ae83b5f`](https://github.com/embeddings-benchmark/mteb/commit/ae83b5f4b67790f07ba6c5bd9a21f7c893b63292))

### Documentation

- Fix typos
  ([`849efbb`](https://github.com/embeddings-benchmark/mteb/commit/849efbb0126f1471bf1ee15b5c9ab54711ff94e2))


## v1.36.18 (2025-03-11)

### Bug Fixes

- Add annotation models for stella zh
  ([#2277](https://github.com/embeddings-benchmark/mteb/pull/2277),
  [`034da4d`](https://github.com/embeddings-benchmark/mteb/commit/034da4d7ebc9a01b549355d1717e734560cc2c13))


## v1.36.17 (2025-03-11)

### Bug Fixes

- Remove syntax warnings occuring in python 3.12
  ([#2325](https://github.com/embeddings-benchmark/mteb/pull/2325),
  [`fc176ad`](https://github.com/embeddings-benchmark/mteb/commit/fc176ad3df466ccb4a31a0f9b90aad1de96ca87a))

- Remove SyntaxWarnings in py312 ([#2325](https://github.com/embeddings-benchmark/mteb/pull/2325),
  [`fc176ad`](https://github.com/embeddings-benchmark/mteb/commit/fc176ad3df466ccb4a31a0f9b90aad1de96ca87a))

- Resolve conflicting dependencies ([#2325](https://github.com/embeddings-benchmark/mteb/pull/2325),
  [`fc176ad`](https://github.com/embeddings-benchmark/mteb/commit/fc176ad3df466ccb4a31a0f9b90aad1de96ca87a))


## v1.36.16 (2025-03-11)

### Bug Fixes

- Resolve conflicting dependencies ([#2323](https://github.com/embeddings-benchmark/mteb/pull/2323),
  [`8f6bf45`](https://github.com/embeddings-benchmark/mteb/commit/8f6bf4558c834cfe2e0bae2bae0b7458d60ae869))


## v1.36.15 (2025-03-11)

### Bug Fixes

- Update voyage name to include Org.
  ([#2322](https://github.com/embeddings-benchmark/mteb/pull/2322),
  [`746b411`](https://github.com/embeddings-benchmark/mteb/commit/746b411c4d76a6e8596379b01e05fea05179c95b))


## v1.36.14 (2025-03-10)

### Bug Fixes

- Bug in voyage implementation ([#2304](https://github.com/embeddings-benchmark/mteb/pull/2304),
  [`6193db1`](https://github.com/embeddings-benchmark/mteb/commit/6193db16cd95232ef742248b8fb374ca108365d7))

- Fix bug in voyage implementation ([#2304](https://github.com/embeddings-benchmark/mteb/pull/2304),
  [`6193db1`](https://github.com/embeddings-benchmark/mteb/commit/6193db16cd95232ef742248b8fb374ca108365d7))

### Continuous Integration

- Add pre-commit hook ([#2194](https://github.com/embeddings-benchmark/mteb/pull/2194),
  [`5b30d84`](https://github.com/embeddings-benchmark/mteb/commit/5b30d84c72c16efa9bf1e9ff5162d401799ab995))


## v1.36.13 (2025-03-09)

### Bug Fixes

- Add `ModelMeta` license & custom validations
  ([#2293](https://github.com/embeddings-benchmark/mteb/pull/2293),
  [`5dce601`](https://github.com/embeddings-benchmark/mteb/commit/5dce60125e56b1719d33f4c4a3e4a35ea7da7bff))


## v1.36.12 (2025-03-09)

### Bug Fixes

- Added Filter Modality ([#2262](https://github.com/embeddings-benchmark/mteb/pull/2262),
  [`f840f7d`](https://github.com/embeddings-benchmark/mteb/commit/f840f7d51fce640d37507f2457636db3c535b05a))


## v1.36.11 (2025-03-08)

### Bug Fixes

- Run remaining MIEB desc stats ([#2288](https://github.com/embeddings-benchmark/mteb/pull/2288),
  [`e628bce`](https://github.com/embeddings-benchmark/mteb/commit/e628bceb5dd05cf9a71c69cee94cf17b9da64c9c))

### Continuous Integration

- Run test_dataset_on_hf separately
  ([#2201](https://github.com/embeddings-benchmark/mteb/pull/2201),
  [`55b9a0e`](https://github.com/embeddings-benchmark/mteb/commit/55b9a0efef5198c37fd39af2b225b721dbba1b2f))


## v1.36.10 (2025-03-07)

### Bug Fixes

- Formatting issue in Performance Plot
  ([#2237](https://github.com/embeddings-benchmark/mteb/pull/2237),
  [`c456111`](https://github.com/embeddings-benchmark/mteb/commit/c456111b98a1e0e4be13f739ce360404f2a74461))


## v1.36.9 (2025-03-07)

### Bug Fixes

- Add WebFAQ Retrieval dataset ([#2236](https://github.com/embeddings-benchmark/mteb/pull/2236),
  [`9d6e1a9`](https://github.com/embeddings-benchmark/mteb/commit/9d6e1a9ed97d29789f87017f4d51c23cd7aa6213))

### Documentation

- Update description of EURLex ([#2231](https://github.com/embeddings-benchmark/mteb/pull/2231),
  [`3a9d271`](https://github.com/embeddings-benchmark/mteb/commit/3a9d271e9cacfd4c4a3072f09a08630008f44b11))


## v1.36.8 (2025-03-03)

### Bug Fixes

- Add mixedbread ([#2210](https://github.com/embeddings-benchmark/mteb/pull/2210),
  [`ee514cb`](https://github.com/embeddings-benchmark/mteb/commit/ee514cb0d491809f12d581d5f2a5a8c596a8269a))

- Add training data annotations to uderver-bloom models
  ([#2210](https://github.com/embeddings-benchmark/mteb/pull/2210),
  [`ee514cb`](https://github.com/embeddings-benchmark/mteb/commit/ee514cb0d491809f12d581d5f2a5a8c596a8269a))


## v1.36.7 (2025-03-03)

### Bug Fixes

- More training data annotations ([#2220](https://github.com/embeddings-benchmark/mteb/pull/2220),
  [`2dd1391`](https://github.com/embeddings-benchmark/mteb/commit/2dd13912b6e5c7c8be93aeb9f7dd873671309961))


## v1.36.6 (2025-03-03)

### Bug Fixes

- Fixed leaderboard crash ([#2221](https://github.com/embeddings-benchmark/mteb/pull/2221),
  [`761a174`](https://github.com/embeddings-benchmark/mteb/commit/761a17451bdc6cbe5baeca0ccffe38db9f9a1696))


## v1.36.5 (2025-02-28)

### Bug Fixes

- Alphabetical ordering of tasks in dropdowns
  ([#2191](https://github.com/embeddings-benchmark/mteb/pull/2191),
  [`fee6fc0`](https://github.com/embeddings-benchmark/mteb/commit/fee6fc065508cae0a2d34dae478d5423bcd2e155))


## v1.36.4 (2025-02-28)

### Bug Fixes

- Update ru models annotation ([#2181](https://github.com/embeddings-benchmark/mteb/pull/2181),
  [`3325f7e`](https://github.com/embeddings-benchmark/mteb/commit/3325f7e661089df9e3ff6ca38786e855054f8df7))


## v1.36.3 (2025-02-28)

### Bug Fixes

- Added training data for sentence-croissant
  ([#2189](https://github.com/embeddings-benchmark/mteb/pull/2189),
  [`0901cf6`](https://github.com/embeddings-benchmark/mteb/commit/0901cf68b0a7841f350a05a5c87345c557604fd8))


## v1.36.2 (2025-02-28)

### Bug Fixes

- Added training data annotation for MMLW models
  ([#2188](https://github.com/embeddings-benchmark/mteb/pull/2188),
  [`0307102`](https://github.com/embeddings-benchmark/mteb/commit/03071024e4e20845bf73809ad55711f71ef953a9))

- Added training data annotations to MXBAI
  ([#2185](https://github.com/embeddings-benchmark/mteb/pull/2185),
  [`1b23d4e`](https://github.com/embeddings-benchmark/mteb/commit/1b23d4e73191c54f9caac91ff41cd64835cc2a37))

- Update MTEB(Scandinavian) to use new DanFEVER
  ([#2180](https://github.com/embeddings-benchmark/mteb/pull/2180),
  [`7daf893`](https://github.com/embeddings-benchmark/mteb/commit/7daf89395b4ee04d5b2adda582e3bd82df0a0d47))


## v1.36.1 (2025-02-27)

### Bug Fixes

- Add more training data annotations
  ([#2178](https://github.com/embeddings-benchmark/mteb/pull/2178),
  [`1959c73`](https://github.com/embeddings-benchmark/mteb/commit/1959c73bb1ce2791e1d171e4c40077079817efc3))

- Add training data for Bilingual Embeddings
  ([#2178](https://github.com/embeddings-benchmark/mteb/pull/2178),
  [`1959c73`](https://github.com/embeddings-benchmark/mteb/commit/1959c73bb1ce2791e1d171e4c40077079817efc3))

- Update training datasets and revision for jina models
  ([#2179](https://github.com/embeddings-benchmark/mteb/pull/2179),
  [`62b33f2`](https://github.com/embeddings-benchmark/mteb/commit/62b33f26c5550e2b9e9fcd78fe78c4fcb2f698aa))

### Features

- Update training datasets and revision for jina models
  ([#2179](https://github.com/embeddings-benchmark/mteb/pull/2179),
  [`62b33f2`](https://github.com/embeddings-benchmark/mteb/commit/62b33f26c5550e2b9e9fcd78fe78c4fcb2f698aa))


## v1.36.0 (2025-02-27)

### Features

- Add MIEB and MIEB-lite as benchmarks
  ([#2035](https://github.com/embeddings-benchmark/mteb/pull/2035),
  [`dea231b`](https://github.com/embeddings-benchmark/mteb/commit/dea231ba7be4343c37f4b56f2df4d67cb005df23))


## v1.35.2 (2025-02-26)

### Bug Fixes

- Add Training data annotations ([#2173](https://github.com/embeddings-benchmark/mteb/pull/2173),
  [`6cc1822`](https://github.com/embeddings-benchmark/mteb/commit/6cc18224970de37201a1a9d861a5a5f849bc21c0))


## v1.35.1 (2025-02-25)

### Bug Fixes

- Incorrect annotations for Mistral-based embedding models
  ([#2157](https://github.com/embeddings-benchmark/mteb/pull/2157),
  [`565e29c`](https://github.com/embeddings-benchmark/mteb/commit/565e29c2f58b0f18f3ee1f0fbffa4f0f8e8e3400))


## v1.35.0 (2025-02-24)

### Features

- Add Qodo-Embed-1-7B model metadata and rename existing model
  ([#2146](https://github.com/embeddings-benchmark/mteb/pull/2146),
  [`0e624b2`](https://github.com/embeddings-benchmark/mteb/commit/0e624b26b168c62afc6e0bbc2c89071a6c80b118))


## v1.34.30 (2025-02-24)

### Bug Fixes

- Add annotations for Voyage exp ([#2144](https://github.com/embeddings-benchmark/mteb/pull/2144),
  [`8538e93`](https://github.com/embeddings-benchmark/mteb/commit/8538e9346f1193ec4f8ba6d00ab3c6d8c13d1884))

- Update NVIDIA-Embed training data
  ([#2144](https://github.com/embeddings-benchmark/mteb/pull/2144),
  [`8538e93`](https://github.com/embeddings-benchmark/mteb/commit/8538e9346f1193ec4f8ba6d00ab3c6d8c13d1884))


## v1.34.29 (2025-02-24)

### Bug Fixes

- Add adapted_from field to Qodo model metadata
  ([#2137](https://github.com/embeddings-benchmark/mteb/pull/2137),
  [`17a120a`](https://github.com/embeddings-benchmark/mteb/commit/17a120a6a96e0c2c4f918a9a1ad21ddd64b035a4))

- Add Qodo models to overview imports
  ([#2137](https://github.com/embeddings-benchmark/mteb/pull/2137),
  [`17a120a`](https://github.com/embeddings-benchmark/mteb/commit/17a120a6a96e0c2c4f918a9a1ad21ddd64b035a4))

- Update NVIDIA-Embed training data
  ([#2143](https://github.com/embeddings-benchmark/mteb/pull/2143),
  [`760fcaf`](https://github.com/embeddings-benchmark/mteb/commit/760fcaf8fdbcd9e01d896330b2cef906066435b1))

### Features

- Add Qodo-Embed-1-1.5B model metadata
  ([#2137](https://github.com/embeddings-benchmark/mteb/pull/2137),
  [`17a120a`](https://github.com/embeddings-benchmark/mteb/commit/17a120a6a96e0c2c4f918a9a1ad21ddd64b035a4))

### Testing

- Fix dataset availability test ([#2141](https://github.com/embeddings-benchmark/mteb/pull/2141),
  [`0163342`](https://github.com/embeddings-benchmark/mteb/commit/0163342850c1f479c4bbc2eaceba4b188bcdeb7d))


## v1.34.28 (2025-02-21)

### Bug Fixes

- Add 2 new Static Sentence Transformer models
  ([#2112](https://github.com/embeddings-benchmark/mteb/pull/2112),
  [`e7735b2`](https://github.com/embeddings-benchmark/mteb/commit/e7735b25700a1810ac9e62e009e1b54482a30334))


## v1.34.27 (2025-02-21)

### Bug Fixes

- Update e5 instruct training data ([#2129](https://github.com/embeddings-benchmark/mteb/pull/2129),
  [`44cfa9b`](https://github.com/embeddings-benchmark/mteb/commit/44cfa9b4edc448aaf078b71b919f331633c3917a))

### Documentation

- Follow google docstring format ([#2115](https://github.com/embeddings-benchmark/mteb/pull/2115),
  [`276840f`](https://github.com/embeddings-benchmark/mteb/commit/276840f83ccea159563423f4e8cbfe685603ccba))


## v1.34.26 (2025-02-20)

### Bug Fixes

- Update ruff to be gradio compatible (>=0.9.3)
  ([#2111](https://github.com/embeddings-benchmark/mteb/pull/2111),
  [`fb14e0c`](https://github.com/embeddings-benchmark/mteb/commit/fb14e0c652e06d5c730856d9e6ed2769bcd4d223))

- Upgrade ruff to be gradio compatible
  ([#2111](https://github.com/embeddings-benchmark/mteb/pull/2111),
  [`fb14e0c`](https://github.com/embeddings-benchmark/mteb/commit/fb14e0c652e06d5c730856d9e6ed2769bcd4d223))

- Upgrade ruff to latests (same as gradio compatible)
  ([#2111](https://github.com/embeddings-benchmark/mteb/pull/2111),
  [`fb14e0c`](https://github.com/embeddings-benchmark/mteb/commit/fb14e0c652e06d5c730856d9e6ed2769bcd4d223))


## v1.34.25 (2025-02-20)

### Bug Fixes

- Add training data to BGE-m3-custom-fr
  ([#2110](https://github.com/embeddings-benchmark/mteb/pull/2110),
  [`cb42f4a`](https://github.com/embeddings-benchmark/mteb/commit/cb42f4a5cfa0a62fe89b6d8da00b5e27c94cb072))


## v1.34.24 (2025-02-20)

### Bug Fixes

- Add codesage-large-v2 ([#2090](https://github.com/embeddings-benchmark/mteb/pull/2090),
  [`c052bbb`](https://github.com/embeddings-benchmark/mteb/commit/c052bbb98a826ce2da35859178a2868620afa61d))


## v1.34.23 (2025-02-20)

### Bug Fixes

- Add warning about task category conversion
  ([#2108](https://github.com/embeddings-benchmark/mteb/pull/2108),
  [`6a71485`](https://github.com/embeddings-benchmark/mteb/commit/6a714858e75a2a8fd2ec749c0ebe8866fe234037))

### Documentation

- Fix typos & refine text ([#2102](https://github.com/embeddings-benchmark/mteb/pull/2102),
  [`caa0b77`](https://github.com/embeddings-benchmark/mteb/commit/caa0b77e4c21310a64f480cbd710c62810deb134))


## v1.34.22 (2025-02-19)

### Bug Fixes

- Update links ([#2098](https://github.com/embeddings-benchmark/mteb/pull/2098),
  [`6b9f945`](https://github.com/embeddings-benchmark/mteb/commit/6b9f945183ceb01c7bc330fe9cddc132491012fb))

- Updated model annotations for GTE, e5, gritlm, and SFR models
  ([#2101](https://github.com/embeddings-benchmark/mteb/pull/2101),
  [`e0b364b`](https://github.com/embeddings-benchmark/mteb/commit/e0b364b5961e392e7662bfab9bf5ddb460b4943f))


## v1.34.21 (2025-02-18)

### Bug Fixes

- Add back task filtering by modalities
  ([#2080](https://github.com/embeddings-benchmark/mteb/pull/2080),
  [`3deb7ea`](https://github.com/embeddings-benchmark/mteb/commit/3deb7eaf3d57752f625abf26d561b45ab0c47d98))


## v1.34.20 (2025-02-17)

### Bug Fixes

- Fixed previous incorrect specification of splits for CMTEB ( MTEB(cmn, v1) )
  ([#2088](https://github.com/embeddings-benchmark/mteb/pull/2088),
  [`6637ff9`](https://github.com/embeddings-benchmark/mteb/commit/6637ff95945b12a61594d09eafe88c82d3dfe4e4))

- Missing fixes for #2086 - change MultilingualSentiment split from test to validation in CMTEB
  ([#2088](https://github.com/embeddings-benchmark/mteb/pull/2088),
  [`6637ff9`](https://github.com/embeddings-benchmark/mteb/commit/6637ff95945b12a61594d09eafe88c82d3dfe4e4))

- Smarter leaderboard caching with cachetools
  ([#2085](https://github.com/embeddings-benchmark/mteb/pull/2085),
  [`1006770`](https://github.com/embeddings-benchmark/mteb/commit/1006770c098869d5c9db2e1b3b13c2c190c34a26))


## v1.34.19 (2025-02-17)

### Bug Fixes

- Fixed previous incorrect specification of splits for CMTEB ( MTEB(cmn, v1) )
  ([#2086](https://github.com/embeddings-benchmark/mteb/pull/2086),
  [`12d9b96`](https://github.com/embeddings-benchmark/mteb/commit/12d9b96842d64a159dba39013b2a121e7b436f9b))


## v1.34.18 (2025-02-17)

### Bug Fixes

- Freeze model/rank columns in leaderboard
  ([#2044](https://github.com/embeddings-benchmark/mteb/pull/2044),
  [`07562f4`](https://github.com/embeddings-benchmark/mteb/commit/07562f4d27c8319249c2a28ff9903ad5bf3b4173))


## v1.34.17 (2025-02-17)

### Bug Fixes

- Added make command for running leaderboard locally
  ([#2083](https://github.com/embeddings-benchmark/mteb/pull/2083),
  [`b14963f`](https://github.com/embeddings-benchmark/mteb/commit/b14963fe6ace93ca9c1e7e066577f7e44250f823))

- Ensure voyage model uses different naming scheme
  ([#2083](https://github.com/embeddings-benchmark/mteb/pull/2083),
  [`b14963f`](https://github.com/embeddings-benchmark/mteb/commit/b14963fe6ace93ca9c1e7e066577f7e44250f823))

- Ensure voyage models doesn't re-use the name
  ([#2083](https://github.com/embeddings-benchmark/mteb/pull/2083),
  [`b14963f`](https://github.com/embeddings-benchmark/mteb/commit/b14963fe6ace93ca9c1e7e066577f7e44250f823))


## v1.34.16 (2025-02-17)

### Bug Fixes

- Add missing `e5` training datasets
  ([#2065](https://github.com/embeddings-benchmark/mteb/pull/2065),
  [`efe2578`](https://github.com/embeddings-benchmark/mteb/commit/efe2578c06265419d6ea613108d156bb4f124f8f))


## v1.34.15 (2025-02-17)

### Bug Fixes

- Generate metadata ([#2063](https://github.com/embeddings-benchmark/mteb/pull/2063),
  [`26360a0`](https://github.com/embeddings-benchmark/mteb/commit/26360a0b856d97cf8b589218365288bdf55ae791))

- Rerun tests that fail - Networking
  ([#2029](https://github.com/embeddings-benchmark/mteb/pull/2029),
  [`efaa990`](https://github.com/embeddings-benchmark/mteb/commit/efaa990b6c7a4916c3349bda4204b2b323927966))

### Continuous Integration

- Rerun tests that fail due to networking issues.
  ([#2029](https://github.com/embeddings-benchmark/mteb/pull/2029),
  [`efaa990`](https://github.com/embeddings-benchmark/mteb/commit/efaa990b6c7a4916c3349bda4204b2b323927966))


## v1.34.14 (2025-02-14)

### Bug Fixes

- Add climate fever v2 ([#1873](https://github.com/embeddings-benchmark/mteb/pull/1873),
  [`8604e07`](https://github.com/embeddings-benchmark/mteb/commit/8604e079fd5bc6adac2d2050713dcceaeee5932d))

- Add mixbai models ([#1539](https://github.com/embeddings-benchmark/mteb/pull/1539),
  [`76e05dd`](https://github.com/embeddings-benchmark/mteb/commit/76e05ddb0006620eaf0b8850c5fb37bb74b943e1))

- Added gte models ([#1539](https://github.com/embeddings-benchmark/mteb/pull/1539),
  [`76e05dd`](https://github.com/embeddings-benchmark/mteb/commit/76e05ddb0006620eaf0b8850c5fb37bb74b943e1))

- Updating paper scripts ([#1958](https://github.com/embeddings-benchmark/mteb/pull/1958),
  [`c6829d3`](https://github.com/embeddings-benchmark/mteb/commit/c6829d34d7a324bb1f3754d39dd52756921d6a9f))


## v1.34.13 (2025-02-13)

### Bug Fixes

- Update embed_dim for jina models ([#2058](https://github.com/embeddings-benchmark/mteb/pull/2058),
  [`50b8e7b`](https://github.com/embeddings-benchmark/mteb/commit/50b8e7ba10c9a33d2febbc25be8b69893d0b50e6))


## v1.34.12 (2025-02-13)

### Bug Fixes

- Add BRIGHT (long) and fix bug in TaskResult.filter_and_validate()
  ([#2041](https://github.com/embeddings-benchmark/mteb/pull/2041),
  [`3537223`](https://github.com/embeddings-benchmark/mteb/commit/35372238b1f345ecf1422cb967186d8059213d07))

- Add BRIGHT Long ([#2041](https://github.com/embeddings-benchmark/mteb/pull/2041),
  [`3537223`](https://github.com/embeddings-benchmark/mteb/commit/35372238b1f345ecf1422cb967186d8059213d07))

- Add BRIGHT(long) ([#2041](https://github.com/embeddings-benchmark/mteb/pull/2041),
  [`3537223`](https://github.com/embeddings-benchmark/mteb/commit/35372238b1f345ecf1422cb967186d8059213d07))

- Add column descriptions to leaderboard
  ([#2039](https://github.com/embeddings-benchmark/mteb/pull/2039),
  [`01fd6fb`](https://github.com/embeddings-benchmark/mteb/commit/01fd6fbb2a7a2f54543a4b2a41ac96fd90cc61b2))


## v1.34.11 (2025-02-12)

### Bug Fixes

- Add Voyage-code-3 ([#2040](https://github.com/embeddings-benchmark/mteb/pull/2040),
  [`0b37966`](https://github.com/embeddings-benchmark/mteb/commit/0b3796658768cf4c213f83246112d5dbad9ac341))


## v1.34.10 (2025-02-12)

### Bug Fixes

- Add versioning to MTEB benchmarks
  ([#2024](https://github.com/embeddings-benchmark/mteb/pull/2024),
  [`65f3407`](https://github.com/embeddings-benchmark/mteb/commit/65f3407e91100b6176052e2e5137f8e88c4eaece))

### Documentation

- Fix README code rendering ([#2037](https://github.com/embeddings-benchmark/mteb/pull/2037),
  [`1b04130`](https://github.com/embeddings-benchmark/mteb/commit/1b041303bcf56708a0eb9080db6c67099d3fe3f6))


## v1.34.9 (2025-02-12)

### Bug Fixes

- Add SONAR metadata ([#2014](https://github.com/embeddings-benchmark/mteb/pull/2014),
  [`92b74b6`](https://github.com/embeddings-benchmark/mteb/commit/92b74b66e41e747eb6251ca87c21ebe071f1d83e))

- Add SONAR metadata and resolve missing models
  ([#2014](https://github.com/embeddings-benchmark/mteb/pull/2014),
  [`92b74b6`](https://github.com/embeddings-benchmark/mteb/commit/92b74b66e41e747eb6251ca87c21ebe071f1d83e))

- Added script for generating and saving a local leaderboard
  ([#2015](https://github.com/embeddings-benchmark/mteb/pull/2015),
  [`477bea5`](https://github.com/embeddings-benchmark/mteb/commit/477bea5777f39b429e9a2509f5d1344754648ca0))


## v1.34.8 (2025-02-10)

### Bug Fixes

- Add Persian-Specific Models ([#2021](https://github.com/embeddings-benchmark/mteb/pull/2021),
  [`1588b9a`](https://github.com/embeddings-benchmark/mteb/commit/1588b9ac31b7a84d1e34ef0325c0c10b0b0e8db5))

### Documentation

- ModelMeta docstrings Typos ([#2017](https://github.com/embeddings-benchmark/mteb/pull/2017),
  [`42cf6a0`](https://github.com/embeddings-benchmark/mteb/commit/42cf6a02378663ceafda33d5eec28957805bf596))

- Update adding_a_model.md ([#2018](https://github.com/embeddings-benchmark/mteb/pull/2018),
  [`e6e21dc`](https://github.com/embeddings-benchmark/mteb/commit/e6e21dcfa56e0af357546a847367f80b8df173c3))

- Update MTEB eng classic benchmark description
  ([#2006](https://github.com/embeddings-benchmark/mteb/pull/2006),
  [`7917646`](https://github.com/embeddings-benchmark/mteb/commit/7917646c4a4dc3642f08815f18dde9064daffe39))


## v1.34.7 (2025-02-07)

### Bug Fixes

- BEIR-NL metadata mistake ([#2010](https://github.com/embeddings-benchmark/mteb/pull/2010),
  [`b1ac052`](https://github.com/embeddings-benchmark/mteb/commit/b1ac0529cdb41e0052191bf2c8f68c7fb874bc21))


## v1.34.6 (2025-02-07)

### Bug Fixes

- Update faq of on leaderboard ([#2004](https://github.com/embeddings-benchmark/mteb/pull/2004),
  [`4fe4c99`](https://github.com/embeddings-benchmark/mteb/commit/4fe4c998be33aa369af9d13fe71b628d6dc14a84))


## v1.34.5 (2025-02-07)

### Bug Fixes

- Added description and resolved bug in rangeslider
  ([#1993](https://github.com/embeddings-benchmark/mteb/pull/1993),
  [`3887d83`](https://github.com/embeddings-benchmark/mteb/commit/3887d83d89b46539769ebca24c827a663a388091))

- Training data for gritlm ([#1932](https://github.com/embeddings-benchmark/mteb/pull/1932),
  [`d810e4e`](https://github.com/embeddings-benchmark/mteb/commit/d810e4eef72c5bf8e26a91355e240584224a80cb))


## v1.34.4 (2025-02-06)

### Bug Fixes

- Added description and resolved bug in rangeslider
  ([#1990](https://github.com/embeddings-benchmark/mteb/pull/1990),
  [`8583383`](https://github.com/embeddings-benchmark/mteb/commit/85833830bf1f816af06b68339196c665dda1d1b0))


## v1.34.3 (2025-02-06)

### Bug Fixes

- Meta information ru_sentence_models
  ([#1991](https://github.com/embeddings-benchmark/mteb/pull/1991),
  [`370b26c`](https://github.com/embeddings-benchmark/mteb/commit/370b26ccec1a0ab8c1029635b4350d6a316bc1a4))


## v1.34.2 (2025-02-05)

### Bug Fixes

- @mrshu's name in `points.md` ([#1246](https://github.com/embeddings-benchmark/mteb/pull/1246),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add clevr license ([#1356](https://github.com/embeddings-benchmark/mteb/pull/1356),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add Flickr30k T2I
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add implementations of common reranker models
  ([#1309](https://github.com/embeddings-benchmark/mteb/pull/1309),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add listing all available benchmarks CLI option
  ([#1256](https://github.com/embeddings-benchmark/mteb/pull/1256),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add metadata dict to QBQTC in C-MTEB
  ([#1292](https://github.com/embeddings-benchmark/mteb/pull/1292),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add multilingual bench
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add multilingual Benchmark ([#1252](https://github.com/embeddings-benchmark/mteb/pull/1252),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add RepLLaMA style models ([#1223](https://github.com/embeddings-benchmark/mteb/pull/1223),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add Retrieval SK Quad dataset for Slovak search evaluation
  ([#1276](https://github.com/embeddings-benchmark/mteb/pull/1276),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add Slovak Hate Speech and Offensive Language Dataset
  ([#1274](https://github.com/embeddings-benchmark/mteb/pull/1274),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add Touche2020v3 and JMTEB ([#1262](https://github.com/embeddings-benchmark/mteb/pull/1262),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add version check e5-v in mieb ([#1723](https://github.com/embeddings-benchmark/mteb/pull/1723),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add/remove subtasks from BLINKIT2IMultiChoice and BLINKIT2TMultiChoice
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Align BLINK retrieval to multi choice
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Allow benchmark to specify eval_splits
  ([#1217](https://github.com/embeddings-benchmark/mteb/pull/1217),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Allow benchmark to specify eval_splits
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Allow Numpy >=2.0 ([#1264](https://github.com/embeddings-benchmark/mteb/pull/1264),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Allow numpy<2.0.0 ([#1291](https://github.com/embeddings-benchmark/mteb/pull/1291),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Avoid spaces in dataset name for CQADupstack and ignore speed tasks
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- BLIP2 better zero-shot classification without text_proj and vision_proj
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Change comparison to bigger than ([#1743](https://github.com/embeddings-benchmark/mteb/pull/1743),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Default prompt_type to None
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Derive `results_directory` path from `results_repo` name
  ([#1275](https://github.com/embeddings-benchmark/mteb/pull/1275),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Don't hardcode repo name when downloading results
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Downsample large retrieval datasets
  ([#1236](https://github.com/embeddings-benchmark/mteb/pull/1236),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- E4ce987 revision no longer exists for multilingual-e5-small on the Hub
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Ensure STS pearson and spearman does not use the p-value only the correlation
  ([#1207](https://github.com/embeddings-benchmark/mteb/pull/1207),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Ensure that e5 ignores the NQ
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Ensure that MLSUMClusteringP2P.v2 use the fast implementation as was intended
  ([#1112](https://github.com/embeddings-benchmark/mteb/pull/1112),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Ensure that MLSUMClusteringP2P.v2 use the fast implementation as was intended
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Ensure that results are returned even when hitting cache
  ([#1215](https://github.com/embeddings-benchmark/mteb/pull/1215),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Export type for `mteb create_meta`
  ([#1114](https://github.com/embeddings-benchmark/mteb/pull/1114),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Fix bug-causing spelling error in function name of e5-mistral-instruct
  ([#1106](https://github.com/embeddings-benchmark/mteb/pull/1106),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Fix task types in MIEB ([#1956](https://github.com/embeddings-benchmark/mteb/pull/1956),
  [`64c17b6`](https://github.com/embeddings-benchmark/mteb/commit/64c17b6e2ddc8c28f4c08c8ac2e09f5112d71e83))

- Fixed formatting for cli
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- French leaderboard inconsistencies
  ([#1956](https://github.com/embeddings-benchmark/mteb/pull/1956),
  [`64c17b6`](https://github.com/embeddings-benchmark/mteb/commit/64c17b6e2ddc8c28f4c08c8ac2e09f5112d71e83))

- Get meta from CrossEncoder ([#1255](https://github.com/embeddings-benchmark/mteb/pull/1255),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Handling in case not torch tensor
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Keep `prompt_name` in kwargs when model doesn't have a `prompts` attr
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Leaderboard and benchmark.py inconstiencies
  ([#1956](https://github.com/embeddings-benchmark/mteb/pull/1956),
  [`64c17b6`](https://github.com/embeddings-benchmark/mteb/commit/64c17b6e2ddc8c28f4c08c8ac2e09f5112d71e83))

- License metadata in wrong format
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- MultilingualSentimentClassification
  ([#1109](https://github.com/embeddings-benchmark/mteb/pull/1109),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- No longer using same query text for all of BLINKIT2TMultiChoice
  ([#1572](https://github.com/embeddings-benchmark/mteb/pull/1572),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- No longer using same query text for all of BLINKIT2TMultiChoice
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Nomic models using prefix correctly
  ([#1125](https://github.com/embeddings-benchmark/mteb/pull/1125),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Nomic models using prefix correctly
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Normalize benchmarks no only include task objects and added getter for benchmarks
  ([#1208](https://github.com/embeddings-benchmark/mteb/pull/1208),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Normalize licenses including casing, uses of "-" etc.
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Normalize licenses including casing, uses of "-" etc.
  ([#1210](https://github.com/embeddings-benchmark/mteb/pull/1210),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- OpenAI BadRequestError by limiting input dimensions to 2048 elements
  ([#1201](https://github.com/embeddings-benchmark/mteb/pull/1201),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- OpenAI BadRequestError by limiting input dimensions to 2048 elem…
  ([#1203](https://github.com/embeddings-benchmark/mteb/pull/1203),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Re-upload dataset to hub to avoid using script upload
  ([#1322](https://github.com/embeddings-benchmark/mteb/pull/1322),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Remove blink subtask
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Remove duplicate multilingual
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Remove non-existent eval split of CMNLI
  ([#1294](https://github.com/embeddings-benchmark/mteb/pull/1294),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Remove projections from image and text embeddings
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Remove subtask from blink it2i
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Remove wrong negatives from revisiting multichoice datasets
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Select benchmarks CLI option ([#1261](https://github.com/embeddings-benchmark/mteb/pull/1261),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Simplify models implementations ([#1085](https://github.com/embeddings-benchmark/mteb/pull/1085),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Sorting benchmark tasks by MTEB, then alphabetical
  ([#1271](https://github.com/embeddings-benchmark/mteb/pull/1271),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update benchmarks ([#1288](https://github.com/embeddings-benchmark/mteb/pull/1288),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update naming as candidate_labels
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Wrong e5 revisions
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

### Chores

- Remove comment
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

### Continuous Integration

- Removed 3.8 dependency ([#1281](https://github.com/embeddings-benchmark/mteb/pull/1281),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

### Documentation

- Add MTEB(code) dataset ([#1237](https://github.com/embeddings-benchmark/mteb/pull/1237),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Add MTEB(code) dataset
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Added coordination point for Jimmy Lee
  ([#1253](https://github.com/embeddings-benchmark/mteb/pull/1253),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Added coordination point for Jimmy lee for his work on the coordination of Crystina and Nandan
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Clarify adding a model ([#1222](https://github.com/embeddings-benchmark/mteb/pull/1222),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Create benchmarks overview table ([#1245](https://github.com/embeddings-benchmark/mteb/pull/1245),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Fix a link in the README ([#1289](https://github.com/embeddings-benchmark/mteb/pull/1289),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Fix broken links in docs ([#1212](https://github.com/embeddings-benchmark/mteb/pull/1212),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Improve searchability in the advanced usage documentation
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Improve searchability in the advanced usage documentation
  ([#1113](https://github.com/embeddings-benchmark/mteb/pull/1113),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Points for paper writing ([#1286](https://github.com/embeddings-benchmark/mteb/pull/1286),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Rework MIEB docs ([#1802](https://github.com/embeddings-benchmark/mteb/pull/1802),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Small point changes & more contributors
  ([#1254](https://github.com/embeddings-benchmark/mteb/pull/1254),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Specify how to use prompts with Sentence Transformers
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update affiliation ([#1247](https://github.com/embeddings-benchmark/mteb/pull/1247),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update affiliation ([#1248](https://github.com/embeddings-benchmark/mteb/pull/1248),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update based on corrections
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update mteb(eng) calculation ([#1258](https://github.com/embeddings-benchmark/mteb/pull/1258),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update points ([#1228](https://github.com/embeddings-benchmark/mteb/pull/1228),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

### Features

- Add blip2 models, still mismatched names
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Leverage SentenceTransformer models' query/passage specific prompts
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Leverage SentenceTransformers' query/passage specific prompts
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Merge main into MIEB ([#1329](https://github.com/embeddings-benchmark/mteb/pull/1329),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Readd arctic models due to metadata
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Update metadata for all models ([#1316](https://github.com/embeddings-benchmark/mteb/pull/1316),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Use Enum for `prompt_type`
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

- Use prompts instead of encode_corpus and encode_queries
  ([#1278](https://github.com/embeddings-benchmark/mteb/pull/1278),
  [`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))

### Refactoring

- Remove E5Wrapper
  ([`bc05a9d`](https://github.com/embeddings-benchmark/mteb/commit/bc05a9d39b60210fb676518faeab055aa1b5448d))


## v1.34.1 (2025-02-05)

### Bug Fixes

- Add instruction for running leaderboard
  ([#1925](https://github.com/embeddings-benchmark/mteb/pull/1925),
  [`d87d17e`](https://github.com/embeddings-benchmark/mteb/commit/d87d17e690b2d3e072e854d671965f7324eb3fe7))

- Changed callback for slider, accounted for None input
  ([#1969](https://github.com/embeddings-benchmark/mteb/pull/1969),
  [`4232427`](https://github.com/embeddings-benchmark/mteb/commit/4232427c61c3f0db474666134f00bbfe163a981c))

### Documentation

- Add instruction for running leaderboard
  ([#1925](https://github.com/embeddings-benchmark/mteb/pull/1925),
  [`d87d17e`](https://github.com/embeddings-benchmark/mteb/commit/d87d17e690b2d3e072e854d671965f7324eb3fe7))


## v1.34.0 (2025-02-04)

### Features

- Add new benchmark BEIR-NL ([#1909](https://github.com/embeddings-benchmark/mteb/pull/1909),
  [`de8f384`](https://github.com/embeddings-benchmark/mteb/commit/de8f384e11fdde2b960a9fda6ed574ca496bfdd5))


## v1.33.1 (2025-02-04)

### Bug Fixes

- Fix task types in MIEB ([#1952](https://github.com/embeddings-benchmark/mteb/pull/1952),
  [`a21f0b7`](https://github.com/embeddings-benchmark/mteb/commit/a21f0b74df99c23e3fec72cd5d577710e2391a60))

- Fix task types in MIEB
  ([`f43b661`](https://github.com/embeddings-benchmark/mteb/commit/f43b661c41a3e1bbe0628af9cf38382b4ca21fa4))


## v1.33.0 (2025-02-04)

### Features

- Merge MIEB into main 🎉 ([#1944](https://github.com/embeddings-benchmark/mteb/pull/1944),
  [`6d63d06`](https://github.com/embeddings-benchmark/mteb/commit/6d63d0668552946bad5924092aa5864cfe442c93))


## v1.32.0 (2025-02-04)

### Features

- Add beir ([#1933](https://github.com/embeddings-benchmark/mteb/pull/1933),
  [`7ef3a90`](https://github.com/embeddings-benchmark/mteb/commit/7ef3a906245c460a0e98c0a85e4861312e3742ed))


## v1.31.8 (2025-02-01)

### Bug Fixes

- Add datasets in CodeRAG-Bench ([#1595](https://github.com/embeddings-benchmark/mteb/pull/1595),
  [`9c762da`](https://github.com/embeddings-benchmark/mteb/commit/9c762da0332009375dc4d5a42aa770bd68d309a4))

- Updated citation for mteb(scandinavian)
  ([#1914](https://github.com/embeddings-benchmark/mteb/pull/1914),
  [`f3526fc`](https://github.com/embeddings-benchmark/mteb/commit/f3526fc0b83cfb25989ec9ad405995bcad19b35d))

### Documentation

- Updated citation for mteb(scandinavian)
  ([#1914](https://github.com/embeddings-benchmark/mteb/pull/1914),
  [`f3526fc`](https://github.com/embeddings-benchmark/mteb/commit/f3526fc0b83cfb25989ec9ad405995bcad19b35d))


## v1.31.7 (2025-02-01)

### Bug Fixes

- Remove SummaryRetrieval as a type
  ([#1915](https://github.com/embeddings-benchmark/mteb/pull/1915),
  [`21d32f0`](https://github.com/embeddings-benchmark/mteb/commit/21d32f0b96135fc8f95ce6fd7e513109274a806b))

- Revert rename and add to description
  ([#1918](https://github.com/embeddings-benchmark/mteb/pull/1918),
  [`75ff333`](https://github.com/embeddings-benchmark/mteb/commit/75ff333d60f1e93dcb645dbccbdc868dc5bb9420))

### Documentation

- Add sort to domains for task metadata
  ([#1922](https://github.com/embeddings-benchmark/mteb/pull/1922),
  [`6f673ba`](https://github.com/embeddings-benchmark/mteb/commit/6f673ba0350a73c3b0bd39a22c704b36640ef1ff))


## v1.31.6 (2025-01-30)

### Bug Fixes

- Added CQADupstack annotations ([#1895](https://github.com/embeddings-benchmark/mteb/pull/1895),
  [`938e90f`](https://github.com/embeddings-benchmark/mteb/commit/938e90f58714c525157d968a278ae3b07fc7b20a))

- Filling missing metadata for leaderboard release
  ([#1895](https://github.com/embeddings-benchmark/mteb/pull/1895),
  [`938e90f`](https://github.com/embeddings-benchmark/mteb/commit/938e90f58714c525157d968a278ae3b07fc7b20a))


## v1.31.5 (2025-01-29)

### Bug Fixes

- Limited plotly version to be less than 6.0.0
  ([#1902](https://github.com/embeddings-benchmark/mteb/pull/1902),
  [`cec0ed4`](https://github.com/embeddings-benchmark/mteb/commit/cec0ed472fc762722bce24dfde6cb331f7006dee))


## v1.31.4 (2025-01-29)

### Bug Fixes

- Allow aggregated tasks within benchmarks
  ([#1771](https://github.com/embeddings-benchmark/mteb/pull/1771),
  [`8fb59a4`](https://github.com/embeddings-benchmark/mteb/commit/8fb59a49b00e7932abec42c045c1cc068c7eba41))

- Simplify in a few areas ([#1771](https://github.com/embeddings-benchmark/mteb/pull/1771),
  [`8fb59a4`](https://github.com/embeddings-benchmark/mteb/commit/8fb59a49b00e7932abec42c045c1cc068c7eba41))

### Features

- Update task filtering, fixing bug on MTEB
  ([#1771](https://github.com/embeddings-benchmark/mteb/pull/1771),
  [`8fb59a4`](https://github.com/embeddings-benchmark/mteb/commit/8fb59a49b00e7932abec42c045c1cc068c7eba41))


## v1.31.3 (2025-01-28)

### Bug Fixes

- External results are preferred when only they have the needed splits
  ([#1893](https://github.com/embeddings-benchmark/mteb/pull/1893),
  [`2a41730`](https://github.com/embeddings-benchmark/mteb/commit/2a4173046a4b64c38c99132417abe60590dc0381))


## v1.31.2 (2025-01-28)

### Bug Fixes

- Update voyage exp metadata ([#1888](https://github.com/embeddings-benchmark/mteb/pull/1888),
  [`e623771`](https://github.com/embeddings-benchmark/mteb/commit/e6237714a1e340a0e07a8f121030e0277a8d5634))


## v1.31.1 (2025-01-26)

### Bug Fixes

- Fix jina v1, 2 models ([#1872](https://github.com/embeddings-benchmark/mteb/pull/1872),
  [`1d66089`](https://github.com/embeddings-benchmark/mteb/commit/1d660892288d02379e67a59b94523410497ee20b))


## v1.31.0 (2025-01-25)

### Features

- Add instruct wrapper ([#1768](https://github.com/embeddings-benchmark/mteb/pull/1768),
  [`ee0f15a`](https://github.com/embeddings-benchmark/mteb/commit/ee0f15ad03313d3a030c6f21ae6aafd9bc95bbb0))


## v1.30.0 (2025-01-25)

### Features

- Integrating ChemTEB ([#1708](https://github.com/embeddings-benchmark/mteb/pull/1708),
  [`4d66434`](https://github.com/embeddings-benchmark/mteb/commit/4d66434c80050ace3b927f3fc1829b8dd377f78a))


## v1.29.16 (2025-01-22)

### Bug Fixes

- Added correct training data annotation to LENS
  ([#1859](https://github.com/embeddings-benchmark/mteb/pull/1859),
  [`e775436`](https://github.com/embeddings-benchmark/mteb/commit/e77543694ae16716c4420dd0b79c0d9f33a938db))


## v1.29.15 (2025-01-22)

### Bug Fixes

- Adding missing model meta ([#1856](https://github.com/embeddings-benchmark/mteb/pull/1856),
  [`692bd26`](https://github.com/embeddings-benchmark/mteb/commit/692bd265e731c934d8318c497b954e271540a6ab))


## v1.29.14 (2025-01-22)

### Bug Fixes

- Fix zeta alpha mistral ([#1736](https://github.com/embeddings-benchmark/mteb/pull/1736),
  [`4985da9`](https://github.com/embeddings-benchmark/mteb/commit/4985da94cbc4c1368debab737fa8195f6bb91ce2))

- Hotfixed public_training_data type annotation
  ([#1857](https://github.com/embeddings-benchmark/mteb/pull/1857),
  [`4bd7328`](https://github.com/embeddings-benchmark/mteb/commit/4bd7328f1d43ff36564eb5941e7b32daf826f456))


## v1.29.13 (2025-01-22)

### Bug Fixes

- Fixed leaderboard search bar ([#1852](https://github.com/embeddings-benchmark/mteb/pull/1852),
  [`fe33061`](https://github.com/embeddings-benchmark/mteb/commit/fe330611b6e433096501d0d9814b2c644c33e984))


## v1.29.12 (2025-01-21)

### Bug Fixes

- Leaderboard Refinements ([#1849](https://github.com/embeddings-benchmark/mteb/pull/1849),
  [`a8cc887`](https://github.com/embeddings-benchmark/mteb/commit/a8cc88778623ee4e46c7c27ea5b5bc98e534165e))


## v1.29.11 (2025-01-21)

### Bug Fixes

- Add additional dataset annotations
  ([#1846](https://github.com/embeddings-benchmark/mteb/pull/1846),
  [`a7a8144`](https://github.com/embeddings-benchmark/mteb/commit/a7a8144a6964641614c7d407e43c75ab5b7c40ca))

- Add reported annotation and re-added public_training_data
  ([#1846](https://github.com/embeddings-benchmark/mteb/pull/1846),
  [`a7a8144`](https://github.com/embeddings-benchmark/mteb/commit/a7a8144a6964641614c7d407e43c75ab5b7c40ca))

- Readded public training data ([#1846](https://github.com/embeddings-benchmark/mteb/pull/1846),
  [`a7a8144`](https://github.com/embeddings-benchmark/mteb/commit/a7a8144a6964641614c7d407e43c75ab5b7c40ca))


## v1.29.10 (2025-01-20)

### Bug Fixes

- Leaderboard: `K` instead of `M` ([#1794](https://github.com/embeddings-benchmark/mteb/pull/1794),
  [`0a83e38`](https://github.com/embeddings-benchmark/mteb/commit/0a83e383efe41e86e51c0d4cdca18d9ed5d42821))

- Remove default params, `public_training_data` and `memory usage` in `ModelMeta`
  ([#1794](https://github.com/embeddings-benchmark/mteb/pull/1794),
  [`0a83e38`](https://github.com/embeddings-benchmark/mteb/commit/0a83e383efe41e86e51c0d4cdca18d9ed5d42821))

- Subsets to run ([#1830](https://github.com/embeddings-benchmark/mteb/pull/1830),
  [`8be6b2e`](https://github.com/embeddings-benchmark/mteb/commit/8be6b2e36abb005822e07c034484c245345f6eb2))


## v1.29.9 (2025-01-17)

### Bug Fixes

- Fixed eval split for MultilingualSentiment in C-MTEB
  ([#1804](https://github.com/embeddings-benchmark/mteb/pull/1804),
  [`96f639b`](https://github.com/embeddings-benchmark/mteb/commit/96f639bc34153caaac422a3a13e0d9f3626d65b9))


## v1.29.8 (2025-01-17)

### Bug Fixes

- Add even more training dataset annotations
  ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Add gritlm ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Added C-MTEB ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Added Chinese Stella models ([#1824](https://github.com/embeddings-benchmark/mteb/pull/1824),
  [`74b495c`](https://github.com/embeddings-benchmark/mteb/commit/74b495cd197846af91d6425891d1f9156cd1db68))

- Added Misc Chinese models ([#1819](https://github.com/embeddings-benchmark/mteb/pull/1819),
  [`9823529`](https://github.com/embeddings-benchmark/mteb/commit/9823529282b131e7f24399eb0639fbc33280d148))

- Added more annotations! ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Added way more training dataset annotations
  ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Allow to load no revision available
  ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Bm25s ([#1827](https://github.com/embeddings-benchmark/mteb/pull/1827),
  [`96420a2`](https://github.com/embeddings-benchmark/mteb/commit/96420a2ad39a61aafb34630f5c6c5a50a3717fdc))

- Leaderboard: `K` instead of `M` ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Loading pre 11 ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Simplify in a few areas ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Update max tokens for OpenAI ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

### Continuous Integration

- Fix model loading test ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Skip AfriSentiLID for now ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

### Documentation

- Add contact to MMTEB benchmarks ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

### Features

- Update task filtering, fixing bug on MTEB
  ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))

- Update task filtering, fixing bug which included cross-lingual tasks in overly many benchmarks
  ([#1765](https://github.com/embeddings-benchmark/mteb/pull/1765),
  [`3b2d074`](https://github.com/embeddings-benchmark/mteb/commit/3b2d074efbe9d665171071dab63796f3ae783802))


## v1.29.7 (2025-01-16)

### Bug Fixes

- Add bge-m3 `ModelMeta` ([#1821](https://github.com/embeddings-benchmark/mteb/pull/1821),
  [`4ac59bc`](https://github.com/embeddings-benchmark/mteb/commit/4ac59bcdfbed8604b05e067b8b7df79f47b0d7a7))

### Continuous Integration

- Only return 1 model_name per file
  ([#1818](https://github.com/embeddings-benchmark/mteb/pull/1818),
  [`d7a7791`](https://github.com/embeddings-benchmark/mteb/commit/d7a77918cc0e8b8f03cbbe5199e8a0fe58e429d9))

### Features

- **models**: Add infly/inf-retriever-v1 model metadata- Add inf_models.py file with metadata for
  infly/inf-retriever-v1 model ([#1744](https://github.com/embeddings-benchmark/mteb/pull/1744),
  [`60c4980`](https://github.com/embeddings-benchmark/mteb/commit/60c49804fe0bbf10a4dde7cc63a9002f5eee6d40))


## v1.29.6 (2025-01-15)

### Bug Fixes

- Added more Chinese models' `ModelMeta`
  ([#1814](https://github.com/embeddings-benchmark/mteb/pull/1814),
  [`748955c`](https://github.com/embeddings-benchmark/mteb/commit/748955c367b5c549f4b8d54945361f5bbc7184f6))


## v1.29.5 (2025-01-15)

### Bug Fixes

- Add additional contacts ([#1817](https://github.com/embeddings-benchmark/mteb/pull/1817),
  [`c4ee9fe`](https://github.com/embeddings-benchmark/mteb/commit/c4ee9fe1ccffaea57b8bf21d42e4031386a95c01))


## v1.29.4 (2025-01-15)

### Bug Fixes

- Added `ModelMeta` for BGE, GTE Chinese and multilingual models
  ([#1811](https://github.com/embeddings-benchmark/mteb/pull/1811),
  [`3f5ee82`](https://github.com/embeddings-benchmark/mteb/commit/3f5ee82a5049eaf235a84fcfc9278f48adecfcb7))

- Zero shot and aggregation on Leaderboard
  ([#1810](https://github.com/embeddings-benchmark/mteb/pull/1810),
  [`0acc166`](https://github.com/embeddings-benchmark/mteb/commit/0acc166d54294ce16dc4750a84ad4abd896ab92d))


## v1.29.3 (2025-01-14)

### Bug Fixes

- Allow to load no revision available
  ([#1801](https://github.com/embeddings-benchmark/mteb/pull/1801),
  [`a202884`](https://github.com/embeddings-benchmark/mteb/commit/a2028840a6b4f77057761664edce8cae2edb64d1))


## v1.29.2 (2025-01-14)

### Bug Fixes

- Loading pre 11 ([#1798](https://github.com/embeddings-benchmark/mteb/pull/1798),
  [`94103e6`](https://github.com/embeddings-benchmark/mteb/commit/94103e6a2e8156678c3858045286cbd50b5d49c5))

### Documentation

- Add contact to MMTEB benchmarks ([#1796](https://github.com/embeddings-benchmark/mteb/pull/1796),
  [`e9e9118`](https://github.com/embeddings-benchmark/mteb/commit/e9e9118b9bf6cbda678c70d6776a8f290833eff3))


## v1.29.1 (2025-01-13)

### Bug Fixes

- Added C-MTEB ([#1786](https://github.com/embeddings-benchmark/mteb/pull/1786),
  [`3ba7e22`](https://github.com/embeddings-benchmark/mteb/commit/3ba7e22d52320166ec003cbd04c5f09bc0eefe24))


## v1.29.0 (2025-01-13)

### Bug Fixes

- Simplify in a few areas ([#1787](https://github.com/embeddings-benchmark/mteb/pull/1787),
  [`4a70e5d`](https://github.com/embeddings-benchmark/mteb/commit/4a70e5d8996a341097c81782b463b1822f9708fe))

### Continuous Integration

- Fix model loading test ([#1775](https://github.com/embeddings-benchmark/mteb/pull/1775),
  [`9b117a8`](https://github.com/embeddings-benchmark/mteb/commit/9b117a8245a8c56470d99b8ca3d6b2f6b6819dd8))

### Features

- Update task filtering, fixing bug on MTEB
  ([#1787](https://github.com/embeddings-benchmark/mteb/pull/1787),
  [`4a70e5d`](https://github.com/embeddings-benchmark/mteb/commit/4a70e5d8996a341097c81782b463b1822f9708fe))

- Update task filtering, fixing bug which included cross-lingual tasks in overly many benchmarks
  ([#1787](https://github.com/embeddings-benchmark/mteb/pull/1787),
  [`4a70e5d`](https://github.com/embeddings-benchmark/mteb/commit/4a70e5d8996a341097c81782b463b1822f9708fe))


## v1.28.7 (2025-01-13)

### Bug Fixes

- Update max tokens for OpenAI ([#1772](https://github.com/embeddings-benchmark/mteb/pull/1772),
  [`0c5c3a5`](https://github.com/embeddings-benchmark/mteb/commit/0c5c3a544bea7dcb4c6e6d75d612638171cf0332))

### Continuous Integration

- Skip AfriSentiLID for now ([#1785](https://github.com/embeddings-benchmark/mteb/pull/1785),
  [`71dbd61`](https://github.com/embeddings-benchmark/mteb/commit/71dbd61c2b1b82e3d19ed0a4914f59886d4f0007))


## v1.28.6 (2025-01-11)

### Bug Fixes

- Added annotations for arctic embed models
  ([#1742](https://github.com/embeddings-benchmark/mteb/pull/1742),
  [`3f093c8`](https://github.com/embeddings-benchmark/mteb/commit/3f093c86a5e4bccd31e8a9ed860d1a33bd64b391))

- Added annotations for training data
  ([#1742](https://github.com/embeddings-benchmark/mteb/pull/1742),
  [`3f093c8`](https://github.com/embeddings-benchmark/mteb/commit/3f093c86a5e4bccd31e8a9ed860d1a33bd64b391))


## v1.28.5 (2025-01-11)

### Bug Fixes

- Leaderboard: `K` instead of `M` ([#1761](https://github.com/embeddings-benchmark/mteb/pull/1761),
  [`972463e`](https://github.com/embeddings-benchmark/mteb/commit/972463e818b411609c4c60c070377e75c2987b4c))


## v1.28.4 (2025-01-10)

### Bug Fixes

- Add similarity to SentenceTransformerWrapper
  ([#1748](https://github.com/embeddings-benchmark/mteb/pull/1748),
  [`3fe9264`](https://github.com/embeddings-benchmark/mteb/commit/3fe92644fa53c0c8cedc92d17fb25f0012a26aab))

- Fixes implementation of similarity()
  ([#1748](https://github.com/embeddings-benchmark/mteb/pull/1748),
  [`3fe9264`](https://github.com/embeddings-benchmark/mteb/commit/3fe92644fa53c0c8cedc92d17fb25f0012a26aab))

- **#1594**: Fixes implementation of similarity()
  ([#1748](https://github.com/embeddings-benchmark/mteb/pull/1748),
  [`3fe9264`](https://github.com/embeddings-benchmark/mteb/commit/3fe92644fa53c0c8cedc92d17fb25f0012a26aab))


## v1.28.3 (2025-01-10)

### Bug Fixes

- Fixed definition of zero-shot in ModelMeta
  ([#1747](https://github.com/embeddings-benchmark/mteb/pull/1747),
  [`407e205`](https://github.com/embeddings-benchmark/mteb/commit/407e20541613018b61fda2f1ec6be0ef9741e194))


## v1.28.2 (2025-01-10)

### Bug Fixes

- Fixed task_type aggregation on leaderboard
  ([#1746](https://github.com/embeddings-benchmark/mteb/pull/1746),
  [`76bb070`](https://github.com/embeddings-benchmark/mteb/commit/76bb070f5966716010e399c6bf1c2278ce83a173))


## v1.28.1 (2025-01-10)

### Bug Fixes

- Leaderboard Speedup ([#1745](https://github.com/embeddings-benchmark/mteb/pull/1745),
  [`9eff8ca`](https://github.com/embeddings-benchmark/mteb/commit/9eff8cae60e3fcf9346969dfbf7f548f3e27bc55))

### Testing

- Add script to test model loading below n_parameters threshold
  ([#1698](https://github.com/embeddings-benchmark/mteb/pull/1698),
  [`8d033f3`](https://github.com/embeddings-benchmark/mteb/commit/8d033f39415cd00840a1e0f6305d453ca6032abf))


## v1.28.0 (2025-01-09)

### Bug Fixes

- Allow kwargs in init for RerankingWrapper
  ([#1676](https://github.com/embeddings-benchmark/mteb/pull/1676),
  [`f5962c6`](https://github.com/embeddings-benchmark/mteb/commit/f5962c6ff0197f38a39a06c1a14ad6d2bb7522f3))

### Features

- Add nomic modern bert ([#1684](https://github.com/embeddings-benchmark/mteb/pull/1684),
  [`95f143a`](https://github.com/embeddings-benchmark/mteb/commit/95f143a075812e4c12d53c0893539db0379052d9))


## v1.27.0 (2025-01-08)

### Features

- Reduce logging for load_results()
  ([`7e16fa2`](https://github.com/embeddings-benchmark/mteb/commit/7e16fa2565b2058e12303a1feedbd0d4dea96a41))


## v1.26.6 (2025-01-08)

### Bug Fixes

- Added zero shot tag to benchmark ([#1710](https://github.com/embeddings-benchmark/mteb/pull/1710),
  [`8702815`](https://github.com/embeddings-benchmark/mteb/commit/87028155f1f4d0b81156e9ff278dcab8a903e7a4))


## v1.26.5 (2025-01-08)

### Bug Fixes

- Rollback BUCC revision ([#1706](https://github.com/embeddings-benchmark/mteb/pull/1706),
  [`9bcb52f`](https://github.com/embeddings-benchmark/mteb/commit/9bcb52ff6c6a13d3de03dc2f0cc95fd3a62d9170))

### Continuous Integration

- Refresh the v2 leaderboard daily ([#1711](https://github.com/embeddings-benchmark/mteb/pull/1711),
  [`25f4f61`](https://github.com/embeddings-benchmark/mteb/commit/25f4f618f1694d1155919c9771c551fa70b5049b))


## v1.26.4 (2025-01-04)

### Bug Fixes

- GermanDPR ([#1703](https://github.com/embeddings-benchmark/mteb/pull/1703),
  [`753d08a`](https://github.com/embeddings-benchmark/mteb/commit/753d08a95dfdcdb3439510abff69e14304caa4e7))

- Register MicroLlama Text Embedding
  ([#1644](https://github.com/embeddings-benchmark/mteb/pull/1644),
  [`6d1d9f4`](https://github.com/embeddings-benchmark/mteb/commit/6d1d9f4191876829740b9bf7234ec62d31805d30))


## v1.26.3 (2025-01-03)

### Bug Fixes

- NanoBeir ([#1687](https://github.com/embeddings-benchmark/mteb/pull/1687),
  [`cff7ed8`](https://github.com/embeddings-benchmark/mteb/commit/cff7ed887715f3f72c4de0793041c15205f87f04))

- Nomic prompts ([#1685](https://github.com/embeddings-benchmark/mteb/pull/1685),
  [`808257c`](https://github.com/embeddings-benchmark/mteb/commit/808257c0311cf5ff54cf579412a43eb460fbc15b))


## v1.26.2 (2025-01-03)

### Bug Fixes

- Update model loader to trust remote code
  ([#1697](https://github.com/embeddings-benchmark/mteb/pull/1697),
  [`43d74e1`](https://github.com/embeddings-benchmark/mteb/commit/43d74e158b6d3217e94e074b96ebe0649651ddf8))


## v1.26.1 (2025-01-03)

### Bug Fixes

- Add revision for jinaai/jina-embeddings-v2-small-en
  ([#1692](https://github.com/embeddings-benchmark/mteb/pull/1692),
  [`6bfc1f2`](https://github.com/embeddings-benchmark/mteb/commit/6bfc1f2a5655c66940e4cb515153bb910498b9b6))

- Add trust_remote_code to Snowflake/snowflake-arctic-embed-m-long
  ([#1695](https://github.com/embeddings-benchmark/mteb/pull/1695),
  [`f4de307`](https://github.com/embeddings-benchmark/mteb/commit/f4de30737c6d862973a4512a33e5f1d269eea5af))


## v1.26.0 (2025-01-02)

### Bug Fixes

- Arg name for openbmb/MiniCPM-Embedding
  ([#1691](https://github.com/embeddings-benchmark/mteb/pull/1691),
  [`4a496b9`](https://github.com/embeddings-benchmark/mteb/commit/4a496b9f1d179f30c39c2b4acc55752816e21048))

### Features

- Add `avsolatorio/NoInstruct-small-Embedding-v0`
  ([#1677](https://github.com/embeddings-benchmark/mteb/pull/1677),
  [`ba1f022`](https://github.com/embeddings-benchmark/mteb/commit/ba1f022d5063ce8f4c5fec095b7cc5218372666a))


## v1.25.20 (2025-01-02)

### Bug Fixes

- Nomic tensor return ([#1683](https://github.com/embeddings-benchmark/mteb/pull/1683),
  [`f5e6401`](https://github.com/embeddings-benchmark/mteb/commit/f5e64013e83033f9b2f60bf6f44bf373ab684ad9))


## v1.25.19 (2025-01-02)

### Bug Fixes

- Trust remote code for snowflake-arctic-embed-m-v2.0
  ([#1682](https://github.com/embeddings-benchmark/mteb/pull/1682),
  [`7b1e67b`](https://github.com/embeddings-benchmark/mteb/commit/7b1e67bf6ce6ad880d7f34250778db12a3675ee2))


## v1.25.18 (2025-01-02)

### Bug Fixes

- Add check for key error in loader
  ([#1675](https://github.com/embeddings-benchmark/mteb/pull/1675),
  [`1aa08fd`](https://github.com/embeddings-benchmark/mteb/commit/1aa08fd41bce7a4372bf8aacde1db9e38a363719))


## v1.25.17 (2025-01-01)

### Bug Fixes

- Add warning for non-retrieval tasks when using bm25s
  ([#1678](https://github.com/embeddings-benchmark/mteb/pull/1678),
  [`c50f26c`](https://github.com/embeddings-benchmark/mteb/commit/c50f26c84dab23a065b3044fe614ceb282b7f792))


## v1.25.16 (2025-01-01)

### Bug Fixes

- Update BUCC dataset revision ([#1674](https://github.com/embeddings-benchmark/mteb/pull/1674),
  [`343edc4`](https://github.com/embeddings-benchmark/mteb/commit/343edc485a366a91925b893033abe76b2a300dcc))


## v1.25.15 (2025-01-01)

### Bug Fixes

- Remove model as a parameter for MulticlassClassification
  ([#1666](https://github.com/embeddings-benchmark/mteb/pull/1666),
  [`5cfcc77`](https://github.com/embeddings-benchmark/mteb/commit/5cfcc77cc8eb45fd7ba15042117988843ed71ddc))

- Use prompts instead of prompt names for voyage
  ([#1665](https://github.com/embeddings-benchmark/mteb/pull/1665),
  [`82e9949`](https://github.com/embeddings-benchmark/mteb/commit/82e9949828b83e30c3efb7ccd3268cf0762dc591))


## v1.25.14 (2025-01-01)

### Bug Fixes

- Pass trust_remote_code=True to CPM model
  ([#1670](https://github.com/embeddings-benchmark/mteb/pull/1670),
  [`f99a178`](https://github.com/embeddings-benchmark/mteb/commit/f99a178c2c085644d0adf64d37d162bc748d5be7))

- Updated metadata for CPM ([#1670](https://github.com/embeddings-benchmark/mteb/pull/1670),
  [`f99a178`](https://github.com/embeddings-benchmark/mteb/commit/f99a178c2c085644d0adf64d37d162bc748d5be7))


## v1.25.13 (2025-01-01)

### Bug Fixes

- Pass trust_remote_code=True to CPM model
  ([#1669](https://github.com/embeddings-benchmark/mteb/pull/1669),
  [`f426159`](https://github.com/embeddings-benchmark/mteb/commit/f4261592066454f1b9b44f6bce59b0c1ea2e9ac7))


## v1.25.12 (2025-01-01)

### Bug Fixes

- Use batch size kwargs for openai APIs
  ([#1668](https://github.com/embeddings-benchmark/mteb/pull/1668),
  [`663653e`](https://github.com/embeddings-benchmark/mteb/commit/663653e1be56e885ef2b2b1ab0eb1c4ac01a1b52))


## v1.25.11 (2025-01-01)

### Bug Fixes

- Update gritlm kwargs ([#1643](https://github.com/embeddings-benchmark/mteb/pull/1643),
  [`19cbf64`](https://github.com/embeddings-benchmark/mteb/commit/19cbf64d727e4980d80340fde55d3d7491924318))


## v1.25.10 (2025-01-01)

### Bug Fixes

- Cast all Model2Vec outputs as floats
  ([#1667](https://github.com/embeddings-benchmark/mteb/pull/1667),
  [`fa0ed6b`](https://github.com/embeddings-benchmark/mteb/commit/fa0ed6b37c232f1832630eb15e96f2de6bf90a57))


## v1.25.9 (2024-12-30)

### Bug Fixes

- Add missing benchmark to benchmarks.py
  ([#1641](https://github.com/embeddings-benchmark/mteb/pull/1641),
  [`27eb549`](https://github.com/embeddings-benchmark/mteb/commit/27eb549d8445e1425dd0fec8cfff80892d20ba4e))


## v1.25.8 (2024-12-30)

### Bug Fixes

- Output_folder for co2 evaluation ([#1642](https://github.com/embeddings-benchmark/mteb/pull/1642),
  [`366b2ce`](https://github.com/embeddings-benchmark/mteb/commit/366b2cef74b0ccc5486799696a3d939cdf64c9c1))


## v1.25.7 (2024-12-29)

### Bug Fixes

- Correction of discrepancies for gte-Qweb model
  ([#1637](https://github.com/embeddings-benchmark/mteb/pull/1637),
  [`2de61b1`](https://github.com/embeddings-benchmark/mteb/commit/2de61b104ce0433955815b977aba2dca4213e775))


## v1.25.6 (2024-12-24)

### Bug Fixes

- Update results_to_dataframe to use BenchmarkResults class
  ([#1628](https://github.com/embeddings-benchmark/mteb/pull/1628),
  [`02ae4fa`](https://github.com/embeddings-benchmark/mteb/commit/02ae4fabc3c0a7d733728c2c645c9e472fca42bb))


## v1.25.5 (2024-12-22)

### Bug Fixes

- GermanDPR Dataset Causes Cross-Encoder Failure Due to Unexpected dict
  ([#1621](https://github.com/embeddings-benchmark/mteb/pull/1621),
  [`748033e`](https://github.com/embeddings-benchmark/mteb/commit/748033ebd094896b2d40a47b0e88ae13bb7fbef8))

- Properly add mteb_model_meta to model object
  ([#1623](https://github.com/embeddings-benchmark/mteb/pull/1623),
  [`72a457e`](https://github.com/embeddings-benchmark/mteb/commit/72a457ebbc6ae7e758adf3720ba082606084b84d))


## v1.25.4 (2024-12-22)

### Bug Fixes

- Override existing results ([#1617](https://github.com/embeddings-benchmark/mteb/pull/1617),
  [`272adb1`](https://github.com/embeddings-benchmark/mteb/commit/272adb1691fb6a2f7863a16b2d32f0af4946c37f))


## v1.25.3 (2024-12-20)

### Bug Fixes

- Set `use_instructions` to True in models using prompts
  ([#1616](https://github.com/embeddings-benchmark/mteb/pull/1616),
  [`0c44482`](https://github.com/embeddings-benchmark/mteb/commit/0c444827f97bc558cbe5573714b3f4d9e7d745c0))

### Features

- Set `use_instructions` to True in models using prompts
  ([#1616](https://github.com/embeddings-benchmark/mteb/pull/1616),
  [`0c44482`](https://github.com/embeddings-benchmark/mteb/commit/0c444827f97bc558cbe5573714b3f4d9e7d745c0))


## v1.25.2 (2024-12-20)

### Bug Fixes

- Disable co2_tracker for API models
  ([#1614](https://github.com/embeddings-benchmark/mteb/pull/1614),
  [`7c8e094`](https://github.com/embeddings-benchmark/mteb/commit/7c8e094743c236a46d892f7cfa59529d64ef141b))


## v1.25.1 (2024-12-16)

### Bug Fixes

- Leaderboard refinements ([#1603](https://github.com/embeddings-benchmark/mteb/pull/1603),
  [`6ecc86f`](https://github.com/embeddings-benchmark/mteb/commit/6ecc86ff2f6fc0ea83332cb9a454df8c7e178ddd))


## v1.25.0 (2024-12-14)

### Bug Fixes

- Max_sim add pad_sequence ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

- Pass is_query to pylate ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

- Resolve issues ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

### Documentation

- Add doc for Model2VecWrapper.__init__(...)
  ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

### Features

- Add ColBert ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

- Add ColBERTWrapper to models & add ColBERTv2
  ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

- Add max_sim operator for IR tasks to support multi-vector models
  ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

- Add revision & prompt_name ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))

- Integrate Jinja templates for ColBERTv2 and add model prompt handling
  ([#1563](https://github.com/embeddings-benchmark/mteb/pull/1563),
  [`fdfdaef`](https://github.com/embeddings-benchmark/mteb/commit/fdfdaeff8597707a70b79e1ff0b0cb5b63a97b01))


## v1.24.2 (2024-12-13)

### Bug Fixes

- Eval langs not correctly passed to monolingual tasks
  ([#1587](https://github.com/embeddings-benchmark/mteb/pull/1587),
  [`373db74`](https://github.com/embeddings-benchmark/mteb/commit/373db747d807c3f2597269ac9abf50291673764d))


## v1.24.1 (2024-12-11)

### Bug Fixes

- Add namaa MrTydi reranking dataset
  ([#1573](https://github.com/embeddings-benchmark/mteb/pull/1573),
  [`7b9b3c9`](https://github.com/embeddings-benchmark/mteb/commit/7b9b3c98a26506d64808bdfb082e1f853f3f4f71))


## v1.24.0 (2024-12-10)

### Chores

- Make lint ([#1574](https://github.com/embeddings-benchmark/mteb/pull/1574),
  [`53756ad`](https://github.com/embeddings-benchmark/mteb/commit/53756ad59d48c8fede1bd4a85a9ad3f1ba948cbb))

### Features

- Add new arctic v2.0 models ([#1574](https://github.com/embeddings-benchmark/mteb/pull/1574),
  [`53756ad`](https://github.com/embeddings-benchmark/mteb/commit/53756ad59d48c8fede1bd4a85a9ad3f1ba948cbb))


## v1.23.2 (2024-12-09)

### Bug Fixes

- Added radar chart displaying capabilities on task types
  ([#1570](https://github.com/embeddings-benchmark/mteb/pull/1570),
  [`c49f838`](https://github.com/embeddings-benchmark/mteb/commit/c49f838c2cc4f557d325681bbd1d6cba62e9e1f7))


## v1.23.1 (2024-12-09)

### Bug Fixes

- Added metadata for miscellaneous models
  ([#1557](https://github.com/embeddings-benchmark/mteb/pull/1557),
  [`ce8c175`](https://github.com/embeddings-benchmark/mteb/commit/ce8c17541e61ca259bf73f1b0d634a9cea3f93bd))


## v1.23.0 (2024-12-08)

### Bug Fixes

- Add training dataset to model meta
  ([#1561](https://github.com/embeddings-benchmark/mteb/pull/1561),
  [`6489fca`](https://github.com/embeddings-benchmark/mteb/commit/6489fca1b47f60fd335e6ae644f89cb15fc5f943))

- Bm25s implementation ([#1568](https://github.com/embeddings-benchmark/mteb/pull/1568),
  [`03347eb`](https://github.com/embeddings-benchmark/mteb/commit/03347ebfe4809056e0fd2894fcae69dcdd2ed964))

- Explicitely set `show_progress_bar` to False
  ([#1564](https://github.com/embeddings-benchmark/mteb/pull/1564),
  [`1d21818`](https://github.com/embeddings-benchmark/mteb/commit/1d21818c3704d1866245c21b0f186ac18fa77b9f))

- Use correct task_type ([#1564](https://github.com/embeddings-benchmark/mteb/pull/1564),
  [`1d21818`](https://github.com/embeddings-benchmark/mteb/commit/1d21818c3704d1866245c21b0f186ac18fa77b9f))

- **publichealth-qa**: Ignore rows with `None` values in `question` or `answer`
  ([#1565](https://github.com/embeddings-benchmark/mteb/pull/1565),
  [`68bd8ac`](https://github.com/embeddings-benchmark/mteb/commit/68bd8ac79b33e48942316b26f253db644b6763ad))

### Documentation

- Fix dependency library name for bm25s
  ([#1568](https://github.com/embeddings-benchmark/mteb/pull/1568),
  [`03347eb`](https://github.com/embeddings-benchmark/mteb/commit/03347ebfe4809056e0fd2894fcae69dcdd2ed964))

### Features

- (cohere_models) cohere_task_type issue, batch requests and tqdm for visualization
  ([#1564](https://github.com/embeddings-benchmark/mteb/pull/1564),
  [`1d21818`](https://github.com/embeddings-benchmark/mteb/commit/1d21818c3704d1866245c21b0f186ac18fa77b9f))

- Batch requests to cohere models ([#1564](https://github.com/embeddings-benchmark/mteb/pull/1564),
  [`1d21818`](https://github.com/embeddings-benchmark/mteb/commit/1d21818c3704d1866245c21b0f186ac18fa77b9f))

- Use tqdm with openai ([#1564](https://github.com/embeddings-benchmark/mteb/pull/1564),
  [`1d21818`](https://github.com/embeddings-benchmark/mteb/commit/1d21818c3704d1866245c21b0f186ac18fa77b9f))


## v1.22.1 (2024-12-07)

### Bug Fixes

- Bm25s implementation ([#1566](https://github.com/embeddings-benchmark/mteb/pull/1566),
  [`ac44e58`](https://github.com/embeddings-benchmark/mteb/commit/ac44e58d0a94b9f571f0ca41e004af31dcef3b1b))

- **bm25s**: Search implementation ([#1566](https://github.com/embeddings-benchmark/mteb/pull/1566),
  [`ac44e58`](https://github.com/embeddings-benchmark/mteb/commit/ac44e58d0a94b9f571f0ca41e004af31dcef3b1b))


## v1.22.0 (2024-12-07)

### Bug Fixes

- Google_models batching and prompt
  ([#1562](https://github.com/embeddings-benchmark/mteb/pull/1562),
  [`611b6a1`](https://github.com/embeddings-benchmark/mteb/commit/611b6a175911d7a238f13439243e3c95652a2d85))

### Documentation

- Correction of SICK-R metadata ([#1558](https://github.com/embeddings-benchmark/mteb/pull/1558),
  [`fc64791`](https://github.com/embeddings-benchmark/mteb/commit/fc64791943950f75ff58f522269f3329df341817))

### Features

- **google_models**: Fix issues and add support for `text-embedding-005` and
  `text-multilingual-embedding-002`
  ([#1562](https://github.com/embeddings-benchmark/mteb/pull/1562),
  [`611b6a1`](https://github.com/embeddings-benchmark/mteb/commit/611b6a175911d7a238f13439243e3c95652a2d85))


## v1.21.8 (2024-12-06)

### Bug Fixes

- Add Model2vec ([#1546](https://github.com/embeddings-benchmark/mteb/pull/1546),
  [`2ee8d44`](https://github.com/embeddings-benchmark/mteb/commit/2ee8d44e9ed994860ceae100fab186a209411f42))


## v1.21.7 (2024-12-05)

### Bug Fixes

- Remove curev1 from multlingual ([#1552](https://github.com/embeddings-benchmark/mteb/pull/1552),
  [`279a4ee`](https://github.com/embeddings-benchmark/mteb/commit/279a4ee5fb6cec07c2d4e85800e51c975fa5a45d))


## v1.21.6 (2024-12-04)

### Bug Fixes

- Fixed metadata errors ([#1547](https://github.com/embeddings-benchmark/mteb/pull/1547),
  [`a44a46c`](https://github.com/embeddings-benchmark/mteb/commit/a44a46c3541f4187e692e3a5dd81e3ec9ef9c4f3))


## v1.21.5 (2024-12-04)

### Bug Fixes

- Add docstring for OpenAIWrapper ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Add lint ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Add nomic models ([#1543](https://github.com/embeddings-benchmark/mteb/pull/1543),
  [`5013df8`](https://github.com/embeddings-benchmark/mteb/commit/5013df813621c179f08b5db52450ac9acd18514d))

- Add sentence trimming to OpenAIWrapper
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Add sleep for too many requests ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Added all-minilm-l12-v2 ([#1542](https://github.com/embeddings-benchmark/mteb/pull/1542),
  [`97ab272`](https://github.com/embeddings-benchmark/mteb/commit/97ab2721e5bb73bcf5ed6352366ab333234532f7))

- Added arctic models ([#1541](https://github.com/embeddings-benchmark/mteb/pull/1541),
  [`df11c38`](https://github.com/embeddings-benchmark/mteb/commit/df11c382eb79d6d1b4e9b9a350a14524808f645f))

- Bug cohere names ([#1538](https://github.com/embeddings-benchmark/mteb/pull/1538),
  [`c2f4c26`](https://github.com/embeddings-benchmark/mteb/commit/c2f4c2649114380345115e338a63b26880dd4963))

- Check tokenizer library installed and update ModelMeta to pass tokenizer_name
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Delete changes for ModelMeta ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Delete evaluate file ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Fix revision to 2 for OpenAI models
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Import tiktoken library inside encode function
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Make tokenizer_name None for default
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

- Pass tokenizer_name, max_tokens to loader
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))

### Features

- Add openai optional dependency set
  ([#1526](https://github.com/embeddings-benchmark/mteb/pull/1526),
  [`37fdfa1`](https://github.com/embeddings-benchmark/mteb/commit/37fdfa1e4ef3d4247589ee52adfb2374bf1ee8a5))


## v1.21.4 (2024-12-04)

### Bug Fixes

- Add more model meta (jina, e5) ([#1537](https://github.com/embeddings-benchmark/mteb/pull/1537),
  [`36bab4d`](https://github.com/embeddings-benchmark/mteb/commit/36bab4d345686be0c5c91a2e67c051a286e369a3))

### Documentation

- Add Model Meta parameters and metadata
  ([#1536](https://github.com/embeddings-benchmark/mteb/pull/1536),
  [`5fa7b7b`](https://github.com/embeddings-benchmark/mteb/commit/5fa7b7b1c450db2ff8a5402e38cce0046600b538))


## v1.21.3 (2024-12-02)

### Bug Fixes

- Proprietary models now get correctly shown in leaderboard
  ([#1530](https://github.com/embeddings-benchmark/mteb/pull/1530),
  [`39349ff`](https://github.com/embeddings-benchmark/mteb/commit/39349ff4bc565bc60fa33adc0916c68eee4eb182))


## v1.21.2 (2024-12-01)

### Bug Fixes

- Task load data error for SICK-BR-STS and XStance
  ([#1534](https://github.com/embeddings-benchmark/mteb/pull/1534),
  [`5b6f20f`](https://github.com/embeddings-benchmark/mteb/commit/5b6f20fe6fbe7673480fbb8c36402ddbe7e203a2))


## v1.21.1 (2024-11-30)

### Bug Fixes

- Correct typos superseeded -> superseded
  ([#1532](https://github.com/embeddings-benchmark/mteb/pull/1532),
  [`343b6e0`](https://github.com/embeddings-benchmark/mteb/commit/343b6e055f1fe6784f3fcf9d99e830101bb3e16f))


## v1.21.0 (2024-11-29)

### Bug Fixes

- Evaluate missing splits ([#1525](https://github.com/embeddings-benchmark/mteb/pull/1525),
  [`8e12250`](https://github.com/embeddings-benchmark/mteb/commit/8e1225047d4eed79484c00440fe3f801c512eca5))

### Features

- Evaluate missing splits ([#1525](https://github.com/embeddings-benchmark/mteb/pull/1525),
  [`8e12250`](https://github.com/embeddings-benchmark/mteb/commit/8e1225047d4eed79484c00440fe3f801c512eca5))


## v1.20.6 (2024-11-29)

### Bug Fixes

- Adding missing metadata on models and mathcing names up with the results repo
  ([#1528](https://github.com/embeddings-benchmark/mteb/pull/1528),
  [`b02ae82`](https://github.com/embeddings-benchmark/mteb/commit/b02ae826bd512d8c28afb185fa856ca76e90fc0b))


## v1.20.5 (2024-11-29)

### Bug Fixes

- Ensure that models match the names on embedding-benchmarks/results
  ([#1519](https://github.com/embeddings-benchmark/mteb/pull/1519),
  [`e3d2b54`](https://github.com/embeddings-benchmark/mteb/commit/e3d2b548d8df716bd5ab8ef4f080d7cff82d51cf))

### Documentation

- Add lang family mapping and map to task table
  ([#1486](https://github.com/embeddings-benchmark/mteb/pull/1486),
  [`cfd43ac`](https://github.com/embeddings-benchmark/mteb/commit/cfd43aca70173b93a4d1163b1e6afd52eef41372))


## v1.20.4 (2024-11-27)

### Bug Fixes

- Align readme with current mteb ([#1493](https://github.com/embeddings-benchmark/mteb/pull/1493),
  [`942f212`](https://github.com/embeddings-benchmark/mteb/commit/942f2125dce5534a167416eefe322dcc71dcbcfe))


## v1.20.3 (2024-11-27)

### Bug Fixes

- Leaderboard only shows models that have ModelMeta
  ([#1508](https://github.com/embeddings-benchmark/mteb/pull/1508),
  [`35245d3`](https://github.com/embeddings-benchmark/mteb/commit/35245d36248c0105accaace879f4662def52f5c0))


## v1.20.2 (2024-11-27)

### Bug Fixes

- Leaderboard demo data loading ([#1507](https://github.com/embeddings-benchmark/mteb/pull/1507),
  [`0affa31`](https://github.com/embeddings-benchmark/mteb/commit/0affa31c23727889f56f4bb27da9154ce13ed67a))


## v1.20.1 (2024-11-26)

### Bug Fixes

- Check if `model` attr of model exists
  ([#1499](https://github.com/embeddings-benchmark/mteb/pull/1499),
  [`917ad7f`](https://github.com/embeddings-benchmark/mteb/commit/917ad7f23704edc974c407efda20edc71375041d))


## v1.20.0 (2024-11-21)

### Chores

- Benchmark naming ([#1459](https://github.com/embeddings-benchmark/mteb/pull/1459),
  [`1cc6c9e`](https://github.com/embeddings-benchmark/mteb/commit/1cc6c9e0fe62ca4e77708b641823fa1a121f048b))

### Features

- Add CUREv1 dataset ([#1459](https://github.com/embeddings-benchmark/mteb/pull/1459),
  [`1cc6c9e`](https://github.com/embeddings-benchmark/mteb/commit/1cc6c9e0fe62ca4e77708b641823fa1a121f048b))

- Add CUREv1 retrieval dataset ([#1459](https://github.com/embeddings-benchmark/mteb/pull/1459),
  [`1cc6c9e`](https://github.com/embeddings-benchmark/mteb/commit/1cc6c9e0fe62ca4e77708b641823fa1a121f048b))

- Add missing domains to medical tasks
  ([#1459](https://github.com/embeddings-benchmark/mteb/pull/1459),
  [`1cc6c9e`](https://github.com/embeddings-benchmark/mteb/commit/1cc6c9e0fe62ca4e77708b641823fa1a121f048b))

- Modify benchmark tasks ([#1459](https://github.com/embeddings-benchmark/mteb/pull/1459),
  [`1cc6c9e`](https://github.com/embeddings-benchmark/mteb/commit/1cc6c9e0fe62ca4e77708b641823fa1a121f048b))


## v1.19.10 (2024-11-19)

### Bug Fixes

- Pinned datasets to <3.0.0 ([#1470](https://github.com/embeddings-benchmark/mteb/pull/1470),
  [`fde124a`](https://github.com/embeddings-benchmark/mteb/commit/fde124a8a0894838aabca90b061191e74c33a82f))

### Documentation

- Add sum per language for task counts
  ([#1468](https://github.com/embeddings-benchmark/mteb/pull/1468),
  [`2fb6fe7`](https://github.com/embeddings-benchmark/mteb/commit/2fb6fe764585f0cf6555d15ba9b2e18d4adddcf3))


## v1.19.9 (2024-11-17)

### Bug Fixes

- Swap touche2020 to maintain compatibility
  ([#1469](https://github.com/embeddings-benchmark/mteb/pull/1469),
  [`9b2aece`](https://github.com/embeddings-benchmark/mteb/commit/9b2aecebe00e17b9db02d4fd3182df92222d680d))


## v1.19.8 (2024-11-15)

### Bug Fixes

- Added links to task info table, switched out license with metric
  ([#1461](https://github.com/embeddings-benchmark/mteb/pull/1461),
  [`58c459b`](https://github.com/embeddings-benchmark/mteb/commit/58c459bcd3e1ee772624f723e86efb86e40db6cb))

- Loading pre 1.11.0 ([#1460](https://github.com/embeddings-benchmark/mteb/pull/1460),
  [`1b920ac`](https://github.com/embeddings-benchmark/mteb/commit/1b920ac06bb83eba9530c3ddd125e09fb146dc95))

- Removed column wrapping on the table, so that it remains readable
  ([#1461](https://github.com/embeddings-benchmark/mteb/pull/1461),
  [`58c459b`](https://github.com/embeddings-benchmark/mteb/commit/58c459bcd3e1ee772624f723e86efb86e40db6cb))


## v1.19.7 (2024-11-14)

### Bug Fixes

- Fix load external results with `None` mteb_version
  ([#1453](https://github.com/embeddings-benchmark/mteb/pull/1453),
  [`14d7523`](https://github.com/embeddings-benchmark/mteb/commit/14d7523850edae97cda2a7264f357da29e0ac867))


## v1.19.6 (2024-11-14)

### Bug Fixes

- Publish ([#1452](https://github.com/embeddings-benchmark/mteb/pull/1452),
  [`feb1ab7`](https://github.com/embeddings-benchmark/mteb/commit/feb1ab7652102696a4aa20a03dc98a7240274a20))


## v1.19.5 (2024-11-14)

### Bug Fixes

- Changed jina-embeddings-v3 number of parameters from 572K to 572M
  ([`3a1a470`](https://github.com/embeddings-benchmark/mteb/commit/3a1a470c8e0ad7b8bce61c7f73a501d6716fce5a))

- Count unique texts, data leaks in calculate metrics
  ([#1438](https://github.com/embeddings-benchmark/mteb/pull/1438),
  [`dd5d226`](https://github.com/embeddings-benchmark/mteb/commit/dd5d226f6a377fbf3f98f714323921539a418d83))

- Fixed sentence-transformer compatibility switch
  ([`3a1a470`](https://github.com/embeddings-benchmark/mteb/commit/3a1a470c8e0ad7b8bce61c7f73a501d6716fce5a))

- Fixed use_instuctions typo in model overview
  ([`3a1a470`](https://github.com/embeddings-benchmark/mteb/commit/3a1a470c8e0ad7b8bce61c7f73a501d6716fce5a))

- Made n_parameters formatting smarter and more robust
  ([`3a1a470`](https://github.com/embeddings-benchmark/mteb/commit/3a1a470c8e0ad7b8bce61c7f73a501d6716fce5a))

- Update task metadata to allow for null
  ([#1448](https://github.com/embeddings-benchmark/mteb/pull/1448),
  [`04ac3f2`](https://github.com/embeddings-benchmark/mteb/commit/04ac3f21139db2ea50fdef4d91c345f61f229d44))


## v1.19.4 (2024-11-11)

### Bug Fixes

- Add Korean AutoRAGRetrieval ([#1388](https://github.com/embeddings-benchmark/mteb/pull/1388),
  [`f79d9ba`](https://github.com/embeddings-benchmark/mteb/commit/f79d9ba06c3d7a69c155bc1287c91bba6f41fa62))

- Add metadata for AutoRAGRetrieval
  ([#1388](https://github.com/embeddings-benchmark/mteb/pull/1388),
  [`f79d9ba`](https://github.com/embeddings-benchmark/mteb/commit/f79d9ba06c3d7a69c155bc1287c91bba6f41fa62))

- Add missing benchmarks in benchmarks.py
  ([#1431](https://github.com/embeddings-benchmark/mteb/pull/1431),
  [`a240ea0`](https://github.com/embeddings-benchmark/mteb/commit/a240ea099aac446702a3f7167fd0921f6eb4e259))

- Make samples_per_label a task attribute
  ([#1419](https://github.com/embeddings-benchmark/mteb/pull/1419),
  [`7f1a1d3`](https://github.com/embeddings-benchmark/mteb/commit/7f1a1d33fdc515f39740d4f15b86b011280f1ee6))

- Run --- 🧹 Running linters --- ([#1388](https://github.com/embeddings-benchmark/mteb/pull/1388),
  [`f79d9ba`](https://github.com/embeddings-benchmark/mteb/commit/f79d9ba06c3d7a69c155bc1287c91bba6f41fa62))

### Features

- Add AutoRAG Korean embedding retrieval benchmark
  ([#1388](https://github.com/embeddings-benchmark/mteb/pull/1388),
  [`f79d9ba`](https://github.com/embeddings-benchmark/mteb/commit/f79d9ba06c3d7a69c155bc1287c91bba6f41fa62))


## v1.19.3 (2024-11-11)

### Bug Fixes

- Add logging for RetrievalEvaluator NaN values for similarity scores
  ([#1398](https://github.com/embeddings-benchmark/mteb/pull/1398),
  [`cc7a106`](https://github.com/embeddings-benchmark/mteb/commit/cc7a10666b7c151e9bff66dc50d1413579dac22a))

- Update recommendation for pushing results
  ([#1401](https://github.com/embeddings-benchmark/mteb/pull/1401),
  [`fccf034`](https://github.com/embeddings-benchmark/mteb/commit/fccf034bd78d74917f9d8fb6053e473fb03e86d8))

### Documentation

- Fix a typo in README ([#1430](https://github.com/embeddings-benchmark/mteb/pull/1430),
  [`9681eb3`](https://github.com/embeddings-benchmark/mteb/commit/9681eb38c781ec9c9f1c5395f45cd30bf73ba3fa))

- Update recommendation for pushing results
  ([#1401](https://github.com/embeddings-benchmark/mteb/pull/1401),
  [`fccf034`](https://github.com/embeddings-benchmark/mteb/commit/fccf034bd78d74917f9d8fb6053e473fb03e86d8))


## v1.19.2 (2024-11-07)

### Bug Fixes

- Added the necessary trust_remote_code
  ([#1406](https://github.com/embeddings-benchmark/mteb/pull/1406),
  [`fd8b283`](https://github.com/embeddings-benchmark/mteb/commit/fd8b283e41b555f648d7046e084bcde7af28baf5))


## v1.19.1 (2024-11-07)

### Bug Fixes

- Add the_ugly_duckling.txt for speedtask to Python wheel
  ([#1402](https://github.com/embeddings-benchmark/mteb/pull/1402),
  [`b1a0ec6`](https://github.com/embeddings-benchmark/mteb/commit/b1a0ec67ffcd41bdb7085c8ee995214eb5c5cee6))


## v1.19.0 (2024-11-06)

### Features

- Standardize descriptive stats ([#1375](https://github.com/embeddings-benchmark/mteb/pull/1375),
  [`2854fa2`](https://github.com/embeddings-benchmark/mteb/commit/2854fa2149d301f6c654c492030d7b5dbf66964f))


## v1.18.9 (2024-11-06)

### Bug Fixes

- Disable `rich` output with `verbosity=0` on `evaluation.run`
  ([#1395](https://github.com/embeddings-benchmark/mteb/pull/1395),
  [`1bb1ca3`](https://github.com/embeddings-benchmark/mteb/commit/1bb1ca34fbf220c07d67443ba37f49fa16291b04))

- Removed unnecesary list comprenhension
  ([#1395](https://github.com/embeddings-benchmark/mteb/pull/1395),
  [`1bb1ca3`](https://github.com/embeddings-benchmark/mteb/commit/1bb1ca34fbf220c07d67443ba37f49fa16291b04))

### Features

- Verbose=0 now supress rich console output
  ([#1395](https://github.com/embeddings-benchmark/mteb/pull/1395),
  [`1bb1ca3`](https://github.com/embeddings-benchmark/mteb/commit/1bb1ca34fbf220c07d67443ba37f49fa16291b04))


## v1.18.8 (2024-11-04)

### Bug Fixes

- Update logging verbosity levels in MTEB
  ([#1384](https://github.com/embeddings-benchmark/mteb/pull/1384),
  [`35daf58`](https://github.com/embeddings-benchmark/mteb/commit/35daf58578870678eb08e3d6230ad516c01fec83))


## v1.18.7 (2024-11-04)

### Bug Fixes

- Leaderboard UI improvements ([#1370](https://github.com/embeddings-benchmark/mteb/pull/1370),
  [`92fe9cb`](https://github.com/embeddings-benchmark/mteb/commit/92fe9cbd6bde7800a987185f0e91bc23ad8c2eb6))


## v1.18.6 (2024-10-31)

### Bug Fixes

- Integrate prompts to task metadata
  ([#1300](https://github.com/embeddings-benchmark/mteb/pull/1300),
  [`029d378`](https://github.com/embeddings-benchmark/mteb/commit/029d378b2c17d7e05a4c9c30a17966917cb83a33))


## v1.18.5 (2024-10-31)

### Bug Fixes

- Speed up leaderboard by caching and skipping validation
  ([#1365](https://github.com/embeddings-benchmark/mteb/pull/1365),
  [`f1bc375`](https://github.com/embeddings-benchmark/mteb/commit/f1bc3758d6e1cc91fbe22b26dcbd1cfe3b640f06))


## v1.18.4 (2024-10-30)

### Bug Fixes

- Make sure test is the default split for FEVER
  ([#1361](https://github.com/embeddings-benchmark/mteb/pull/1361),
  [`d9626ab`](https://github.com/embeddings-benchmark/mteb/commit/d9626abbc5d438024a21e7f21a29d4741bb94188))


## v1.18.3 (2024-10-30)

### Bug Fixes

- Update KorSarcasm to avoid trust-remote code
  ([#1364](https://github.com/embeddings-benchmark/mteb/pull/1364),
  [`756ba7e`](https://github.com/embeddings-benchmark/mteb/commit/756ba7e46e6daa8d1bff6b4d3db254296e37e7dc))


## v1.18.2 (2024-10-30)

### Bug Fixes

- Upload BrazilianToxicTweetsClassification to hf
  ([#1352](https://github.com/embeddings-benchmark/mteb/pull/1352),
  [`9c7a1c2`](https://github.com/embeddings-benchmark/mteb/commit/9c7a1c2a8f99966c2d98ec9efe7666ea8b5672a5))


## v1.18.1 (2024-10-30)

### Bug Fixes

- Add jina, uae, stella models ([#1319](https://github.com/embeddings-benchmark/mteb/pull/1319),
  [`0b846ff`](https://github.com/embeddings-benchmark/mteb/commit/0b846ff3ad8ec9f16342c913d81da74aa9ca0643))

- Remove accidentally commited file
  ([`16a333e`](https://github.com/embeddings-benchmark/mteb/commit/16a333ee3b9c1b26468a8b13d42e2697e0474a85))


## v1.18.0 (2024-10-28)

### Features

- Update English benchmarks and mark MMTEB benchmarks as beta
  ([#1341](https://github.com/embeddings-benchmark/mteb/pull/1341),
  [`61371dd`](https://github.com/embeddings-benchmark/mteb/commit/61371dd2db8e3c3a5c7ecd4bb9afff973f7d01d8))


## v1.17.0 (2024-10-26)

### Features

- Update metadata for all models ([#1316](https://github.com/embeddings-benchmark/mteb/pull/1316),
  [`f8fed9b`](https://github.com/embeddings-benchmark/mteb/commit/f8fed9b1567982d8dd590a3a08262537427fcf09))


## v1.16.5 (2024-10-25)

### Bug Fixes

- Add implementations of common reranker models
  ([#1309](https://github.com/embeddings-benchmark/mteb/pull/1309),
  [`f5f90d3`](https://github.com/embeddings-benchmark/mteb/commit/f5f90d3df694b47a7ca81d46c95f1dd2b389ca06))


## v1.16.4 (2024-10-25)

### Bug Fixes

- Re-upload dataset to hub to avoid using script upload
  ([#1322](https://github.com/embeddings-benchmark/mteb/pull/1322),
  [`f00a262`](https://github.com/embeddings-benchmark/mteb/commit/f00a2622821eeec68e191561ca9f2b346f0a5dc6))


## v1.16.3 (2024-10-24)

### Bug Fixes

- Remove duplicate multilingual
  ([`2f14519`](https://github.com/embeddings-benchmark/mteb/commit/2f1451955da42070bf6aea4c317c4bc3da755a38))


## v1.16.2 (2024-10-24)

### Bug Fixes

- Add Slovak Hate Speech and Offensive Language Dataset
  ([#1274](https://github.com/embeddings-benchmark/mteb/pull/1274),
  [`f3d8014`](https://github.com/embeddings-benchmark/mteb/commit/f3d8014fc91dfdf400ab7713683afe4cf785cabf))


## v1.16.1 (2024-10-22)

### Bug Fixes

- Add Retrieval SK Quad dataset for Slovak search evaluation
  ([#1276](https://github.com/embeddings-benchmark/mteb/pull/1276),
  [`fc53498`](https://github.com/embeddings-benchmark/mteb/commit/fc534980b27d3909eaa06943e60480fde41d926e))


## v1.16.0 (2024-10-21)

### Features

- Use prompts instead of encode_corpus and encode_queries
  ([#1278](https://github.com/embeddings-benchmark/mteb/pull/1278),
  [`2a61821`](https://github.com/embeddings-benchmark/mteb/commit/2a61821d9294eb5b0cb053e1e676c199f23be12b))


## v1.15.8 (2024-10-20)

### Bug Fixes

- Remove non-existent eval split of CMNLI
  ([#1294](https://github.com/embeddings-benchmark/mteb/pull/1294),
  [`5b4b262`](https://github.com/embeddings-benchmark/mteb/commit/5b4b262555d8c9d55aec4d178b68be33616c145d))


## v1.15.7 (2024-10-16)

### Bug Fixes

- Add metadata dict to QBQTC in C-MTEB
  ([#1292](https://github.com/embeddings-benchmark/mteb/pull/1292),
  [`4a88a1d`](https://github.com/embeddings-benchmark/mteb/commit/4a88a1d8b515efae58a06b687681a04598ae8db2))


## v1.15.6 (2024-10-14)

### Bug Fixes

- Allow numpy<2.0.0 ([#1291](https://github.com/embeddings-benchmark/mteb/pull/1291),
  [`60cef98`](https://github.com/embeddings-benchmark/mteb/commit/60cef9834045c372861d4700ac87b5e9753e94a3))


## v1.15.5 (2024-10-13)

### Bug Fixes

- Update benchmarks ([#1288](https://github.com/embeddings-benchmark/mteb/pull/1288),
  [`f55a888`](https://github.com/embeddings-benchmark/mteb/commit/f55a888b8244798b0bd5763b8a09481d051fe935))

### Documentation

- Fix a link in the README ([#1289](https://github.com/embeddings-benchmark/mteb/pull/1289),
  [`c1ebd6e`](https://github.com/embeddings-benchmark/mteb/commit/c1ebd6e84acdc790dd3190e854e1a46539c2c902))

- Points for paper writing ([#1286](https://github.com/embeddings-benchmark/mteb/pull/1286),
  [`426f4a1`](https://github.com/embeddings-benchmark/mteb/commit/426f4a136f579f264f3e60a6919ac7d68d265107))


## v1.15.4 (2024-10-07)

### Bug Fixes

- Allow Numpy >=2.0 ([#1264](https://github.com/embeddings-benchmark/mteb/pull/1264),
  [`b2318f5`](https://github.com/embeddings-benchmark/mteb/commit/b2318f5f900eaa6164f8414b9c51268b7b51ecd4))

### Continuous Integration

- Removed 3.8 dependency ([#1281](https://github.com/embeddings-benchmark/mteb/pull/1281),
  [`81081a3`](https://github.com/embeddings-benchmark/mteb/commit/81081a36b4f7b06b9eef28a488a4e68e3d4c912f))


## v1.15.3 (2024-10-06)

### Bug Fixes

- Sorting benchmark tasks by MTEB, then alphabetical
  ([#1271](https://github.com/embeddings-benchmark/mteb/pull/1271),
  [`513ceaf`](https://github.com/embeddings-benchmark/mteb/commit/513ceaf58736662f676f4a03e55fe3449f8760fb))


## v1.15.2 (2024-10-03)

### Bug Fixes

- Derive `results_directory` path from `results_repo` name
  ([#1275](https://github.com/embeddings-benchmark/mteb/pull/1275),
  [`b589c29`](https://github.com/embeddings-benchmark/mteb/commit/b589c29ec3f5af2c962ce4539e0d3695ae8869c3))

- Don't hardcode repo name when downloading results
  ([#1275](https://github.com/embeddings-benchmark/mteb/pull/1275),
  [`b589c29`](https://github.com/embeddings-benchmark/mteb/commit/b589c29ec3f5af2c962ce4539e0d3695ae8869c3))

- Select benchmarks CLI option ([#1261](https://github.com/embeddings-benchmark/mteb/pull/1261),
  [`e717d6e`](https://github.com/embeddings-benchmark/mteb/commit/e717d6ea2778d08ee09fd2ae14671aeba7eabd60))


## v1.15.1 (2024-10-03)

### Bug Fixes

- Add Touche2020v3 and JMTEB ([#1262](https://github.com/embeddings-benchmark/mteb/pull/1262),
  [`5074918`](https://github.com/embeddings-benchmark/mteb/commit/507491884b0bb9e4599594740bb9886ba7b9b2a7))


## v1.15.0 (2024-10-03)

### Bug Fixes

- Default prompt_type to None ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- E4ce987 revision no longer exists for multilingual-e5-small on the Hub
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- Keep `prompt_name` in kwargs when model doesn't have a `prompts` attr
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- Wrong e5 revisions ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

### Documentation

- Specify how to use prompts with Sentence Transformers
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- Update affiliation ([#1248](https://github.com/embeddings-benchmark/mteb/pull/1248),
  [`647c295`](https://github.com/embeddings-benchmark/mteb/commit/647c295c4dc178f902ab4633d4e1d6e8213487eb))

- Update mteb(eng) calculation ([#1258](https://github.com/embeddings-benchmark/mteb/pull/1258),
  [`11518ed`](https://github.com/embeddings-benchmark/mteb/commit/11518edd03b916a63e84505592f0d1a32a058d49))

### Features

- Leverage SentenceTransformer models' query/passage specific prompts
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- Leverage SentenceTransformers' query/passage specific prompts
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- Readd arctic models due to metadata
  ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

- Use Enum for `prompt_type` ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))

### Refactoring

- Remove E5Wrapper ([#1221](https://github.com/embeddings-benchmark/mteb/pull/1221),
  [`c809b84`](https://github.com/embeddings-benchmark/mteb/commit/c809b84d3c7b5bf6f5bca6bbbbdac313e9327d2e))


## v1.14.26 (2024-09-29)

### Bug Fixes

- Add listing all available benchmarks CLI option
  ([#1256](https://github.com/embeddings-benchmark/mteb/pull/1256),
  [`5e1e290`](https://github.com/embeddings-benchmark/mteb/commit/5e1e29064ac6bb49a09c3dbb5d655c0d2b5379e1))


## v1.14.25 (2024-09-29)

### Bug Fixes

- Get meta from CrossEncoder ([#1255](https://github.com/embeddings-benchmark/mteb/pull/1255),
  [`0ad5dad`](https://github.com/embeddings-benchmark/mteb/commit/0ad5dad6591ccbfbf3304525ea20fad9c2710cce))


## v1.14.24 (2024-09-28)

### Bug Fixes

- Downsample large retrieval datasets
  ([#1236](https://github.com/embeddings-benchmark/mteb/pull/1236),
  [`b754f1a`](https://github.com/embeddings-benchmark/mteb/commit/b754f1a578cc6f6868f0666e7a0e2ac9158fe13c))

### Documentation

- Small point changes & more contributors
  ([#1254](https://github.com/embeddings-benchmark/mteb/pull/1254),
  [`0d7664d`](https://github.com/embeddings-benchmark/mteb/commit/0d7664d22cf4eeedb5f84e162ab3aa0115dc7c7c))


## v1.14.23 (2024-09-28)

### Bug Fixes

- Add multilingual bench ([#1252](https://github.com/embeddings-benchmark/mteb/pull/1252),
  [`6a6259c`](https://github.com/embeddings-benchmark/mteb/commit/6a6259c39eda1d831073b7516ec158c617f782f3))

- Add multilingual Benchmark ([#1252](https://github.com/embeddings-benchmark/mteb/pull/1252),
  [`6a6259c`](https://github.com/embeddings-benchmark/mteb/commit/6a6259c39eda1d831073b7516ec158c617f782f3))

### Documentation

- Added coordination point for Jimmy Lee
  ([#1253](https://github.com/embeddings-benchmark/mteb/pull/1253),
  [`6b27ce0`](https://github.com/embeddings-benchmark/mteb/commit/6b27ce0a589e0e8f1370dacd414fa61cf882fb8c))

- Added coordination point for Jimmy lee for his work on the coordination of Crystina and Nandan
  ([#1253](https://github.com/embeddings-benchmark/mteb/pull/1253),
  [`6b27ce0`](https://github.com/embeddings-benchmark/mteb/commit/6b27ce0a589e0e8f1370dacd414fa61cf882fb8c))

- Update affiliation ([#1247](https://github.com/embeddings-benchmark/mteb/pull/1247),
  [`45de3ec`](https://github.com/embeddings-benchmark/mteb/commit/45de3eca1c2487573104e6facd113d2c40907a0f))


## v1.14.22 (2024-09-27)

### Bug Fixes

- @mrshu's name in `points.md` ([#1246](https://github.com/embeddings-benchmark/mteb/pull/1246),
  [`3c06694`](https://github.com/embeddings-benchmark/mteb/commit/3c06694cabb6f6b8d71543aace90f1086cf296e5))

### Documentation

- Add MTEB(code) dataset ([#1237](https://github.com/embeddings-benchmark/mteb/pull/1237),
  [`f808863`](https://github.com/embeddings-benchmark/mteb/commit/f808863f5e393fc472058539fb113efe47e0abf4))

- Create benchmarks overview table ([#1245](https://github.com/embeddings-benchmark/mteb/pull/1245),
  [`fda9be1`](https://github.com/embeddings-benchmark/mteb/commit/fda9be1c085b5d83be58253e32e4aca8e4a2d594))

- Update points ([#1228](https://github.com/embeddings-benchmark/mteb/pull/1228),
  [`a636dc2`](https://github.com/embeddings-benchmark/mteb/commit/a636dc28e968e5689f465983cbdad40481893e6f))


## v1.14.21 (2024-09-20)

### Bug Fixes

- Add RepLLaMA style models ([#1223](https://github.com/embeddings-benchmark/mteb/pull/1223),
  [`bedcfb3`](https://github.com/embeddings-benchmark/mteb/commit/bedcfb3e3991c0573aebb05add24c4f627e14f92))

### Documentation

- Clarify adding a model ([#1222](https://github.com/embeddings-benchmark/mteb/pull/1222),
  [`25b7a2f`](https://github.com/embeddings-benchmark/mteb/commit/25b7a2fd1c6c1c24a48bf3c2c4d8c00dfa2820b9))


## v1.14.20 (2024-09-17)

### Bug Fixes

- Allow benchmark to specify eval_splits
  ([#1217](https://github.com/embeddings-benchmark/mteb/pull/1217),
  [`00260b5`](https://github.com/embeddings-benchmark/mteb/commit/00260b5497b4c82583be6383d8b22a3fceb64b54))


## v1.14.19 (2024-09-14)

### Bug Fixes

- Ensure that results are returned even when hitting cache
  ([#1215](https://github.com/embeddings-benchmark/mteb/pull/1215),
  [`64e01ae`](https://github.com/embeddings-benchmark/mteb/commit/64e01ae9d6fcf125a4ea6516263fa062b2aafeef))

### Documentation

- Fix broken links in docs ([#1212](https://github.com/embeddings-benchmark/mteb/pull/1212),
  [`b1bd941`](https://github.com/embeddings-benchmark/mteb/commit/b1bd9410715aeadf26af34d6845ddd0a7ee3ade8))


## v1.14.18 (2024-09-10)

### Bug Fixes

- Normalize benchmarks no only include task objects and added getter for benchmarks
  ([#1208](https://github.com/embeddings-benchmark/mteb/pull/1208),
  [`f93154f`](https://github.com/embeddings-benchmark/mteb/commit/f93154f465b99bd9737b2ecfd54b3beb491a996d))


## v1.14.17 (2024-09-09)

### Bug Fixes

- Normalize licenses including casing, uses of "-" etc.
  ([#1210](https://github.com/embeddings-benchmark/mteb/pull/1210),
  [`768c031`](https://github.com/embeddings-benchmark/mteb/commit/768c031d3e1e29e39edcf20dd4f9f1ea6092db50))

- Normalize licenses including casing, uses of "-" etc.
  ([`a8f7d80`](https://github.com/embeddings-benchmark/mteb/commit/a8f7d80e20efd97b0c00ef2c028eba830ce1d308))


## v1.14.16 (2024-09-09)

### Bug Fixes

- Added points for ideation and coordination
  ([#1194](https://github.com/embeddings-benchmark/mteb/pull/1194),
  [`8b0834d`](https://github.com/embeddings-benchmark/mteb/commit/8b0834dc1c2480052faec786c9bcd3067e0e2e0a))

- Ensure STS pearson and spearman does not use the p-value only the correlation
  ([#1207](https://github.com/embeddings-benchmark/mteb/pull/1207),
  [`5aa401d`](https://github.com/embeddings-benchmark/mteb/commit/5aa401dcc7ec5bdf6ccbc9cfe1207267a08c4523))

- OpenAI BadRequestError by limiting input dimensions to 2048 elements
  ([#1203](https://github.com/embeddings-benchmark/mteb/pull/1203),
  [`ba562ce`](https://github.com/embeddings-benchmark/mteb/commit/ba562cef8a123f1b760d70b66ad6e1d959c7c3bc))

- OpenAI BadRequestError by limiting input dimensions to 2048 elem…
  ([#1203](https://github.com/embeddings-benchmark/mteb/pull/1203),
  [`ba562ce`](https://github.com/embeddings-benchmark/mteb/commit/ba562cef8a123f1b760d70b66ad6e1d959c7c3bc))

### Continuous Integration

- Remove positional argument ([#1191](https://github.com/embeddings-benchmark/mteb/pull/1191),
  [`b75cd29`](https://github.com/embeddings-benchmark/mteb/commit/b75cd299f724ef78a2b5951f140b509169f1c784))

### Documentation

- Add @xhluca to contributor list ([#1196](https://github.com/embeddings-benchmark/mteb/pull/1196),
  [`660bd1c`](https://github.com/embeddings-benchmark/mteb/commit/660bd1cc858707cbb037d801c1c64729c7d17474))

- Add affiliation of @mrshu ([#1199](https://github.com/embeddings-benchmark/mteb/pull/1199),
  [`75cabc9`](https://github.com/embeddings-benchmark/mteb/commit/75cabc9344c63ccf674689db7970bff17634e3d2))

- Add reranker / cross encoder to README advanced usage
  ([#1186](https://github.com/embeddings-benchmark/mteb/pull/1186),
  [`aa5479d`](https://github.com/embeddings-benchmark/mteb/commit/aa5479da71a40b545dd339d345101d3a02e688c3))

- Added points for ideation and coordination
  ([#1194](https://github.com/embeddings-benchmark/mteb/pull/1194),
  [`8b0834d`](https://github.com/embeddings-benchmark/mteb/commit/8b0834dc1c2480052faec786c9bcd3067e0e2e0a))

- Adding contributor details ([#1195](https://github.com/embeddings-benchmark/mteb/pull/1195),
  [`acd631a`](https://github.com/embeddings-benchmark/mteb/commit/acd631a972495fe0b410644fdac1d3eb84ccdb31))

- Adding contributor details ([#1184](https://github.com/embeddings-benchmark/mteb/pull/1184),
  [`0fc93dc`](https://github.com/embeddings-benchmark/mteb/commit/0fc93dca1170866d2236bfee664b82e05f230b2d))

- Authorship-info for crystina-z ([#1198](https://github.com/embeddings-benchmark/mteb/pull/1198),
  [`08c1efe`](https://github.com/embeddings-benchmark/mteb/commit/08c1efe57387c429ddbec3d36bfa717f99879b8b))

- Update contributors table ([#1189](https://github.com/embeddings-benchmark/mteb/pull/1189),
  [`929733b`](https://github.com/embeddings-benchmark/mteb/commit/929733b4ea172a5d9deeb85ef59af71e5492b863))


## v1.14.15 (2024-09-01)

### Bug Fixes

- Add save prediction cli ([#1187](https://github.com/embeddings-benchmark/mteb/pull/1187),
  [`826cdf5`](https://github.com/embeddings-benchmark/mteb/commit/826cdf513d233d8a71019bf75fef7f3f76991b5e))


## v1.14.14 (2024-09-01)

### Bug Fixes

- Remove test set form eval sets as test labels are unknown
  ([#1190](https://github.com/embeddings-benchmark/mteb/pull/1190),
  [`d375ff7`](https://github.com/embeddings-benchmark/mteb/commit/d375ff7b252309492c7f30f0706f4a4d9388d95c))


## v1.14.13 (2024-09-01)

### Bug Fixes

- Added multilingual and indic benchmark
  ([#1174](https://github.com/embeddings-benchmark/mteb/pull/1174),
  [`74ec7d2`](https://github.com/embeddings-benchmark/mteb/commit/74ec7d2dfb53b1b77746351d153205fb9bbd9383))

- Create_meta ([#1179](https://github.com/embeddings-benchmark/mteb/pull/1179),
  [`a43d1ff`](https://github.com/embeddings-benchmark/mteb/commit/a43d1ff6abf15c6c1dc389e85531af760d654e00))

- Unsure that tests work with the new update
  ([#1174](https://github.com/embeddings-benchmark/mteb/pull/1174),
  [`74ec7d2`](https://github.com/embeddings-benchmark/mteb/commit/74ec7d2dfb53b1b77746351d153205fb9bbd9383))

### Documentation

- Add missing Author ([#1188](https://github.com/embeddings-benchmark/mteb/pull/1188),
  [`24eb38e`](https://github.com/embeddings-benchmark/mteb/commit/24eb38ee25192c5f1800c6d4f07cf6c45717a63b))

- Added notebook for creating authorlist, point table and affil table
  ([#1174](https://github.com/embeddings-benchmark/mteb/pull/1174),
  [`74ec7d2`](https://github.com/embeddings-benchmark/mteb/commit/74ec7d2dfb53b1b77746351d153205fb9bbd9383))

- Adding collaborators user details
  ([#1182](https://github.com/embeddings-benchmark/mteb/pull/1182),
  [`d1e6e5f`](https://github.com/embeddings-benchmark/mteb/commit/d1e6e5f1f487c0562f768f8427ffdfac8a217eef))


## v1.14.12 (2024-08-25)

### Bug Fixes

- Add points ([#1180](https://github.com/embeddings-benchmark/mteb/pull/1180),
  [`86fc8a9`](https://github.com/embeddings-benchmark/mteb/commit/86fc8a96ae39f70aeb0d2b3a43062d4112d3a2b8))


## v1.14.11 (2024-08-25)

### Bug Fixes

- Add citation rumteb ([#1181](https://github.com/embeddings-benchmark/mteb/pull/1181),
  [`605587a`](https://github.com/embeddings-benchmark/mteb/commit/605587a5e9f7aac42885de1975fa5d40e13c4f52))


## v1.14.10 (2024-08-22)

### Bug Fixes

- Add multilingual mocks and test descriptive_stats
  ([#1173](https://github.com/embeddings-benchmark/mteb/pull/1173),
  [`f941aa0`](https://github.com/embeddings-benchmark/mteb/commit/f941aa0c6b405e4b377c74487271fb927bd4a05b))

- Remove unavailable test split from OCNLI
  ([#1175](https://github.com/embeddings-benchmark/mteb/pull/1175),
  [`b21223a`](https://github.com/embeddings-benchmark/mteb/commit/b21223a39baa1d403fe6f6ebe1ef2fef9e15882f))


## v1.14.9 (2024-08-21)

### Bug Fixes

- Added multilingual and indic benchmark
  ([`8c9dada`](https://github.com/embeddings-benchmark/mteb/commit/8c9dadac6b23dc5f05656cc3284109f7d79e03b7))


## v1.14.8 (2024-08-21)

### Bug Fixes

- Added construction of MTEB(eng, v2) and MTEB(eng, v2 lite)
  ([#1168](https://github.com/embeddings-benchmark/mteb/pull/1168),
  [`e861ef2`](https://github.com/embeddings-benchmark/mteb/commit/e861ef24524858800f96012edc887fbd4e67750a))

- Added contruction of MTEB(eng, v2) and its lite version
  ([#1168](https://github.com/embeddings-benchmark/mteb/pull/1168),
  [`e861ef2`](https://github.com/embeddings-benchmark/mteb/commit/e861ef24524858800f96012edc887fbd4e67750a))


## v1.14.7 (2024-08-21)

### Bug Fixes

- Added task selection for the EU benchmark
  ([#1166](https://github.com/embeddings-benchmark/mteb/pull/1166),
  [`0f4f2df`](https://github.com/embeddings-benchmark/mteb/commit/0f4f2dff35381632097ded347ffe5b4b8a767b76))


## v1.14.6 (2024-08-21)

### Bug Fixes

- Add function for metadata create ([#1167](https://github.com/embeddings-benchmark/mteb/pull/1167),
  [`56382b5`](https://github.com/embeddings-benchmark/mteb/commit/56382b53ef33cd747b3fc65aa0d517480c52ef17))


## v1.14.5 (2024-08-19)

### Bug Fixes

- Merge pr 1128 ([#1169](https://github.com/embeddings-benchmark/mteb/pull/1169),
  [`04dbcbb`](https://github.com/embeddings-benchmark/mteb/commit/04dbcbb8535cb81bde1d86ec0e05c0fa865a38d5))

- Update COIR datasets ([#1159](https://github.com/embeddings-benchmark/mteb/pull/1159),
  [`c5bd338`](https://github.com/embeddings-benchmark/mteb/commit/c5bd3381f2efdc985d1538038c50300d733d1923))


## v1.14.4 (2024-08-19)

### Bug Fixes

- Ensure that summarization only calculates the metric across the statistic.
  ([#1157](https://github.com/embeddings-benchmark/mteb/pull/1157),
  [`e57efad`](https://github.com/embeddings-benchmark/mteb/commit/e57efad4655460506111cecbfba9144d813424d4))


## v1.14.3 (2024-08-19)

### Bug Fixes

- Add trust remote code ([#1155](https://github.com/embeddings-benchmark/mteb/pull/1155),
  [`1a1ea2e`](https://github.com/embeddings-benchmark/mteb/commit/1a1ea2eaf9eded2435a613ac12203bbb288d4838))

- Load_data in MintakaRetrieval ([#1155](https://github.com/embeddings-benchmark/mteb/pull/1155),
  [`1a1ea2e`](https://github.com/embeddings-benchmark/mteb/commit/1a1ea2eaf9eded2435a613ac12203bbb288d4838))

### Documentation

- Format notebooks and upgrade ruff
  ([#1164](https://github.com/embeddings-benchmark/mteb/pull/1164),
  [`4e98381`](https://github.com/embeddings-benchmark/mteb/commit/4e98381993fa1c1e55ceeb4e1ba6f5d99d4f0192))


## v1.14.2 (2024-08-15)

### Bug Fixes

- Added Task selection and aggregation
  ([#1140](https://github.com/embeddings-benchmark/mteb/pull/1140),
  [`57f4fb2`](https://github.com/embeddings-benchmark/mteb/commit/57f4fb254391285cb93a82228abf61baafc2b9ba))

- Convert arbitrary imports to absolute imports.
  ([#1140](https://github.com/embeddings-benchmark/mteb/pull/1140),
  [`57f4fb2`](https://github.com/embeddings-benchmark/mteb/commit/57f4fb254391285cb93a82228abf61baafc2b9ba))

- Move benchmarks from script to benchmarks.py
  ([#1140](https://github.com/embeddings-benchmark/mteb/pull/1140),
  [`57f4fb2`](https://github.com/embeddings-benchmark/mteb/commit/57f4fb254391285cb93a82228abf61baafc2b9ba))

### Documentation

- Updating the contributor information
  ([#1161](https://github.com/embeddings-benchmark/mteb/pull/1161),
  [`1325a6a`](https://github.com/embeddings-benchmark/mteb/commit/1325a6a922bc474d2e19b9c307fe30e7bd2a2579))


## v1.14.1 (2024-08-13)

### Bug Fixes

- An item automatically added by linter
  ([#1152](https://github.com/embeddings-benchmark/mteb/pull/1152),
  [`d1794cd`](https://github.com/embeddings-benchmark/mteb/commit/d1794cd2edaf9236699d6d1ccba4cbc15b5f8708))

- Linter error ([#1152](https://github.com/embeddings-benchmark/mteb/pull/1152),
  [`d1794cd`](https://github.com/embeddings-benchmark/mteb/commit/d1794cd2edaf9236699d6d1ccba4cbc15b5f8708))

- Remove kwargs LLM2Vec doesn't have
  ([#1152](https://github.com/embeddings-benchmark/mteb/pull/1152),
  [`d1794cd`](https://github.com/embeddings-benchmark/mteb/commit/d1794cd2edaf9236699d6d1ccba4cbc15b5f8708))

- Removed test set for AFQMC with no gold labels
  ([#1153](https://github.com/embeddings-benchmark/mteb/pull/1153),
  [`14cff3e`](https://github.com/embeddings-benchmark/mteb/commit/14cff3e7830150e28b45098a008048189c7d1e1e))


## v1.14.0 (2024-08-12)

### Bug Fixes

- Convert arbitrary imports to absolute imports.
  ([#1144](https://github.com/embeddings-benchmark/mteb/pull/1144),
  [`ebe6def`](https://github.com/embeddings-benchmark/mteb/commit/ebe6defb6942e7e98edcece6eb8be9820ce7899a))

- Move benchmarks from script to benchmarks.py
  ([#1144](https://github.com/embeddings-benchmark/mteb/pull/1144),
  [`ebe6def`](https://github.com/embeddings-benchmark/mteb/commit/ebe6defb6942e7e98edcece6eb8be9820ce7899a))

### Features

- Cleaned up scripts folder ([#1144](https://github.com/embeddings-benchmark/mteb/pull/1144),
  [`ebe6def`](https://github.com/embeddings-benchmark/mteb/commit/ebe6defb6942e7e98edcece6eb8be9820ce7899a))


## v1.13.2 (2024-08-11)

### Bug Fixes

- Remove unused tests ([#1148](https://github.com/embeddings-benchmark/mteb/pull/1148),
  [`28f592f`](https://github.com/embeddings-benchmark/mteb/commit/28f592f2289b31d8de5e49b76a49e19f94810ebc))


## v1.13.1 (2024-08-10)

### Bug Fixes

- Add CoIR as Benchmark ([#1142](https://github.com/embeddings-benchmark/mteb/pull/1142),
  [`1b80fce`](https://github.com/embeddings-benchmark/mteb/commit/1b80fcebd3a8f1f4b4347895cc8c84b181f1c31c))


## v1.13.0 (2024-08-09)

### Features

- Added in functionality to allow loading outdated results
  ([#1141](https://github.com/embeddings-benchmark/mteb/pull/1141),
  [`d76e686`](https://github.com/embeddings-benchmark/mteb/commit/d76e6868e5cdf04ed790f9fd04d046e15f798fcc))


## v1.12.94 (2024-08-08)

### Bug Fixes

- Add CoIR tasks ([#1130](https://github.com/embeddings-benchmark/mteb/pull/1130),
  [`aa97e39`](https://github.com/embeddings-benchmark/mteb/commit/aa97e3931494e9c68aab47345003b47e1933ad3b))


## v1.12.93 (2024-08-04)

### Bug Fixes

- Allow more linient TaskMetadata ([#1131](https://github.com/embeddings-benchmark/mteb/pull/1131),
  [`8b36887`](https://github.com/embeddings-benchmark/mteb/commit/8b368879bfb7b4d198121ebce3c343f6ba1dea37))


## v1.12.92 (2024-08-02)

### Bug Fixes

- IWSLT2017BitextMining loading dataset.
  ([#1132](https://github.com/embeddings-benchmark/mteb/pull/1132),
  [`d264046`](https://github.com/embeddings-benchmark/mteb/commit/d264046ec76abe8a4e6496ad47dcd83934fdabc3))

- The way of loading and transform dataset and typo of filename on IWSLT2017BitextMining
  ([#1132](https://github.com/embeddings-benchmark/mteb/pull/1132),
  [`d264046`](https://github.com/embeddings-benchmark/mteb/commit/d264046ec76abe8a4e6496ad47dcd83934fdabc3))


## v1.12.91 (2024-08-01)

### Bug Fixes

- When create meta merge results with existing README.md
  ([#1117](https://github.com/embeddings-benchmark/mteb/pull/1117),
  [`61a4c31`](https://github.com/embeddings-benchmark/mteb/commit/61a4c31dc9feed661eeadf018749214ffb32972e))


## v1.12.90 (2024-07-30)

### Bug Fixes

- Handling in case not torch tensor
  ([#1125](https://github.com/embeddings-benchmark/mteb/pull/1125),
  [`9cd2199`](https://github.com/embeddings-benchmark/mteb/commit/9cd2199a16776fc14ef2dd7c8e87b1f4eaebfc35))

- Nomic models using prefix correctly
  ([#1125](https://github.com/embeddings-benchmark/mteb/pull/1125),
  [`9cd2199`](https://github.com/embeddings-benchmark/mteb/commit/9cd2199a16776fc14ef2dd7c8e87b1f4eaebfc35))

### Chores

- Remove comment ([#1125](https://github.com/embeddings-benchmark/mteb/pull/1125),
  [`9cd2199`](https://github.com/embeddings-benchmark/mteb/commit/9cd2199a16776fc14ef2dd7c8e87b1f4eaebfc35))


## v1.12.89 (2024-07-25)

### Bug Fixes

- Ensure that e5 ignores the NQ ([#1085](https://github.com/embeddings-benchmark/mteb/pull/1085),
  [`30e0617`](https://github.com/embeddings-benchmark/mteb/commit/30e061705e7e3015c108516457e95d24cce4c02a))

- Simplify models implementations ([#1085](https://github.com/embeddings-benchmark/mteb/pull/1085),
  [`30e0617`](https://github.com/embeddings-benchmark/mteb/commit/30e061705e7e3015c108516457e95d24cce4c02a))


## v1.12.88 (2024-07-25)

### Bug Fixes

- Export type for `mteb create_meta`
  ([#1114](https://github.com/embeddings-benchmark/mteb/pull/1114),
  [`475967e`](https://github.com/embeddings-benchmark/mteb/commit/475967e396fa9644367df9c95e64a9db5fe48917))

### Documentation

- Improve searchability in the advanced usage documentation
  ([#1113](https://github.com/embeddings-benchmark/mteb/pull/1113),
  [`7492e04`](https://github.com/embeddings-benchmark/mteb/commit/7492e0481ab42d923b83abc979c558502e65eeb2))

- Update based on corrections ([#1113](https://github.com/embeddings-benchmark/mteb/pull/1113),
  [`7492e04`](https://github.com/embeddings-benchmark/mteb/commit/7492e0481ab42d923b83abc979c558502e65eeb2))


## v1.12.87 (2024-07-25)

### Bug Fixes

- Ensure that MLSUMClusteringP2P.v2 use the fast implementation as was intended
  ([#1112](https://github.com/embeddings-benchmark/mteb/pull/1112),
  [`47df0f3`](https://github.com/embeddings-benchmark/mteb/commit/47df0f389d9467b43084f05fb5ee8e4016e03cff))

- Fixed formatting for cli ([#1112](https://github.com/embeddings-benchmark/mteb/pull/1112),
  [`47df0f3`](https://github.com/embeddings-benchmark/mteb/commit/47df0f389d9467b43084f05fb5ee8e4016e03cff))

### Documentation

- Improve searchability in the advanced usage documentation
  ([`7e036de`](https://github.com/embeddings-benchmark/mteb/commit/7e036de02e1b04aab33b3c6363654a9a92fc0e37))


## v1.12.86 (2024-07-25)

### Bug Fixes

- Avoid spaces in dataset name for CQADupstack and ignore speed tasks
  ([`553620c`](https://github.com/embeddings-benchmark/mteb/commit/553620c28971028aa4b7f8e514fc464d8e3148ea))

- MultilingualSentimentClassification
  ([#1109](https://github.com/embeddings-benchmark/mteb/pull/1109),
  [`46e2634`](https://github.com/embeddings-benchmark/mteb/commit/46e2634dc51784498a7158b63cad3b288e029d45))


## v1.12.85 (2024-07-22)

### Bug Fixes

- Fix bug-causing spelling error in function name of e5-mistral-instruct
  ([#1106](https://github.com/embeddings-benchmark/mteb/pull/1106),
  [`2759cc1`](https://github.com/embeddings-benchmark/mteb/commit/2759cc1d3110fde44345d975a857851bb2b6acb0))


## v1.12.84 (2024-07-18)

### Bug Fixes

- Added fix for voyage models to handle too large batches
  ([#1098](https://github.com/embeddings-benchmark/mteb/pull/1098),
  [`2f25626`](https://github.com/embeddings-benchmark/mteb/commit/2f25626f2c66c17477b40c81de95dc43a6cfd885))

- Added misisng license ([#1097](https://github.com/embeddings-benchmark/mteb/pull/1097),
  [`81f0e22`](https://github.com/embeddings-benchmark/mteb/commit/81f0e223d24c121203af27164e89fa37cb6f66b5))

### Continuous Integration

- Splitting doc CI in two to ensure that it runs on PRs AND on main
  ([#1096](https://github.com/embeddings-benchmark/mteb/pull/1096),
  [`304bac0`](https://github.com/embeddings-benchmark/mteb/commit/304bac040ce7133a049a14b71ec587375e7c003c))


## v1.12.83 (2024-07-18)

### Bug Fixes

- Update lang codes for language classification
  ([#1094](https://github.com/embeddings-benchmark/mteb/pull/1094),
  [`930e23c`](https://github.com/embeddings-benchmark/mteb/commit/930e23c583b6a614d4f70542ddecbf5f73cebde5))


## v1.12.82 (2024-07-18)

### Bug Fixes

- Add google API models ([#1087](https://github.com/embeddings-benchmark/mteb/pull/1087),
  [`11443dc`](https://github.com/embeddings-benchmark/mteb/commit/11443dc9c0a9f4c5fbf686dffbf0b9a6f63493fe))


## v1.12.81 (2024-07-16)

### Bug Fixes

- Add bm25s as a model for retrieval
  ([#1082](https://github.com/embeddings-benchmark/mteb/pull/1082),
  [`5269f2c`](https://github.com/embeddings-benchmark/mteb/commit/5269f2c4538cc3b0e691de235d1cd2011e1bab5a))


## v1.12.80 (2024-07-15)

### Bug Fixes

- Ensure that CQADupstackRetrieval is included in results if possible
  ([#1079](https://github.com/embeddings-benchmark/mteb/pull/1079),
  [`3109b4d`](https://github.com/embeddings-benchmark/mteb/commit/3109b4deaf2ca080036240d4f734b304da7fc23a))

### Continuous Integration

- Add missing "fi" at the end ([#1084](https://github.com/embeddings-benchmark/mteb/pull/1084),
  [`c977d35`](https://github.com/embeddings-benchmark/mteb/commit/c977d35996c12ee94546470bb7875d68bb304084))

- Allow docs ci to be run by contributors on PRs
  ([#1084](https://github.com/embeddings-benchmark/mteb/pull/1084),
  [`c977d35`](https://github.com/embeddings-benchmark/mteb/commit/c977d35996c12ee94546470bb7875d68bb304084))

- Allow docs ci to be run by contributors on PRs
  ([#1081](https://github.com/embeddings-benchmark/mteb/pull/1081),
  [`db45092`](https://github.com/embeddings-benchmark/mteb/commit/db45092f31a1be05f264a0752fa68964a35b5927))

- Fix ci docs to work on prs and in main
  ([#1084](https://github.com/embeddings-benchmark/mteb/pull/1084),
  [`c977d35`](https://github.com/embeddings-benchmark/mteb/commit/c977d35996c12ee94546470bb7875d68bb304084))

- Run docs ci on PRs ([#1078](https://github.com/embeddings-benchmark/mteb/pull/1078),
  [`bf02080`](https://github.com/embeddings-benchmark/mteb/commit/bf02080f6062bc28bff597663ed1759a63233535))


## v1.12.79 (2024-07-12)

### Bug Fixes

- Added models x tasks which haven't yet been run
  ([#1038](https://github.com/embeddings-benchmark/mteb/pull/1038),
  [`ae2e2b7`](https://github.com/embeddings-benchmark/mteb/commit/ae2e2b7c8140176f0aa4f3d7d90856f5acbbabb6))


## v1.12.78 (2024-07-12)

### Bug Fixes

- Multi-gpu/simpler models; Add gte
  ([#1059](https://github.com/embeddings-benchmark/mteb/pull/1059),
  [`66ae979`](https://github.com/embeddings-benchmark/mteb/commit/66ae979f29eef1d66cb98870e19773ca76d30819))


## v1.12.77 (2024-07-12)

### Bug Fixes

- Added benchmark construction ([#1003](https://github.com/embeddings-benchmark/mteb/pull/1003),
  [`dd76c67`](https://github.com/embeddings-benchmark/mteb/commit/dd76c674b86c7370de73a3d4d49f9e161f77ea96))


## v1.12.76 (2024-07-12)

### Bug Fixes

- Reduce precision of results for readability
  ([#1070](https://github.com/embeddings-benchmark/mteb/pull/1070),
  [`636018b`](https://github.com/embeddings-benchmark/mteb/commit/636018be6ec04a2051881abd0068ce018e485be5))

- Remove precision for more readable scores and lower memory footprint
  ([#1070](https://github.com/embeddings-benchmark/mteb/pull/1070),
  [`636018b`](https://github.com/embeddings-benchmark/mteb/commit/636018be6ec04a2051881abd0068ce018e485be5))

- Restructure test suite to avoid to many external dependencies
  ([#1070](https://github.com/embeddings-benchmark/mteb/pull/1070),
  [`636018b`](https://github.com/embeddings-benchmark/mteb/commit/636018be6ec04a2051881abd0068ce018e485be5))

### Documentation

- Update contributor table ([#1071](https://github.com/embeddings-benchmark/mteb/pull/1071),
  [`778d7a3`](https://github.com/embeddings-benchmark/mteb/commit/778d7a3bf85b2023cc8ba9b2c35a810dcfa5e924))

### Refactoring

- Update TaskMetadata ([#1076](https://github.com/embeddings-benchmark/mteb/pull/1076),
  [`57c1c12`](https://github.com/embeddings-benchmark/mteb/commit/57c1c12727b6300465cb647f2e55c9af0049d759))


## v1.12.75 (2024-07-09)

### Bug Fixes

- Standardize PairClassification results
  ([#1063](https://github.com/embeddings-benchmark/mteb/pull/1063),
  [`e244777`](https://github.com/embeddings-benchmark/mteb/commit/e2447774c07381b09553f44844e8d89307929157))


## v1.12.74 (2024-07-09)

### Bug Fixes

- Fix Jina model loading ([#1062](https://github.com/embeddings-benchmark/mteb/pull/1062),
  [`4ce6989`](https://github.com/embeddings-benchmark/mteb/commit/4ce6989572454df8cfac43493a3b910383320079))


## v1.12.73 (2024-07-09)

### Bug Fixes

- Simplify ([#1017](https://github.com/embeddings-benchmark/mteb/pull/1017),
  [`a28c722`](https://github.com/embeddings-benchmark/mteb/commit/a28c722d002c6c607175d3c84760cd32dc3849a4))


## v1.12.72 (2024-07-09)

### Bug Fixes

- Ensure that argument are passed correctly for fatihdial
  ([#1048](https://github.com/embeddings-benchmark/mteb/pull/1048),
  [`2f1dc38`](https://github.com/embeddings-benchmark/mteb/commit/2f1dc38f39b57b0e6f78eb5bf36f5d8a48458b2f))

- Fixes for encode_conversation and better default for CLI device
  ([#1048](https://github.com/embeddings-benchmark/mteb/pull/1048),
  [`2f1dc38`](https://github.com/embeddings-benchmark/mteb/commit/2f1dc38f39b57b0e6f78eb5bf36f5d8a48458b2f))

- Reformat encode_corpus to always check for encode_conversation functionality. Additionally remove
  encode_corpus from DresModel ([#1048](https://github.com/embeddings-benchmark/mteb/pull/1048),
  [`2f1dc38`](https://github.com/embeddings-benchmark/mteb/commit/2f1dc38f39b57b0e6f78eb5bf36f5d8a48458b2f))


## v1.12.71 (2024-07-08)

### Bug Fixes

- Reuse corpus utils for LLM2Vec ([#1040](https://github.com/embeddings-benchmark/mteb/pull/1040),
  [`0cb5807`](https://github.com/embeddings-benchmark/mteb/commit/0cb5807385c248d0b356cf19c5deb1e0cbba9976))


## v1.12.70 (2024-07-08)

### Bug Fixes

- Add STS22.v2 and LivedoorNewsClustering.v2
  ([#1055](https://github.com/embeddings-benchmark/mteb/pull/1055),
  [`01c551a`](https://github.com/embeddings-benchmark/mteb/commit/01c551ad2c8648b6345a44205b50bc551084c5e6))

- Ensure reranking kwargs are properly passed
  ([#1058](https://github.com/embeddings-benchmark/mteb/pull/1058),
  [`9f922ae`](https://github.com/embeddings-benchmark/mteb/commit/9f922ae4c740ef11d02e1531ec9000bd9cd3db8e))


## v1.12.69 (2024-07-08)

### Bug Fixes

- Update salesforce_models ([#1053](https://github.com/embeddings-benchmark/mteb/pull/1053),
  [`061c3e3`](https://github.com/embeddings-benchmark/mteb/commit/061c3e3e02e2769430678580bb72765b84670807))

### Documentation

- Points ([#1057](https://github.com/embeddings-benchmark/mteb/pull/1057),
  [`30f12ca`](https://github.com/embeddings-benchmark/mteb/commit/30f12caaef466447aab547f17ad9de57ece44875))


## v1.12.68 (2024-07-05)

### Bug Fixes

- Update __init__ in the classfication task for the swalihiNewsClassfication model to be seen
  ([#1044](https://github.com/embeddings-benchmark/mteb/pull/1044),
  [`0822cd6`](https://github.com/embeddings-benchmark/mteb/commit/0822cd676a6f4f8f6e40a8cee6f9bcba10410afa))


## v1.12.67 (2024-07-04)

### Bug Fixes

- Added Missing Models ([#1014](https://github.com/embeddings-benchmark/mteb/pull/1014),
  [`602f4c0`](https://github.com/embeddings-benchmark/mteb/commit/602f4c0c6c39dde976604d057844d5e1d2e3c5bb))

- Fixed subset warning for later versions of polars
  ([#1014](https://github.com/embeddings-benchmark/mteb/pull/1014),
  [`602f4c0`](https://github.com/embeddings-benchmark/mteb/commit/602f4c0c6c39dde976604d057844d5e1d2e3c5bb))


## v1.12.66 (2024-07-03)

### Bug Fixes

- Added encode_kwargs as the input for encode arguments and added batch_size to CLI
  ([#1030](https://github.com/embeddings-benchmark/mteb/pull/1030),
  [`9032f02`](https://github.com/embeddings-benchmark/mteb/commit/9032f02b56fb0ee6aa43f2cfbfa3a2125ac13c2b))

### Continuous Integration

- Fix outdated CI for pandas ([#1030](https://github.com/embeddings-benchmark/mteb/pull/1030),
  [`9032f02`](https://github.com/embeddings-benchmark/mteb/commit/9032f02b56fb0ee6aa43f2cfbfa3a2125ac13c2b))


## v1.12.65 (2024-07-03)

### Bug Fixes

- Change e5 default to use cuda if available
  ([#1033](https://github.com/embeddings-benchmark/mteb/pull/1033),
  [`91400d3`](https://github.com/embeddings-benchmark/mteb/commit/91400d36a794facc1c8ed703617d2b3ea034a40b))


## v1.12.64 (2024-07-03)

### Bug Fixes

- Minor update __init__ was not included make the repo not being able to be called as the module
  ([#1032](https://github.com/embeddings-benchmark/mteb/pull/1032),
  [`593d349`](https://github.com/embeddings-benchmark/mteb/commit/593d349b05030458983cf18ed87a9564f7b20f47))

### Continuous Integration

- Fix docs ci again ([#1031](https://github.com/embeddings-benchmark/mteb/pull/1031),
  [`2daeeb9`](https://github.com/embeddings-benchmark/mteb/commit/2daeeb977ba5262d273f1b43caee97bae940215c))


## v1.12.63 (2024-07-02)

### Bug Fixes

- Added imenes fix ([#1028](https://github.com/embeddings-benchmark/mteb/pull/1028),
  [`3e5ba61`](https://github.com/embeddings-benchmark/mteb/commit/3e5ba61de0d8b3081f394f1c618937769417d0b8))

- Found and fixes two other mistakes for MLSUMClustering (one caused by python 3.8 and the other in
  the downsampling function) ([#1028](https://github.com/embeddings-benchmark/mteb/pull/1028),
  [`3e5ba61`](https://github.com/embeddings-benchmark/mteb/commit/3e5ba61de0d8b3081f394f1c618937769417d0b8))

- Parallel bitext tasks scores ([#1028](https://github.com/embeddings-benchmark/mteb/pull/1028),
  [`3e5ba61`](https://github.com/embeddings-benchmark/mteb/commit/3e5ba61de0d8b3081f394f1c618937769417d0b8))

### Continuous Integration

- Fix outdated CI for pandas ([#1029](https://github.com/embeddings-benchmark/mteb/pull/1029),
  [`039c976`](https://github.com/embeddings-benchmark/mteb/commit/039c9765d6918ce375fea7aaa5e393ade12dac9d))


## v1.12.62 (2024-07-02)


## v1.12.61 (2024-07-02)

### Bug Fixes

- The Swahili Classification Task ([#998](https://github.com/embeddings-benchmark/mteb/pull/998),
  [`7e4ba06`](https://github.com/embeddings-benchmark/mteb/commit/7e4ba0634feb41b27c9e17b3adbe0e1df914af4b))


## v1.12.60 (2024-07-02)

### Bug Fixes

- Added imenes fix ([#1026](https://github.com/embeddings-benchmark/mteb/pull/1026),
  [`ca9e81d`](https://github.com/embeddings-benchmark/mteb/commit/ca9e81de8e950eef823e31e6799fe9e3fa5767dd))

- Added imenes fix
  ([`11d5570`](https://github.com/embeddings-benchmark/mteb/commit/11d557045c97f211bb1887d56e09477939b9a8f6))

- Fix broken clustering tasks ([#1026](https://github.com/embeddings-benchmark/mteb/pull/1026),
  [`ca9e81d`](https://github.com/embeddings-benchmark/mteb/commit/ca9e81de8e950eef823e31e6799fe9e3fa5767dd))

- Fixed subset warning for later versions of polars
  ([#1016](https://github.com/embeddings-benchmark/mteb/pull/1016),
  [`4e1f43b`](https://github.com/embeddings-benchmark/mteb/commit/4e1f43b1d56843fe36637068d64e5a09e63964fd))

- Found and fixes two other mistakes for MLSUMClustering (one caused by python 3.8 and the other in
  the downsampling function) ([#1026](https://github.com/embeddings-benchmark/mteb/pull/1026),
  [`ca9e81d`](https://github.com/embeddings-benchmark/mteb/commit/ca9e81de8e950eef823e31e6799fe9e3fa5767dd))

### Documentation

- Add Mariya's points ([#1016](https://github.com/embeddings-benchmark/mteb/pull/1016),
  [`4e1f43b`](https://github.com/embeddings-benchmark/mteb/commit/4e1f43b1d56843fe36637068d64e5a09e63964fd))

- Added result tables ([#996](https://github.com/embeddings-benchmark/mteb/pull/996),
  [`f809cad`](https://github.com/embeddings-benchmark/mteb/commit/f809cadcbaa259f8bbfc5d3fe98de217023883f1))

- Update figure ([#1027](https://github.com/embeddings-benchmark/mteb/pull/1027),
  [`0acf259`](https://github.com/embeddings-benchmark/mteb/commit/0acf2590b9b6059fc5dbe66c31b1fc5f313efdd6))

- Update points.md ([#1010](https://github.com/embeddings-benchmark/mteb/pull/1010),
  [`1d1f7aa`](https://github.com/embeddings-benchmark/mteb/commit/1d1f7aadc71b89444837ea59bd142107789f648a))


## v1.12.59 (2024-06-30)

### Bug Fixes

- Add more russian models ([#1000](https://github.com/embeddings-benchmark/mteb/pull/1000),
  [`d97310d`](https://github.com/embeddings-benchmark/mteb/commit/d97310d6bc090d4e7bb0fde93e52b9241044bc7e))


## v1.12.58 (2024-06-28)

### Bug Fixes

- Added utility function for downloading and loading models results
  ([#995](https://github.com/embeddings-benchmark/mteb/pull/995),
  [`abef8dc`](https://github.com/embeddings-benchmark/mteb/commit/abef8dc2366d56c790a99ed205b97fc81d0894c2))


## v1.12.57 (2024-06-27)

### Bug Fixes

- Revert changes to danFEVER ([#994](https://github.com/embeddings-benchmark/mteb/pull/994),
  [`56ba2d7`](https://github.com/embeddings-benchmark/mteb/commit/56ba2d7657976180ef95d83eec9743de851c5542))

- Update DanFever to use avoid using overwrite=True and ignore identical ids
  ([#994](https://github.com/embeddings-benchmark/mteb/pull/994),
  [`56ba2d7`](https://github.com/embeddings-benchmark/mteb/commit/56ba2d7657976180ef95d83eec9743de851c5542))

- Updated DanFever to use avoid using overwrite=True and ignore identical ids
  ([#994](https://github.com/embeddings-benchmark/mteb/pull/994),
  [`56ba2d7`](https://github.com/embeddings-benchmark/mteb/commit/56ba2d7657976180ef95d83eec9743de851c5542))

- Updated logging, adding utility helper when specifying the wrong task and added --overwrite flag
  to CLI ([#994](https://github.com/embeddings-benchmark/mteb/pull/994),
  [`56ba2d7`](https://github.com/embeddings-benchmark/mteb/commit/56ba2d7657976180ef95d83eec9743de851c5542))


## v1.12.56 (2024-06-27)

### Bug Fixes

- Convert scores in meta to be expressed in 0-100 instead of 0-1
  ([#993](https://github.com/embeddings-benchmark/mteb/pull/993),
  [`870a2d9`](https://github.com/embeddings-benchmark/mteb/commit/870a2d9417127dfacc18f42943af46a97c2b90e0))

- Convert scores in meta to be in 0-100 instead of 0-1
  ([#993](https://github.com/embeddings-benchmark/mteb/pull/993),
  [`870a2d9`](https://github.com/embeddings-benchmark/mteb/commit/870a2d9417127dfacc18f42943af46a97c2b90e0))


## v1.12.55 (2024-06-26)

### Bug Fixes

- Add Nomic and Multilingual Cohere ([#987](https://github.com/embeddings-benchmark/mteb/pull/987),
  [`e398f20`](https://github.com/embeddings-benchmark/mteb/commit/e398f2012c189da1a88d18e9b9bc0e87882c675d))


## v1.12.54 (2024-06-25)

### Bug Fixes

- Update Retrieval Statistics Checker
  ([#985](https://github.com/embeddings-benchmark/mteb/pull/985),
  [`230c311`](https://github.com/embeddings-benchmark/mteb/commit/230c3110f5dbee38ebd45425973cb262defac3fc))


## v1.12.53 (2024-06-25)

### Bug Fixes

- Add MIRACL retrieval ([#833](https://github.com/embeddings-benchmark/mteb/pull/833),
  [`306e480`](https://github.com/embeddings-benchmark/mteb/commit/306e4807c50e49536e0ec34d052e49bab998c0b2))


## v1.12.52 (2024-06-25)

### Bug Fixes

- Voyage bclavie/mmarco-japanese-hard-negatives
  ([#980](https://github.com/embeddings-benchmark/mteb/pull/980),
  [`06e0a8b`](https://github.com/embeddings-benchmark/mteb/commit/06e0a8b54885f8886a6ffb57af2f17ed5c1743a7))


## v1.12.51 (2024-06-25)

### Bug Fixes

- Ensure that results from parallel datasets are formatted corrected.
  ([#974](https://github.com/embeddings-benchmark/mteb/pull/974),
  [`6004ec7`](https://github.com/embeddings-benchmark/mteb/commit/6004ec7b6e99afb2d31a41784ac0b3d4a6ded935))

- Ensure that results from parallel datasets are formatted correctly
  ([#974](https://github.com/embeddings-benchmark/mteb/pull/974),
  [`6004ec7`](https://github.com/embeddings-benchmark/mteb/commit/6004ec7b6e99afb2d31a41784ac0b3d4a6ded935))


## v1.12.50 (2024-06-25)

### Bug Fixes

- GritLM Retrieval instructions ([#981](https://github.com/embeddings-benchmark/mteb/pull/981),
  [`f1c9fc7`](https://github.com/embeddings-benchmark/mteb/commit/f1c9fc775ef0edfa83e6d32e365d1e12663273b8))


## v1.12.49 (2024-06-24)

### Bug Fixes

- Add max_fraction_of_documents_to_embed to clustering datasets with `max_document_to_embed`
  ([#977](https://github.com/embeddings-benchmark/mteb/pull/977),
  [`5367193`](https://github.com/embeddings-benchmark/mteb/commit/536719374e0e179fb1354f02e0a511d7ceefaa70))


## v1.12.48 (2024-06-21)

### Bug Fixes

- Add Speed tasks for CPU and GPU & system info
  ([#967](https://github.com/embeddings-benchmark/mteb/pull/967),
  [`d51fa81`](https://github.com/embeddings-benchmark/mteb/commit/d51fa818ad93588049b8e44b9845c044cb0eb93d))

- Bug in calculate_metadata_metrics for some retrieval datasets
  ([#965](https://github.com/embeddings-benchmark/mteb/pull/965),
  [`f3b5b1c`](https://github.com/embeddings-benchmark/mteb/commit/f3b5b1c77febd326615d645afb13635088c53474))

### Documentation

- Fix cli commands ([#966](https://github.com/embeddings-benchmark/mteb/pull/966),
  [`d403580`](https://github.com/embeddings-benchmark/mteb/commit/d40358089c959f812d150e40f9338f2d784bd73c))


## v1.12.47 (2024-06-20)

### Bug Fixes

- Add E5 model test case in test CLI ([#958](https://github.com/embeddings-benchmark/mteb/pull/958),
  [`6c0f597`](https://github.com/embeddings-benchmark/mteb/commit/6c0f597e62e165f6ecef3116c0002093fbdd610c))


## v1.12.46 (2024-06-20)

### Bug Fixes

- Added models to run along with minor fixes
  ([#953](https://github.com/embeddings-benchmark/mteb/pull/953),
  [`9b5891d`](https://github.com/embeddings-benchmark/mteb/commit/9b5891d426eb2bc282265c620c47b35b2ea903d3))

- Allow for arbitrary None as a model revision
  ([#953](https://github.com/embeddings-benchmark/mteb/pull/953),
  [`9b5891d`](https://github.com/embeddings-benchmark/mteb/commit/9b5891d426eb2bc282265c620c47b35b2ea903d3))

- Ensure prompt name is passed correctly in cases of encode_corpus and encode_queries
  ([#953](https://github.com/embeddings-benchmark/mteb/pull/953),
  [`9b5891d`](https://github.com/embeddings-benchmark/mteb/commit/9b5891d426eb2bc282265c620c47b35b2ea903d3))

- Loader naming ([#953](https://github.com/embeddings-benchmark/mteb/pull/953),
  [`9b5891d`](https://github.com/embeddings-benchmark/mteb/commit/9b5891d426eb2bc282265c620c47b35b2ea903d3))


## v1.12.45 (2024-06-20)

### Bug Fixes

- Add baseline models for Russian ([#962](https://github.com/embeddings-benchmark/mteb/pull/962),
  [`664f6da`](https://github.com/embeddings-benchmark/mteb/commit/664f6da9c14940eeddf1c9e794bcdb69563c5d5e))

- Update annotations for PawsX, Opusparcus and SummEval
  ([#963](https://github.com/embeddings-benchmark/mteb/pull/963),
  [`211d5ae`](https://github.com/embeddings-benchmark/mteb/commit/211d5ae402a7b3a9c86d62db0c4892106e03ad8e))

### Documentation

- Add missing points ([#959](https://github.com/embeddings-benchmark/mteb/pull/959),
  [`3ebd148`](https://github.com/embeddings-benchmark/mteb/commit/3ebd148ad1956615a43c77e5a1b4c3ea8bcbe59b))


## v1.12.44 (2024-06-19)

### Bug Fixes

- Add test case for results folder structure
  ([#956](https://github.com/embeddings-benchmark/mteb/pull/956),
  [`b0f597c`](https://github.com/embeddings-benchmark/mteb/commit/b0f597c19643feee029573a1b46bd34c84161e1d))


## v1.12.43 (2024-06-18)

### Bug Fixes

- Merge CrosslingualTask into MultilingualTask
  ([#952](https://github.com/embeddings-benchmark/mteb/pull/952),
  [`07f80c4`](https://github.com/embeddings-benchmark/mteb/commit/07f80c479a7223b188341e3be6ac5e0424f297b7))


## v1.12.42 (2024-06-18)

### Bug Fixes

- Backward compatibility fixes for clustering
  ([#954](https://github.com/embeddings-benchmark/mteb/pull/954),
  [`623d833`](https://github.com/embeddings-benchmark/mteb/commit/623d83300157921fe71bc78aa6700c85a5f45486))


## v1.12.41 (2024-06-18)

### Bug Fixes

- Add MINERS Bitext retrieval benchmark
  ([#951](https://github.com/embeddings-benchmark/mteb/pull/951),
  [`f95b9e0`](https://github.com/embeddings-benchmark/mteb/commit/f95b9e0e17ec36272e249fbb754b7f7020727303))


## v1.12.40 (2024-06-18)

### Bug Fixes

- Compare Cluster and ClusterFast scores and speedup
  ([#892](https://github.com/embeddings-benchmark/mteb/pull/892),
  [`2bb7623`](https://github.com/embeddings-benchmark/mteb/commit/2bb76239368c497efb92d5ae09a914eedd44a66d))

### Documentation

- Add point for PR 948 ([#950](https://github.com/embeddings-benchmark/mteb/pull/950),
  [`34286f2`](https://github.com/embeddings-benchmark/mteb/commit/34286f2a36d8bf11c0bab1160d38c5cae3b95461))


## v1.12.39 (2024-06-18)

### Bug Fixes

- Add LinceMT Bitext Mining (MINERS) ([#948](https://github.com/embeddings-benchmark/mteb/pull/948),
  [`6bd165e`](https://github.com/embeddings-benchmark/mteb/commit/6bd165e95d1fe2cc623ffa7faa79aae8201d5dc8))

- Add Phinc Bitext Mining (MINERS) ([#947](https://github.com/embeddings-benchmark/mteb/pull/947),
  [`0e71110`](https://github.com/embeddings-benchmark/mteb/commit/0e711102dbe2bdef4063099ba83af99fef16839c))

- Pair classification inconsistency ([#945](https://github.com/embeddings-benchmark/mteb/pull/945),
  [`6660f43`](https://github.com/embeddings-benchmark/mteb/commit/6660f432bd501eb2bbdd131fe61d697ef547e755))

### Documentation

- Added a script to extract the bibtex citations and generate the consolidated bib file
  ([#904](https://github.com/embeddings-benchmark/mteb/pull/904),
  [`ab23552`](https://github.com/embeddings-benchmark/mteb/commit/ab235525d39402d9e1460eb7662d96928e055927))


## v1.12.38 (2024-06-17)

### Bug Fixes

- Merge miracl evaluator ([#906](https://github.com/embeddings-benchmark/mteb/pull/906),
  [`8ab4c14`](https://github.com/embeddings-benchmark/mteb/commit/8ab4c141313d65a6f9e265a156894889e3d32565))


## v1.12.37 (2024-06-17)

### Bug Fixes

- Add JaGovFaqs and NLPJournal datasets
  ([#938](https://github.com/embeddings-benchmark/mteb/pull/938),
  [`f38c79b`](https://github.com/embeddings-benchmark/mteb/commit/f38c79b33eb3d306a557c56234b43601a4307ffc))

- Add jmteb ([#938](https://github.com/embeddings-benchmark/mteb/pull/938),
  [`f38c79b`](https://github.com/embeddings-benchmark/mteb/commit/f38c79b33eb3d306a557c56234b43601a4307ffc))

- Correct label for sib200 ([#938](https://github.com/embeddings-benchmark/mteb/pull/938),
  [`f38c79b`](https://github.com/embeddings-benchmark/mteb/commit/f38c79b33eb3d306a557c56234b43601a4307ffc))


## v1.12.36 (2024-06-17)

### Bug Fixes

- Temporarily limit numpy version due to bug in datasets
  ([#940](https://github.com/embeddings-benchmark/mteb/pull/940),
  [`0b0705e`](https://github.com/embeddings-benchmark/mteb/commit/0b0705e0b485d3236703b68408139ca14f286b53))


## v1.12.35 (2024-06-17)

### Bug Fixes

- RAR-b initial PR ([#929](https://github.com/embeddings-benchmark/mteb/pull/929),
  [`b75a9c9`](https://github.com/embeddings-benchmark/mteb/commit/b75a9c9fe6c976f12a9e6d7219901b9fedb99a69))

### Documentation

- Update annotations for tasks ([#936](https://github.com/embeddings-benchmark/mteb/pull/936),
  [`8823369`](https://github.com/embeddings-benchmark/mteb/commit/8823369b91c82d7641367cfb0b9cdfec075143b1))


## v1.12.34 (2024-06-16)


## v1.12.33 (2024-06-15)

### Bug Fixes

- Add NollySenti Bitext Mining (MINERS)
  ([#915](https://github.com/embeddings-benchmark/mteb/pull/915),
  [`df68f8c`](https://github.com/embeddings-benchmark/mteb/commit/df68f8c7aadbee9b09da4193b9d6badc3990a97b))

- Add NusaParagraph Emotion Classification
  ([#928](https://github.com/embeddings-benchmark/mteb/pull/928),
  [`5e4ad44`](https://github.com/embeddings-benchmark/mteb/commit/5e4ad442535a20719f42457ca034269df3de3646))

- Add NusaParagraph Topic Classification
  ([#927](https://github.com/embeddings-benchmark/mteb/pull/927),
  [`e13f037`](https://github.com/embeddings-benchmark/mteb/commit/e13f0371341dd08b7868f5e049478dae891a27a5))


## v1.12.32 (2024-06-15)

### Bug Fixes

- Add NusaTranslation Bitext Mining (MINERS)
  ([#914](https://github.com/embeddings-benchmark/mteb/pull/914),
  [`78f34ad`](https://github.com/embeddings-benchmark/mteb/commit/78f34ad33841ee1a15073cee8a5af1d7e881c67d))

- Add NusaX Bitext Mining (MINERS) ([#910](https://github.com/embeddings-benchmark/mteb/pull/910),
  [`25c7606`](https://github.com/embeddings-benchmark/mteb/commit/25c76061db029f4dd65e4542a1199b2f460b34ab))


## v1.12.31 (2024-06-15)

### Bug Fixes

- Add STS dataset SemRel2024 ([#917](https://github.com/embeddings-benchmark/mteb/pull/917),
  [`319ed83`](https://github.com/embeddings-benchmark/mteb/commit/319ed833ae744cfc1006060638324a7c3fe67256))

### Documentation

- Added documentatio for wiki clustering
  ([#934](https://github.com/embeddings-benchmark/mteb/pull/934),
  [`8e39cfb`](https://github.com/embeddings-benchmark/mteb/commit/8e39cfb036a922d3f457eeb898b9d0e3857a9d21))

- Added documentation for wiki clustering
  ([#934](https://github.com/embeddings-benchmark/mteb/pull/934),
  [`8e39cfb`](https://github.com/embeddings-benchmark/mteb/commit/8e39cfb036a922d3f457eeb898b9d0e3857a9d21))


## v1.12.30 (2024-06-15)

### Bug Fixes

- Mteb meta now includes all scores
  ([`49f2c3b`](https://github.com/embeddings-benchmark/mteb/commit/49f2c3b8957c9d4c65afd377187b4a673234992c))

- Update annotations for multilingual classification tasks
  ([#923](https://github.com/embeddings-benchmark/mteb/pull/923),
  [`568651b`](https://github.com/embeddings-benchmark/mteb/commit/568651bab6dd57ae8fc2216d5e192da351f44a98))


## v1.12.29 (2024-06-15)

### Bug Fixes

- Added the ability to make task specific instructions, tested using e5 instruct
  ([#888](https://github.com/embeddings-benchmark/mteb/pull/888),
  [`40208cf`](https://github.com/embeddings-benchmark/mteb/commit/40208cfaf0bf588515e3a077b4d7e5fc5fa98200))


## v1.12.28 (2024-06-15)

### Bug Fixes

- Fix create_meta function in cli.py ([#912](https://github.com/embeddings-benchmark/mteb/pull/912),
  [`af365b7`](https://github.com/embeddings-benchmark/mteb/commit/af365b795e2c54b25b0e02bbb0788ba9d3c403c7))


## v1.12.27 (2024-06-13)

### Bug Fixes

- Update annotations for English STS tasks
  ([#908](https://github.com/embeddings-benchmark/mteb/pull/908),
  [`f1dd8bb`](https://github.com/embeddings-benchmark/mteb/commit/f1dd8bb60fe730d34fcd7bb909834ad1d9daba80))

### Documentation

- Add points for #911 ([#913](https://github.com/embeddings-benchmark/mteb/pull/913),
  [`5deeb3c`](https://github.com/embeddings-benchmark/mteb/commit/5deeb3c6351fbedde24dc77d8981e32562d9b516))


## v1.12.26 (2024-06-13)

### Bug Fixes

- Incorrect handling of qrel_revision fix: #909
  ([#911](https://github.com/embeddings-benchmark/mteb/pull/911),
  [`63cd4b7`](https://github.com/embeddings-benchmark/mteb/commit/63cd4b7f99e9371a90e26014a328231b1f3db810))

### Documentation

- Update annotations for multilingual STS tasks
  ([#902](https://github.com/embeddings-benchmark/mteb/pull/902),
  [`4383fd3`](https://github.com/embeddings-benchmark/mteb/commit/4383fd3a65c16df9598581f32643408e33273cb0))


## v1.12.25 (2024-06-11)

### Bug Fixes

- Backfilled bibtex citations data ([#900](https://github.com/embeddings-benchmark/mteb/pull/900),
  [`77d0e06`](https://github.com/embeddings-benchmark/mteb/commit/77d0e06e88e7645d14c278b25a9bbb6c97b8fed6))

### Documentation

- Add points for paper writing ([#901](https://github.com/embeddings-benchmark/mteb/pull/901),
  [`fbbc44b`](https://github.com/embeddings-benchmark/mteb/commit/fbbc44b51702930d3dafbc6f7529233c5de7a138))


## v1.12.24 (2024-06-09)

### Bug Fixes

- Add openai and voyage models ([#887](https://github.com/embeddings-benchmark/mteb/pull/887),
  [`ad9b3ce`](https://github.com/embeddings-benchmark/mteb/commit/ad9b3ce2bfd4ac90dab74d397d740afbe8f142a5))


## v1.12.23 (2024-06-08)

### Bug Fixes

- Abstention metric for small datasets
  ([#893](https://github.com/embeddings-benchmark/mteb/pull/893),
  [`9d28296`](https://github.com/embeddings-benchmark/mteb/commit/9d2829671c5e2833bcc65c37f65c2ed87c3ae825))

### Documentation

- Update points.md ([#890](https://github.com/embeddings-benchmark/mteb/pull/890),
  [`4318c82`](https://github.com/embeddings-benchmark/mteb/commit/4318c8223dcd1f82038eb63d2cbe4970af1d539b))


## v1.12.22 (2024-06-06)

### Bug Fixes

- Add GritLM ([#880](https://github.com/embeddings-benchmark/mteb/pull/880),
  [`0c99a4e`](https://github.com/embeddings-benchmark/mteb/commit/0c99a4ebcc4a6f3a47fe5e59161c832ddbea9294))


## v1.12.21 (2024-06-05)

### Bug Fixes

- Add error reporting for Retrieval ([#873](https://github.com/embeddings-benchmark/mteb/pull/873),
  [`5397bd2`](https://github.com/embeddings-benchmark/mteb/commit/5397bd2153700ff026697038d6174fe67b6033d8))

### Documentation

- Added source for CmedqaRetrieval ([#886](https://github.com/embeddings-benchmark/mteb/pull/886),
  [`3e910ff`](https://github.com/embeddings-benchmark/mteb/commit/3e910ff76b5cc4713682d2cef2dd860e38de513a))

- Update CmedqaRetrieval description to specify source
  ([#886](https://github.com/embeddings-benchmark/mteb/pull/886),
  [`3e910ff`](https://github.com/embeddings-benchmark/mteb/commit/3e910ff76b5cc4713682d2cef2dd860e38de513a))


## v1.12.20 (2024-06-05)

### Bug Fixes

- Updated CLI for MTEB ([#882](https://github.com/embeddings-benchmark/mteb/pull/882),
  [`c6f618b`](https://github.com/embeddings-benchmark/mteb/commit/c6f618b0ab5b265acf8ac736db1ddca8d73222c5))


## v1.12.19 (2024-06-05)

### Bug Fixes

- Add CEDR, SensitiveTopics for multilabel and RuBQ for reranking
  ([#881](https://github.com/embeddings-benchmark/mteb/pull/881),
  [`9128df4`](https://github.com/embeddings-benchmark/mteb/commit/9128df46f4d9deb06ad1878b321942ac281cfc7d))

### Documentation

- Minor fix for point validation to avoid error when people split up points
  ([`7d0d631`](https://github.com/embeddings-benchmark/mteb/commit/7d0d6319f67251b65b37ec835adea605aa05c893))


## v1.12.18 (2024-06-05)

### Bug Fixes

- Added benchmark object ([#876](https://github.com/embeddings-benchmark/mteb/pull/876),
  [`fb843d0`](https://github.com/embeddings-benchmark/mteb/commit/fb843d040e8af63b4d0c2d61b78a69ca55652bcd))

- Ensure result are consistently stored in the same way
  ([#876](https://github.com/embeddings-benchmark/mteb/pull/876),
  [`fb843d0`](https://github.com/embeddings-benchmark/mteb/commit/fb843d040e8af63b4d0c2d61b78a69ca55652bcd))

### Documentation

- Minor updated to repro. workflow docs
  ([#876](https://github.com/embeddings-benchmark/mteb/pull/876),
  [`fb843d0`](https://github.com/embeddings-benchmark/mteb/commit/fb843d040e8af63b4d0c2d61b78a69ca55652bcd))

- Update PR template ([#876](https://github.com/embeddings-benchmark/mteb/pull/876),
  [`fb843d0`](https://github.com/embeddings-benchmark/mteb/commit/fb843d040e8af63b4d0c2d61b78a69ca55652bcd))


## v1.12.17 (2024-06-04)

### Bug Fixes

- Add FaithDialRetrieval dataset ([#874](https://github.com/embeddings-benchmark/mteb/pull/874),
  [`582381b`](https://github.com/embeddings-benchmark/mteb/commit/582381bd75ec111ba1ed8c81dc2df21336091546))


## v1.12.16 (2024-06-04)

### Bug Fixes

- Add feedbackQA dataset ([#856](https://github.com/embeddings-benchmark/mteb/pull/856),
  [`0796efa`](https://github.com/embeddings-benchmark/mteb/commit/0796efa4f987fa65920b3853455c1e592ca6b697))


## v1.12.15 (2024-06-04)

### Bug Fixes

- Add MIRACL reranking ([#830](https://github.com/embeddings-benchmark/mteb/pull/830),
  [`4218675`](https://github.com/embeddings-benchmark/mteb/commit/421867588007e7664a1584e3f273d2978b3bade1))

- Fixed missing references ([#830](https://github.com/embeddings-benchmark/mteb/pull/830),
  [`4218675`](https://github.com/embeddings-benchmark/mteb/commit/421867588007e7664a1584e3f273d2978b3bade1))

- MIRACL reranking ([#830](https://github.com/embeddings-benchmark/mteb/pull/830),
  [`4218675`](https://github.com/embeddings-benchmark/mteb/commit/421867588007e7664a1584e3f273d2978b3bade1))

- Miracl reranking fix ([#830](https://github.com/embeddings-benchmark/mteb/pull/830),
  [`4218675`](https://github.com/embeddings-benchmark/mteb/commit/421867588007e7664a1584e3f273d2978b3bade1))

### Documentation

- Affiliation modification ([#871](https://github.com/embeddings-benchmark/mteb/pull/871),
  [`6e43bbf`](https://github.com/embeddings-benchmark/mteb/commit/6e43bbf389b8c71e1d15b50054a0183789d6c580))


## v1.12.14 (2024-06-03)

### Bug Fixes

- Dataset not available ([#872](https://github.com/embeddings-benchmark/mteb/pull/872),
  [`397519b`](https://github.com/embeddings-benchmark/mteb/commit/397519b4aea562dc2912d49874ec819550f2e28f))


## v1.12.13 (2024-06-03)

### Bug Fixes

- Main score for longEmbed ([#869](https://github.com/embeddings-benchmark/mteb/pull/869),
  [`db8bb5e`](https://github.com/embeddings-benchmark/mteb/commit/db8bb5ecaa6f5038cb090aba224461ab20015741))


## v1.12.12 (2024-06-03)

### Bug Fixes

- Updated the revision of dataset ([#866](https://github.com/embeddings-benchmark/mteb/pull/866),
  [`e48eef9`](https://github.com/embeddings-benchmark/mteb/commit/e48eef9457b246adaba5e16064c425374d78dad8))


## v1.12.11 (2024-06-02)

### Bug Fixes

- Convert MLSUM to fast ([#865](https://github.com/embeddings-benchmark/mteb/pull/865),
  [`b70bb5a`](https://github.com/embeddings-benchmark/mteb/commit/b70bb5afd808b7a46094029ab07604a08fbd7356))

### Documentation

- Removed spacing is github user name
  ([`facdb76`](https://github.com/embeddings-benchmark/mteb/commit/facdb7653cd14e92dfe853da6054e8fa0f7247bb))


## v1.12.10 (2024-06-02)

### Bug Fixes

- Add check_label_distribution for ClusteringFast
  ([#862](https://github.com/embeddings-benchmark/mteb/pull/862),
  [`6afb8a9`](https://github.com/embeddings-benchmark/mteb/commit/6afb8a9b923e59a32affd6ca3671d257387ce086))


## v1.12.9 (2024-06-02)

### Bug Fixes

- Add model implementations and script for running the models
  ([#845](https://github.com/embeddings-benchmark/mteb/pull/845),
  [`b331c34`](https://github.com/embeddings-benchmark/mteb/commit/b331c340e8f48e5530c9c71003b252769a55cacb))

- Added models implementations and script for running the models
  ([#845](https://github.com/embeddings-benchmark/mteb/pull/845),
  [`b331c34`](https://github.com/embeddings-benchmark/mteb/commit/b331c340e8f48e5530c9c71003b252769a55cacb))


## v1.12.8 (2024-06-02)

### Bug Fixes

- Formatted
  ([`8d3fc1b`](https://github.com/embeddings-benchmark/mteb/commit/8d3fc1b0c6efa8e6cd31db6ab14884b080fb53ab))


## v1.12.7 (2024-06-02)

### Bug Fixes

- Add Russian tasks (RU-MTEB) ([#815](https://github.com/embeddings-benchmark/mteb/pull/815),
  [`e9d61bb`](https://github.com/embeddings-benchmark/mteb/commit/e9d61bba729c0c50f4baa0aec3280ed94b116a96))


## v1.12.6 (2024-06-01)

### Bug Fixes

- Use model revision in results folder
  ([#842](https://github.com/embeddings-benchmark/mteb/pull/842),
  [`2c6065b`](https://github.com/embeddings-benchmark/mteb/commit/2c6065b28e5212deecc6af973ca97d0c56d16264))


## v1.12.5 (2024-05-29)

### Bug Fixes

- Find missing dataset revisions ([#844](https://github.com/embeddings-benchmark/mteb/pull/844),
  [`c9b4c0c`](https://github.com/embeddings-benchmark/mteb/commit/c9b4c0c425c67f6219c38761f72ab50fed356880))


## v1.12.4 (2024-05-28)

### Bug Fixes

- Add model meta to create reproducible workflow
  ([#807](https://github.com/embeddings-benchmark/mteb/pull/807),
  [`0319105`](https://github.com/embeddings-benchmark/mteb/commit/0319105734444de0626c068a3284832d96233dac))

- Updated CLI to use new task filter ([#826](https://github.com/embeddings-benchmark/mteb/pull/826),
  [`fb5fec8`](https://github.com/embeddings-benchmark/mteb/commit/fb5fec8b763c107fbcc9bdc853a64d6d8a8d0043))

### Documentation

- Added points ([#826](https://github.com/embeddings-benchmark/mteb/pull/826),
  [`fb5fec8`](https://github.com/embeddings-benchmark/mteb/commit/fb5fec8b763c107fbcc9bdc853a64d6d8a8d0043))


## v1.12.3 (2024-05-27)

### Bug Fixes

- Convert blurbs to fast for s2s and p2p
  ([#832](https://github.com/embeddings-benchmark/mteb/pull/832),
  [`a00fdba`](https://github.com/embeddings-benchmark/mteb/commit/a00fdba44d1fadd0be449b7f988dd08145cd5b87))


## v1.12.2 (2024-05-27)

### Bug Fixes

- (1) Add `StatcanDialogueDatasetRetrieval` (2) Fix `DRESModel.encode_conversations` to allow list
  of dictionaries ([#779](https://github.com/embeddings-benchmark/mteb/pull/779),
  [`7943ff0`](https://github.com/embeddings-benchmark/mteb/commit/7943ff05d4ec4d4c4da81a5a6c50fd298de907dd))


## v1.12.1 (2024-05-27)

### Bug Fixes

- Rename *fast to *v2 for clustering task
  ([#825](https://github.com/embeddings-benchmark/mteb/pull/825),
  [`365c0d4`](https://github.com/embeddings-benchmark/mteb/commit/365c0d4c67162c9d3a50fcd5c38d3dae37a96eac))


## v1.12.0 (2024-05-27)

### Features

- Replace get_tasks as default filtering.
  ([#806](https://github.com/embeddings-benchmark/mteb/pull/806),
  [`0ca3bc1`](https://github.com/embeddings-benchmark/mteb/commit/0ca3bc1219981af0c901b4918e78658453f3949b))


## v1.11.19 (2024-05-25)

### Bug Fixes

- Convert AlloProfClustering to Fast ([#822](https://github.com/embeddings-benchmark/mteb/pull/822),
  [`4cae4f9`](https://github.com/embeddings-benchmark/mteb/commit/4cae4f98a759bed66d83ba54de54ac79439acfcc))


## v1.11.18 (2024-05-25)

### Bug Fixes

- Convert HALClustering to Fast ([#817](https://github.com/embeddings-benchmark/mteb/pull/817),
  [`c7adcd8`](https://github.com/embeddings-benchmark/mteb/commit/c7adcd8769894b63ba77fe29f6d6a289826d746c))

### Documentation

- Adding contributor information ([#794](https://github.com/embeddings-benchmark/mteb/pull/794),
  [`3ba6362`](https://github.com/embeddings-benchmark/mteb/commit/3ba636209d81f1fda5cfd614e6a94ba5baf7d274))

- Typo in points for #716 ([#821](https://github.com/embeddings-benchmark/mteb/pull/821),
  [`c0b5991`](https://github.com/embeddings-benchmark/mteb/commit/c0b5991649e64bb30e755c94f1ecb8259650d84e))


## v1.11.17 (2024-05-24)

### Bug Fixes

- Convert BigPatent to Fast ([#813](https://github.com/embeddings-benchmark/mteb/pull/813),
  [`9b85380`](https://github.com/embeddings-benchmark/mteb/commit/9b85380beaf0395363b4c67c6b056f067da11024))

- Convert Biorxiv and Medrxiv clustering to fast
  ([#788](https://github.com/embeddings-benchmark/mteb/pull/788),
  [`13147aa`](https://github.com/embeddings-benchmark/mteb/commit/13147aa1ebf196007d7a6bf6c3f3d384fa55b8a2))

- Fixes MultilabelClassification eval_split
  ([#812](https://github.com/embeddings-benchmark/mteb/pull/812),
  [`9b602ad`](https://github.com/embeddings-benchmark/mteb/commit/9b602ade56d6be9018aac74a5addc8c3fd18c43a))


## v1.11.16 (2024-05-24)

### Bug Fixes

- Speed up Reranking tasks ([#793](https://github.com/embeddings-benchmark/mteb/pull/793),
  [`9707621`](https://github.com/embeddings-benchmark/mteb/commit/970762142550d8ba4b58de506b90dc45043935ea))


## v1.11.15 (2024-05-24)

### Bug Fixes

- Converted VG to AbsTaskClusteringFast
  ([#694](https://github.com/embeddings-benchmark/mteb/pull/694),
  [`ece878e`](https://github.com/embeddings-benchmark/mteb/commit/ece878eb274c4d72084d4488367e944c7db99fe1))

- Converted VG to hierarchical ([#694](https://github.com/embeddings-benchmark/mteb/pull/694),
  [`ece878e`](https://github.com/embeddings-benchmark/mteb/commit/ece878eb274c4d72084d4488367e944c7db99fe1))

- Fixed JSON in 694.jsonl ([#694](https://github.com/embeddings-benchmark/mteb/pull/694),
  [`ece878e`](https://github.com/embeddings-benchmark/mteb/commit/ece878eb274c4d72084d4488367e944c7db99fe1))

- Fixed subsampling ([#694](https://github.com/embeddings-benchmark/mteb/pull/694),
  [`ece878e`](https://github.com/embeddings-benchmark/mteb/commit/ece878eb274c4d72084d4488367e944c7db99fe1))

### Chores

- Add points
  ([`e82ab71`](https://github.com/embeddings-benchmark/mteb/commit/e82ab710af174d4bc705f68320282fa54a3fc819))


## v1.11.14 (2024-05-24)

### Bug Fixes

- Added ArXiv Hierarchical clustering (S2S and P2P)
  ([#699](https://github.com/embeddings-benchmark/mteb/pull/699),
  [`396eefa`](https://github.com/embeddings-benchmark/mteb/commit/396eefa32e99c4aff372543ae2aa88be32a34381))

- Broken dataset references ([#803](https://github.com/embeddings-benchmark/mteb/pull/803),
  [`9bbd2dd`](https://github.com/embeddings-benchmark/mteb/commit/9bbd2dd6779f36e2c75a5ae20af1f10ce613f934))

- Convert iterables to list ([#699](https://github.com/embeddings-benchmark/mteb/pull/699),
  [`396eefa`](https://github.com/embeddings-benchmark/mteb/commit/396eefa32e99c4aff372543ae2aa88be32a34381))

### Documentation

- Fixed points
  ([`774ca70`](https://github.com/embeddings-benchmark/mteb/commit/774ca70f5f51eae7b467f990bf4bc49dc771e21e))


## v1.11.13 (2024-05-23)

### Bug Fixes

- Add Xstance and ensure valid dataset paths
  ([#795](https://github.com/embeddings-benchmark/mteb/pull/795),
  [`47eb54c`](https://github.com/embeddings-benchmark/mteb/commit/47eb54c205cea7b7fe059769347a8d8126ac32e3))

- Added new dataset XStance pair classification
  ([#795](https://github.com/embeddings-benchmark/mteb/pull/795),
  [`47eb54c`](https://github.com/embeddings-benchmark/mteb/commit/47eb54c205cea7b7fe059769347a8d8126ac32e3))

- Ensure dataset paths a valid ([#795](https://github.com/embeddings-benchmark/mteb/pull/795),
  [`47eb54c`](https://github.com/embeddings-benchmark/mteb/commit/47eb54c205cea7b7fe059769347a8d8126ac32e3))

- GPT4-o generated queries for 14 languages
  ([#718](https://github.com/embeddings-benchmark/mteb/pull/718),
  [`411e232`](https://github.com/embeddings-benchmark/mteb/commit/411e232e3f696d0c3cb2921cb3e51b53d52d331c))


## v1.11.12 (2024-05-22)

### Bug Fixes

- Update CO2 tracker attribute ([#791](https://github.com/embeddings-benchmark/mteb/pull/791),
  [`bef9508`](https://github.com/embeddings-benchmark/mteb/commit/bef9508d60cb23d955b113413b167e5db9ee8e4b))

### Chores

- Add points ([#791](https://github.com/embeddings-benchmark/mteb/pull/791),
  [`bef9508`](https://github.com/embeddings-benchmark/mteb/commit/bef9508d60cb23d955b113413b167e5db9ee8e4b))


## v1.11.11 (2024-05-22)

### Bug Fixes

- Convert Polish cluster to fast ([#787](https://github.com/embeddings-benchmark/mteb/pull/787),
  [`a1fa96e`](https://github.com/embeddings-benchmark/mteb/commit/a1fa96eff6fe7a15c26f0db9876b1d2277d68138))


## v1.11.10 (2024-05-22)

### Bug Fixes

- Select language in multilingual tasks
  ([#789](https://github.com/embeddings-benchmark/mteb/pull/789),
  [`15ef9af`](https://github.com/embeddings-benchmark/mteb/commit/15ef9afb0c48edbda74baedf36c09d84a50de12f))


## v1.11.9 (2024-05-22)

### Bug Fixes

- Remove duplicate subset of flores clustering
  ([#785](https://github.com/embeddings-benchmark/mteb/pull/785),
  [`d5977b8`](https://github.com/embeddings-benchmark/mteb/commit/d5977b89b62ba118db20dff8b83a9f54d8961ebf))


## v1.11.8 (2024-05-22)

### Bug Fixes

- Added metadata to a bunch of datasets
  ([#781](https://github.com/embeddings-benchmark/mteb/pull/781),
  [`f38bdac`](https://github.com/embeddings-benchmark/mteb/commit/f38bdac87c8ad000fb567b1e913766e6087e1573))


## v1.11.7 (2024-05-22)

### Bug Fixes

- Restructering retrieval tasks ([#777](https://github.com/embeddings-benchmark/mteb/pull/777),
  [`ac95521`](https://github.com/embeddings-benchmark/mteb/commit/ac955216b9bf61ab2d1e0ee6560923586ad84ea7))

- Restructure folders ([#777](https://github.com/embeddings-benchmark/mteb/pull/777),
  [`ac95521`](https://github.com/embeddings-benchmark/mteb/commit/ac955216b9bf61ab2d1e0ee6560923586ad84ea7))

### Documentation

- Remove duplicate info ([#780](https://github.com/embeddings-benchmark/mteb/pull/780),
  [`706b602`](https://github.com/embeddings-benchmark/mteb/commit/706b602c16fd7ccaea8a601c68d50b1351322b8a))

- Remove duplicate name ([#780](https://github.com/embeddings-benchmark/mteb/pull/780),
  [`706b602`](https://github.com/embeddings-benchmark/mteb/commit/706b602c16fd7ccaea8a601c68d50b1351322b8a))


## v1.11.6 (2024-05-21)

### Bug Fixes

- Metadata for Polish STS ([#758](https://github.com/embeddings-benchmark/mteb/pull/758),
  [`e16807e`](https://github.com/embeddings-benchmark/mteb/commit/e16807e43382feb1bc5248bbbbdf9833cb7b035a))

### Documentation

- Add missing points ([#778](https://github.com/embeddings-benchmark/mteb/pull/778),
  [`24a9d45`](https://github.com/embeddings-benchmark/mteb/commit/24a9d45a1eafad2a6fec2a73a1f7a8d03c14fb61))


## v1.11.5 (2024-05-21)

### Bug Fixes

- Add AfriSenti dataset ([#512](https://github.com/embeddings-benchmark/mteb/pull/512),
  [`e59e7cc`](https://github.com/embeddings-benchmark/mteb/commit/e59e7cc2fbec7f6345d2fa816c8ce1a9f5afd5d4))

- Add LegalBench datasets - 11 ([#661](https://github.com/embeddings-benchmark/mteb/pull/661),
  [`c3acf03`](https://github.com/embeddings-benchmark/mteb/commit/c3acf039b3d3c373c694c844699bee9e4b658d11))

- Add LegalBench datasets - 12 ([#665](https://github.com/embeddings-benchmark/mteb/pull/665),
  [`088589f`](https://github.com/embeddings-benchmark/mteb/commit/088589f0dd5d35deb97c94645e48cd62581c55c9))

- Add turkic datasets ([#588](https://github.com/embeddings-benchmark/mteb/pull/588),
  [`cdc41a6`](https://github.com/embeddings-benchmark/mteb/commit/cdc41a6a0d9764cd9512b4d7590500683eacc661))

- Added Indic NLP News Classification tasks
  ([#610](https://github.com/embeddings-benchmark/mteb/pull/610),
  [`1a08f76`](https://github.com/embeddings-benchmark/mteb/commit/1a08f760e754c661996d6030fa0d757944ae0c3e))

- Added new dataset: TenKGnadClassification
  ([#735](https://github.com/embeddings-benchmark/mteb/pull/735),
  [`7fdae83`](https://github.com/embeddings-benchmark/mteb/commit/7fdae839018d29023f510e6e723689b39bdb6fe8))

- Convert TwentyNewsgroups to ClusteringFast
  ([#772](https://github.com/embeddings-benchmark/mteb/pull/772),
  [`74cead0`](https://github.com/embeddings-benchmark/mteb/commit/74cead062ab552ed89194933ae46304b1161c6c7))


## v1.11.4 (2024-05-21)

### Bug Fixes

- Added option to load historic result files
  ([#770](https://github.com/embeddings-benchmark/mteb/pull/770),
  [`f6d8d65`](https://github.com/embeddings-benchmark/mteb/commit/f6d8d65f090ccdb35816b07b91d83e62637c1419))

### Features

- Standardized results output ([#770](https://github.com/embeddings-benchmark/mteb/pull/770),
  [`f6d8d65`](https://github.com/embeddings-benchmark/mteb/commit/f6d8d65f090ccdb35816b07b91d83e62637c1419))


## v1.11.3 (2024-05-21)

### Bug Fixes

- Restructering retrieval tasks ([#776](https://github.com/embeddings-benchmark/mteb/pull/776),
  [`4b71792`](https://github.com/embeddings-benchmark/mteb/commit/4b71792c80914e5dd7e92fb1ef0e629c4baf996c))


## v1.11.2 (2024-05-21)

### Bug Fixes

- Remove v_measures from metadata ([#775](https://github.com/embeddings-benchmark/mteb/pull/775),
  [`56a5e10`](https://github.com/embeddings-benchmark/mteb/commit/56a5e10db4dc16e943a26da0de464986ecad77a9))


## v1.11.1 (2024-05-21)

### Bug Fixes

- Convert StackExchangeClustering to fast
  ([#740](https://github.com/embeddings-benchmark/mteb/pull/740),
  [`7c23ac9`](https://github.com/embeddings-benchmark/mteb/commit/7c23ac936be36ca09e07e8156cbb9a0bbcf9f2de))

- Queries has been converted from dict to list by this
  ([`140de21`](https://github.com/embeddings-benchmark/mteb/commit/140de2139880e508e1697d782bbbb4a101e18c1f))


## v1.11.0 (2024-05-20)

### Features

- Standardized results output ([#759](https://github.com/embeddings-benchmark/mteb/pull/759),
  [`1029a27`](https://github.com/embeddings-benchmark/mteb/commit/1029a271418e54cd48231cd77aaad90908da2b11))


## v1.10.18 (2024-05-20)

### Bug Fixes

- Add JMTEB Clustering datasets ([#768](https://github.com/embeddings-benchmark/mteb/pull/768),
  [`970b03c`](https://github.com/embeddings-benchmark/mteb/commit/970b03ca7fd1898b4a5e47fea775fa72b6f0f668))


## v1.10.17 (2024-05-20)

### Bug Fixes

- Add JSICK dataset ([#769](https://github.com/embeddings-benchmark/mteb/pull/769),
  [`755fe35`](https://github.com/embeddings-benchmark/mteb/commit/755fe35c3a145049b38e3ee145dcc6964786e164))


## v1.10.16 (2024-05-20)

### Bug Fixes

- Convert SIB200Class to SIB200Cluster
  ([#767](https://github.com/embeddings-benchmark/mteb/pull/767),
  [`211f339`](https://github.com/embeddings-benchmark/mteb/commit/211f33985dae80f6102b830fad8cd2655e94cb76))


## v1.10.15 (2024-05-19)

### Bug Fixes

- Add `bibtex_citation` in CMTEBClustering
  ([#765](https://github.com/embeddings-benchmark/mteb/pull/765),
  [`bf6e2a0`](https://github.com/embeddings-benchmark/mteb/commit/bf6e2a0aee2d87f2e91cd5b7b6ec1d447c02cff2))


## v1.10.14 (2024-05-19)

### Bug Fixes

- Convert CLSClustering and ThuNews to fast
  ([#757](https://github.com/embeddings-benchmark/mteb/pull/757),
  [`43e0246`](https://github.com/embeddings-benchmark/mteb/commit/43e0246aa3a2b6417876c934912f71ae71eeb74f))


## v1.10.13 (2024-05-18)

### Bug Fixes

- Add MalteseNewsClassification ([#546](https://github.com/embeddings-benchmark/mteb/pull/546),
  [`5314bf5`](https://github.com/embeddings-benchmark/mteb/commit/5314bf5729409e78653021f8cbf04df78068903c))

### Continuous Integration

- Fix slovaksum dir ([#764](https://github.com/embeddings-benchmark/mteb/pull/764),
  [`1c14edb`](https://github.com/embeddings-benchmark/mteb/commit/1c14edb9b0fabc5bd0f3ab12b027151e2cc864e4))


## v1.10.12 (2024-05-18)

### Bug Fixes

- Add PublicHealthQA ([#750](https://github.com/embeddings-benchmark/mteb/pull/750),
  [`ca7266e`](https://github.com/embeddings-benchmark/mteb/commit/ca7266e6c81d8155a42d3532b725126c9c896a94))


## v1.10.11 (2024-05-18)

### Bug Fixes

- Quick fix CI/CD issue due to dataset redirect
  ([#761](https://github.com/embeddings-benchmark/mteb/pull/761),
  [`495d6f6`](https://github.com/embeddings-benchmark/mteb/commit/495d6f68efd4a4da7adb5dc8b4f5de6557c0c7e3))


## v1.10.10 (2024-05-17)

### Bug Fixes

- Convert Multilingual/Crosslingual to fast-loading format
  ([#635](https://github.com/embeddings-benchmark/mteb/pull/635),
  [`aa82ada`](https://github.com/embeddings-benchmark/mteb/commit/aa82ada7b29826df0104193ddcc98cf7bb76c884))


## v1.10.9 (2024-05-17)

### Bug Fixes

- Convert WikiClustering to fast ([#745](https://github.com/embeddings-benchmark/mteb/pull/745),
  [`153827b`](https://github.com/embeddings-benchmark/mteb/commit/153827b022980ce9f454d1ef622f78e5856b0a77))


## v1.10.8 (2024-05-17)

### Bug Fixes

- Add new dataset: GermanGovServiceRetrieval
  ([#731](https://github.com/embeddings-benchmark/mteb/pull/731),
  [`66792ef`](https://github.com/embeddings-benchmark/mteb/commit/66792efbfc12cc43df87ab17684f5202775ff253))


## v1.10.7 (2024-05-17)

### Bug Fixes

- Convert Reddit cluster s2s and p2p to fast
  ([#729](https://github.com/embeddings-benchmark/mteb/pull/729),
  [`b02f252`](https://github.com/embeddings-benchmark/mteb/commit/b02f252c707bce6de61af56ca3b5d7e921f6ef09))


## v1.10.6 (2024-05-17)

### Bug Fixes

- RomanianReviewsSentiment moved to new repo
  ([#751](https://github.com/embeddings-benchmark/mteb/pull/751),
  [`b636c23`](https://github.com/embeddings-benchmark/mteb/commit/b636c23f4a32d4c7929eeccda5e4afc8454062ca))


## v1.10.5 (2024-05-16)

### Bug Fixes

- MindSmallReranking not loading ([#748](https://github.com/embeddings-benchmark/mteb/pull/748),
  [`a5c4f5b`](https://github.com/embeddings-benchmark/mteb/commit/a5c4f5bc03e38c416fc4b6ed4317058bfa73ffd2))


## v1.10.4 (2024-05-16)

### Bug Fixes

- Add RUParaPhraserSTS ([#716](https://github.com/embeddings-benchmark/mteb/pull/716),
  [`5b13b5c`](https://github.com/embeddings-benchmark/mteb/commit/5b13b5c36022e28d6b6e0ce8e7dc21ce61fb67d8))


## v1.10.3 (2024-05-16)

### Bug Fixes

- Add IndicGenBenchFlores Dataset ([#733](https://github.com/embeddings-benchmark/mteb/pull/733),
  [`84dcd3c`](https://github.com/embeddings-benchmark/mteb/commit/84dcd3c0fc77875e6caed755a3bccbc0ddce8b20))


## v1.10.2 (2024-05-16)

### Bug Fixes

- Add LegalBench datasets - 13 ([#678](https://github.com/embeddings-benchmark/mteb/pull/678),
  [`5032bf8`](https://github.com/embeddings-benchmark/mteb/commit/5032bf8a27553e1bc703ba9b9458fe809c970a86))

- Add LegalBench datasets - 14 ([#680](https://github.com/embeddings-benchmark/mteb/pull/680),
  [`59d89e0`](https://github.com/embeddings-benchmark/mteb/commit/59d89e000c87a8dcde4a4fd6c946e0a75dd75474))

- IWSLT2017 ([#727](https://github.com/embeddings-benchmark/mteb/pull/727),
  [`5c9468c`](https://github.com/embeddings-benchmark/mteb/commit/5c9468c01168fc0c6ac6898fc11ff718ddc94830))

- N_samples for ces and svk classification
  ([`40e4ccf`](https://github.com/embeddings-benchmark/mteb/commit/40e4ccf905ffac9a89021eaf2bfa5fcbb299c91e))

- Proper task_subtype, take the string label
  ([`40e4ccf`](https://github.com/embeddings-benchmark/mteb/commit/40e4ccf905ffac9a89021eaf2bfa5fcbb299c91e))

### Chores

- Fix command in PR template ([#700](https://github.com/embeddings-benchmark/mteb/pull/700),
  [`6dc9730`](https://github.com/embeddings-benchmark/mteb/commit/6dc973093589433f96d981bdca72374a4a92d163))

### Documentation

- Add Task Count table per language ([#701](https://github.com/embeddings-benchmark/mteb/pull/701),
  [`d9deea0`](https://github.com/embeddings-benchmark/mteb/commit/d9deea0b8de5b16890fbbd5602479593862fec9a))


## v1.10.1 (2024-05-15)

### Bug Fixes

- Adding ArEntail dataset ([#676](https://github.com/embeddings-benchmark/mteb/pull/676),
  [`06347a6`](https://github.com/embeddings-benchmark/mteb/commit/06347a6c4c815ef47e9575c8f46518d5a7f2c892))


## v1.10.0 (2024-05-14)

### Bug Fixes

- Add LegalBench datasets - 17 ([#693](https://github.com/embeddings-benchmark/mteb/pull/693),
  [`83ed1ee`](https://github.com/embeddings-benchmark/mteb/commit/83ed1ee810c23e32ee607a845878495dfe601ba1))

- Make tracker optional ([#712](https://github.com/embeddings-benchmark/mteb/pull/712),
  [`2123473`](https://github.com/embeddings-benchmark/mteb/commit/2123473c3ab1565825ffaa3f652b6c9002233161))

- Making package import optional ([#712](https://github.com/embeddings-benchmark/mteb/pull/712),
  [`2123473`](https://github.com/embeddings-benchmark/mteb/commit/2123473c3ab1565825ffaa3f652b6c9002233161))

### Chores

- Adding points ([#712](https://github.com/embeddings-benchmark/mteb/pull/712),
  [`2123473`](https://github.com/embeddings-benchmark/mteb/commit/2123473c3ab1565825ffaa3f652b6c9002233161))

- Fix package install ([#712](https://github.com/embeddings-benchmark/mteb/pull/712),
  [`2123473`](https://github.com/embeddings-benchmark/mteb/commit/2123473c3ab1565825ffaa3f652b6c9002233161))

### Features

- Add carbon emissions estimation ([#712](https://github.com/embeddings-benchmark/mteb/pull/712),
  [`2123473`](https://github.com/embeddings-benchmark/mteb/commit/2123473c3ab1565825ffaa3f652b6c9002233161))


## v1.9.3 (2024-05-14)

### Bug Fixes

- Add LegalBench datasets - 15 ([#682](https://github.com/embeddings-benchmark/mteb/pull/682),
  [`6fbe071`](https://github.com/embeddings-benchmark/mteb/commit/6fbe071fec806866857ef5ee898941d6f4749726))


## v1.9.2 (2024-05-14)

### Bug Fixes

- Add supply chain tasks of LegalBench
  ([#690](https://github.com/embeddings-benchmark/mteb/pull/690),
  [`d9f1a55`](https://github.com/embeddings-benchmark/mteb/commit/d9f1a5501b333251d28bebcc2fcfafd686ba8457))


## v1.9.1 (2024-05-14)

### Bug Fixes

- Clustering model is now initiated at each level
  ([`74a19a7`](https://github.com/embeddings-benchmark/mteb/commit/74a19a7a534865d89bc830afccc616f376327d29))

- Corrected number of samples and mean length in SNLHierarchicalClustering
  ([`74a19a7`](https://github.com/embeddings-benchmark/mteb/commit/74a19a7a534865d89bc830afccc616f376327d29))

- Double assignemnt in RomanianReviewsSentiment
  ([#692](https://github.com/embeddings-benchmark/mteb/pull/692),
  [`b4155b5`](https://github.com/embeddings-benchmark/mteb/commit/b4155b555ddf4f77ecab45775f6b3c0289bd4bb9))

- Fixed indentation in clustering fast
  ([`74a19a7`](https://github.com/embeddings-benchmark/mteb/commit/74a19a7a534865d89bc830afccc616f376327d29))

- Ran linting and fixed metadata
  ([`74a19a7`](https://github.com/embeddings-benchmark/mteb/commit/74a19a7a534865d89bc830afccc616f376327d29))

### Chores

- Add openreview username ([#683](https://github.com/embeddings-benchmark/mteb/pull/683),
  [`93a1248`](https://github.com/embeddings-benchmark/mteb/commit/93a1248559d23efb49d572d28ae92a91a81511ae))

- Adding myself as a contributor ([#683](https://github.com/embeddings-benchmark/mteb/pull/683),
  [`93a1248`](https://github.com/embeddings-benchmark/mteb/commit/93a1248559d23efb49d572d28ae92a91a81511ae))

- Update results ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))

### Documentation

- Add contribution ([#688](https://github.com/embeddings-benchmark/mteb/pull/688),
  [`e3d230a`](https://github.com/embeddings-benchmark/mteb/commit/e3d230a5a4da54d85314d40c0643ffe0c0ac02de))

- Adding contributor information ([#683](https://github.com/embeddings-benchmark/mteb/pull/683),
  [`93a1248`](https://github.com/embeddings-benchmark/mteb/commit/93a1248559d23efb49d572d28ae92a91a81511ae))

- Adjust description ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))

- Update points system ([#666](https://github.com/embeddings-benchmark/mteb/pull/666),
  [`d731c98`](https://github.com/embeddings-benchmark/mteb/commit/d731c98ed38ae1299acd27495adfade37c068dd3))

- Update points.md ([#687](https://github.com/embeddings-benchmark/mteb/pull/687),
  [`85c7858`](https://github.com/embeddings-benchmark/mteb/commit/85c7858ceb47acbf943bb8b96a6c845fb0affe7b))

### Features

- Belebele retrieval ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))

- Support langs ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))

### Refactoring

- Apply suggestions ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))

- Change num_samples and remove answers
  ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))

- Update avg length ([#636](https://github.com/embeddings-benchmark/mteb/pull/636),
  [`4a2b9db`](https://github.com/embeddings-benchmark/mteb/commit/4a2b9db43987f26df77c940f25e980bf459144c4))


## v1.9.0 (2024-05-13)

### Bug Fixes

- Refactored out get_main_score ([#658](https://github.com/embeddings-benchmark/mteb/pull/658),
  [`7166c31`](https://github.com/embeddings-benchmark/mteb/commit/7166c317c1748b4b11772fe59b8410d5f53aa0f0))

- Reformatted according to wishes in PR
  ([#658](https://github.com/embeddings-benchmark/mteb/pull/658),
  [`7166c31`](https://github.com/embeddings-benchmark/mteb/commit/7166c317c1748b4b11772fe59b8410d5f53aa0f0))

- Two Korean classification datasets added
  ([#670](https://github.com/embeddings-benchmark/mteb/pull/670),
  [`7c2299c`](https://github.com/embeddings-benchmark/mteb/commit/7c2299c9d0b41fb00978b97807bcbb4d946ef105))

### Documentation

- Remove prompt kwargs for example ([#681](https://github.com/embeddings-benchmark/mteb/pull/681),
  [`df490cf`](https://github.com/embeddings-benchmark/mteb/commit/df490cfc16164c2b63e46c8cfe25b01a84e1153d))

- Remove prompt kwargs for example
  ([`43f7157`](https://github.com/embeddings-benchmark/mteb/commit/43f7157235023e46fa36060c73c64ceee49c8d36))

### Features

- Standardize MTEB results ([#658](https://github.com/embeddings-benchmark/mteb/pull/658),
  [`7166c31`](https://github.com/embeddings-benchmark/mteb/commit/7166c317c1748b4b11772fe59b8410d5f53aa0f0))


## v1.8.11 (2024-05-12)

### Bug Fixes

- Add cyrillic turkic lang classification
  ([#659](https://github.com/embeddings-benchmark/mteb/pull/659),
  [`11b4888`](https://github.com/embeddings-benchmark/mteb/commit/11b4888aba0709b071512a5a042ad1d1043c06d1))


## v1.8.10 (2024-05-12)

### Bug Fixes

- Ensure that task.languages return a list of languages
  ([#671](https://github.com/embeddings-benchmark/mteb/pull/671),
  [`2a8f8f5`](https://github.com/embeddings-benchmark/mteb/commit/2a8f8f58b14f071df9c05fe6b861d76e86e899c5))

### Documentation

- Added points ([#671](https://github.com/embeddings-benchmark/mteb/pull/671),
  [`2a8f8f5`](https://github.com/embeddings-benchmark/mteb/commit/2a8f8f58b14f071df9c05fe6b861d76e86e899c5))


## v1.8.9 (2024-05-11)

### Bug Fixes

- Add Marathi news classification ([#504](https://github.com/embeddings-benchmark/mteb/pull/504),
  [`d93488a`](https://github.com/embeddings-benchmark/mteb/commit/d93488a6fe461eb00e3deb47397908af9a467805))

- Changed itertools.chain to itertools.chain.from_iter
  ([`2aa0c67`](https://github.com/embeddings-benchmark/mteb/commit/2aa0c67b05acd9dadb9b1731f8a8bb28de58702f))

- Fixed undersampling for training set in Multitask classification
  ([`2aa0c67`](https://github.com/embeddings-benchmark/mteb/commit/2aa0c67b05acd9dadb9b1731f8a8bb28de58702f))

- Fixed validation and import on MultiEURLEX
  ([`2aa0c67`](https://github.com/embeddings-benchmark/mteb/commit/2aa0c67b05acd9dadb9b1731f8a8bb28de58702f))

- Multilabels are not turned into an array
  ([`2aa0c67`](https://github.com/embeddings-benchmark/mteb/commit/2aa0c67b05acd9dadb9b1731f8a8bb28de58702f))

- Removed duplicate code for selecting train sentences
  ([`2aa0c67`](https://github.com/embeddings-benchmark/mteb/commit/2aa0c67b05acd9dadb9b1731f8a8bb28de58702f))

- Sped up sampling by using select() instead of indexing
  ([`2aa0c67`](https://github.com/embeddings-benchmark/mteb/commit/2aa0c67b05acd9dadb9b1731f8a8bb28de58702f))


## v1.8.8 (2024-05-11)

### Bug Fixes

- Add LegalBench datasets - 8 ([#648](https://github.com/embeddings-benchmark/mteb/pull/648),
  [`10a0354`](https://github.com/embeddings-benchmark/mteb/commit/10a03544b9cd7b6839bc3c725dcddea27f2cdccf))

- Mmteb | Arabic Retrieval Task | SadeemQuestionRetrieval
  ([#643](https://github.com/embeddings-benchmark/mteb/pull/643),
  [`f2d6c1a`](https://github.com/embeddings-benchmark/mteb/commit/f2d6c1a6cb6cb13edffad6a7735148da806e972d))


## v1.8.7 (2024-05-09)

### Bug Fixes

- Added Tswana News Classification dataset and eval results
  ([#653](https://github.com/embeddings-benchmark/mteb/pull/653),
  [`89f671f`](https://github.com/embeddings-benchmark/mteb/commit/89f671f43c6cc0f5963f15f0b52ec75b66ee545f))


## v1.8.6 (2024-05-08)

### Bug Fixes

- Swiss judgement classification dataset
  ([#569](https://github.com/embeddings-benchmark/mteb/pull/569),
  [`2d69957`](https://github.com/embeddings-benchmark/mteb/commit/2d69957b0f4a72ed60f755d73c429aa622c513e6))


## v1.8.5 (2024-05-08)

### Bug Fixes

- Added Crosslingual Semantic Discrimination Task with Four Evaluation Pairs
  ([#645](https://github.com/embeddings-benchmark/mteb/pull/645),
  [`542ee28`](https://github.com/embeddings-benchmark/mteb/commit/542ee28fdc9b8e9ef8819eb17282c46d76368659))


## v1.8.4 (2024-05-08)

### Bug Fixes

- Making ScalaClassification multilingual
  ([#606](https://github.com/embeddings-benchmark/mteb/pull/606),
  [`0be4b1f`](https://github.com/embeddings-benchmark/mteb/commit/0be4b1f54b6200625e9721366bd89e8873864294))

### Continuous Integration

- Added repeat to windows test to hopefully alleviate the ci issues
  ([#646](https://github.com/embeddings-benchmark/mteb/pull/646),
  [`d6ef5b6`](https://github.com/embeddings-benchmark/mteb/commit/d6ef5b63ebfb4bd8c8478385d3dc842373d67375))


## v1.8.3 (2024-05-07)

### Bug Fixes

- Add LegalBench datasets - 7 ([#644](https://github.com/embeddings-benchmark/mteb/pull/644),
  [`dcb4c4c`](https://github.com/embeddings-benchmark/mteb/commit/dcb4c4cb6b2080f3ec388deb610266d4a152f343))


## v1.8.2 (2024-05-06)

### Bug Fixes

- Add IndonesianMongabay classification dataset
  ([#634](https://github.com/embeddings-benchmark/mteb/pull/634),
  [`f20831f`](https://github.com/embeddings-benchmark/mteb/commit/f20831f17b1a0909919c4fd07dc1c84902b51f23))


## v1.8.1 (2024-05-06)

### Bug Fixes

- XNLI 2.0 subset added ([#623](https://github.com/embeddings-benchmark/mteb/pull/623),
  [`f62b830`](https://github.com/embeddings-benchmark/mteb/commit/f62b83047651d94fd0bb1187e40141bfc65741b8))


## v1.8.0 (2024-05-05)

### Bug Fixes

- Update typing ([#560](https://github.com/embeddings-benchmark/mteb/pull/560),
  [`7f48327`](https://github.com/embeddings-benchmark/mteb/commit/7f4832772fa693368b07d4c3d19a672ab7a83bad))

### Chores

- Add comments for corpus and query lang
  ([#560](https://github.com/embeddings-benchmark/mteb/pull/560),
  [`7f48327`](https://github.com/embeddings-benchmark/mteb/commit/7f4832772fa693368b07d4c3d19a672ab7a83bad))

### Features

- Add MLQA dataset ([#560](https://github.com/embeddings-benchmark/mteb/pull/560),
  [`7f48327`](https://github.com/embeddings-benchmark/mteb/commit/7f4832772fa693368b07d4c3d19a672ab7a83bad))

- Add MLQA dataset for CrossLingual Retrieval
  ([#560](https://github.com/embeddings-benchmark/mteb/pull/560),
  [`7f48327`](https://github.com/embeddings-benchmark/mteb/commit/7f4832772fa693368b07d4c3d19a672ab7a83bad))


## v1.7.64 (2024-05-05)

### Bug Fixes

- Add JavaneseIMDBClassification ([#632](https://github.com/embeddings-benchmark/mteb/pull/632),
  [`73aa304`](https://github.com/embeddings-benchmark/mteb/commit/73aa304b6d33eb6788fcf15e2138cdd995ac851c))


## v1.7.63 (2024-05-05)

### Bug Fixes

- Add FilipinoShopeeReviewsClassification
  ([#630](https://github.com/embeddings-benchmark/mteb/pull/630),
  [`cc7582f`](https://github.com/embeddings-benchmark/mteb/commit/cc7582fcf17144b15ae23a3def702d54788cf13a))


## v1.7.62 (2024-05-05)

### Bug Fixes

- CataloniaTweetClassification dataset added
  ([#629](https://github.com/embeddings-benchmark/mteb/pull/629),
  [`fe48afe`](https://github.com/embeddings-benchmark/mteb/commit/fe48afe166da101d73ce5224e9cbd1cfe253cd45))


## v1.7.61 (2024-05-05)

### Bug Fixes

- 2 datasets for Georgian language ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

- Add avg length and license ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

- Inherit crosslinguality ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

- Language code ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

### Chores

- Add points ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

- Add some metadata ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

- Flaky ci? ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

### Code Style

- Make lint ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

### Features

- Georgian faq dataset ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

- Tbilisi city hall dataset ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))

### Refactoring

- Fill metadata ([#633](https://github.com/embeddings-benchmark/mteb/pull/633),
  [`1dfae84`](https://github.com/embeddings-benchmark/mteb/commit/1dfae840c8876fe7eb6be204dfa09bf15e00098b))


## v1.7.60 (2024-05-04)

### Bug Fixes

- Add Frenk hr dataset ([#628](https://github.com/embeddings-benchmark/mteb/pull/628),
  [`0957eb4`](https://github.com/embeddings-benchmark/mteb/commit/0957eb461801c709c59acabdae22bd9cf3ec91dc))


## v1.7.59 (2024-05-04)

### Bug Fixes

- Add Frenk en dataset ([#627](https://github.com/embeddings-benchmark/mteb/pull/627),
  [`7dd0f16`](https://github.com/embeddings-benchmark/mteb/commit/7dd0f165279c0b52149d7377f7a9a5e863e89dbe))

### Documentation

- Add info for LongEmbed ([#625](https://github.com/embeddings-benchmark/mteb/pull/625),
  [`531cbe0`](https://github.com/embeddings-benchmark/mteb/commit/531cbe030a09208b336bebba3e4b3df78c03ec8d))


## v1.7.58 (2024-05-02)

### Bug Fixes

- Add LegalBench datasets - 6 ([#622](https://github.com/embeddings-benchmark/mteb/pull/622),
  [`51adc5f`](https://github.com/embeddings-benchmark/mteb/commit/51adc5f5caf0fe83a8b024257f36a7507580d760))

- Add new Polish clustering tasks (PL-MTEB)
  ([#607](https://github.com/embeddings-benchmark/mteb/pull/607),
  [`25087de`](https://github.com/embeddings-benchmark/mteb/commit/25087de051933acc23a6c9fba0612d79f38484a4))

### Continuous Integration

- Only attempt push when on main branch
  ([`c64c189`](https://github.com/embeddings-benchmark/mteb/commit/c64c1898213c615aa309b307aed21fb6ec22c000))


## v1.7.57 (2024-05-02)

### Bug Fixes

- Add LegalBench datasets - 5 ([#613](https://github.com/embeddings-benchmark/mteb/pull/613),
  [`87568f6`](https://github.com/embeddings-benchmark/mteb/commit/87568f695fd408d76efa91b6dc8492e70fa2cde3))

- Gather label_feature
  ([`ff6f08d`](https://github.com/embeddings-benchmark/mteb/commit/ff6f08dfe67b643e251db5bf10b8437be54a521f))


## v1.7.56 (2024-05-02)

### Bug Fixes

- Fix Docs CI Workflow ([#621](https://github.com/embeddings-benchmark/mteb/pull/621),
  [`778355d`](https://github.com/embeddings-benchmark/mteb/commit/778355db702d1759bcb8ed67f78035908e3534db))


## v1.7.55 (2024-05-02)

### Bug Fixes

- BSARDRetrieval dataset ([#615](https://github.com/embeddings-benchmark/mteb/pull/615),
  [`3c32c40`](https://github.com/embeddings-benchmark/mteb/commit/3c32c40c6231adddb072649f04dd88b747e31bbd))


## v1.7.54 (2024-05-02)

### Bug Fixes

- Add Hindi dialect classification ([#616](https://github.com/embeddings-benchmark/mteb/pull/616),
  [`9939715`](https://github.com/embeddings-benchmark/mteb/commit/9939715c0ca2ea7621d05b2d04a485839582f3a4))


## v1.7.53 (2024-05-02)

### Bug Fixes

- CodeEditSearch: Instruction -> Diff retrieval
  ([#594](https://github.com/embeddings-benchmark/mteb/pull/594),
  [`a9aa486`](https://github.com/embeddings-benchmark/mteb/commit/a9aa486e09cb8c1db38510e6d432fa4901c5f11a))

### Documentation

- Add points for LegalBench datasets - 4 (#611)
  ([#612](https://github.com/embeddings-benchmark/mteb/pull/612),
  [`fbce958`](https://github.com/embeddings-benchmark/mteb/commit/fbce958171edc12f6b27a9401f2dd4454754d054))


## v1.7.52 (2024-05-01)

### Bug Fixes

- Add LegalBench datasets - 4 ([#611](https://github.com/embeddings-benchmark/mteb/pull/611),
  [`ad314cf`](https://github.com/embeddings-benchmark/mteb/commit/ad314cf9c717ac490efdee4a73322de26da01c03))


## v1.7.51 (2024-05-01)

### Bug Fixes

- Add new languages to NTREX ([#543](https://github.com/embeddings-benchmark/mteb/pull/543),
  [`809e1ee`](https://github.com/embeddings-benchmark/mteb/commit/809e1ee0a4ba016c6ae35c925f55aa737a9224a8))


## v1.7.50 (2024-04-30)

### Bug Fixes

- Fast loading for cross lingual tasks
  ([#572](https://github.com/embeddings-benchmark/mteb/pull/572),
  [`aa2ffe8`](https://github.com/embeddings-benchmark/mteb/commit/aa2ffe8dad092bb333260148652cfad67cf757c7))


## v1.7.49 (2024-04-30)

### Bug Fixes

- Add linting ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))

- Add multlingual sentiment classification
  ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))

- Add sentiment classification datasets as one multilingual task
  ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))

- Remove old tasks and fix typo ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))

### Chores

- Add nb samples ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))

- Add points and remove old eval files
  ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))

- Remove useless import ([#575](https://github.com/embeddings-benchmark/mteb/pull/575),
  [`ef53f68`](https://github.com/embeddings-benchmark/mteb/commit/ef53f6898d4c184771d2f4f2b56e21361ffaa2de))


## v1.7.48 (2024-04-30)

### Bug Fixes

- Add LegalBench datasets - 3 ([#579](https://github.com/embeddings-benchmark/mteb/pull/579),
  [`07b04e6`](https://github.com/embeddings-benchmark/mteb/commit/07b04e683bdb836948a973c2e106b993d7e38dc7))


## v1.7.47 (2024-04-30)

### Bug Fixes

- TweetTopicSingleClassification added
  ([#603](https://github.com/embeddings-benchmark/mteb/pull/603),
  [`ec2579e`](https://github.com/embeddings-benchmark/mteb/commit/ec2579e0c0be10ecc4561d99ebc0372eb02cce9c))

### Documentation

- Fixed error in points
  ([`6385153`](https://github.com/embeddings-benchmark/mteb/commit/638515361ca37e2b6c684abf72c9d359cf2f2ffe))


## v1.7.46 (2024-04-29)

### Bug Fixes

- Add Multiple Sentiment/Hate Speech Classification Datasets
  ([#598](https://github.com/embeddings-benchmark/mteb/pull/598),
  [`9db1dde`](https://github.com/embeddings-benchmark/mteb/commit/9db1dde9dfd2c099a252c1fd199afd73996f8b45))


## v1.7.45 (2024-04-29)

### Bug Fixes

- Count_languages func added ([#589](https://github.com/embeddings-benchmark/mteb/pull/589),
  [`f740929`](https://github.com/embeddings-benchmark/mteb/commit/f740929329f05d2fc4893237a3d1cfe4f1e151a5))


## v1.7.44 (2024-04-29)

### Bug Fixes

- Add XNLI as multilingual pair classification
  ([#600](https://github.com/embeddings-benchmark/mteb/pull/600),
  [`7f8f14d`](https://github.com/embeddings-benchmark/mteb/commit/7f8f14d446861f700a674b7f60cb99aa050c6e78))


## v1.7.43 (2024-04-29)

### Bug Fixes

- Fix instructions in DRESModel ([#580](https://github.com/embeddings-benchmark/mteb/pull/580),
  [`a3e8b91`](https://github.com/embeddings-benchmark/mteb/commit/a3e8b91625d2bff021333158c50b13b074d014b3))


## v1.7.42 (2024-04-29)

### Bug Fixes

- SIB200Classification dataset added ([#545](https://github.com/embeddings-benchmark/mteb/pull/545),
  [`1846e73`](https://github.com/embeddings-benchmark/mteb/commit/1846e732bd802fffeafabd473e3e7354736c9af1))


## v1.7.41 (2024-04-28)

### Bug Fixes

- Add Arabic Jordanian General Tweets (AJGT) corpus
  ([#592](https://github.com/embeddings-benchmark/mteb/pull/592),
  [`41b51a9`](https://github.com/embeddings-benchmark/mteb/commit/41b51a9f66b5c03e12881bbd38d7817a413ec700))


## v1.7.40 (2024-04-28)

### Bug Fixes

- Add BibleNLP dataset ([#583](https://github.com/embeddings-benchmark/mteb/pull/583),
  [`593fc8f`](https://github.com/embeddings-benchmark/mteb/commit/593fc8fcdfc6e0c2a38c559748c332451559b3ee))


## v1.7.39 (2024-04-28)

### Bug Fixes

- Add South African Language Identification
  ([#590](https://github.com/embeddings-benchmark/mteb/pull/590),
  [`d5a9a7e`](https://github.com/embeddings-benchmark/mteb/commit/d5a9a7e3571eaf6cd53063ad0ae913532ca9ce0b))


## v1.7.38 (2024-04-27)

### Bug Fixes

- Add Odia news classification ([#503](https://github.com/embeddings-benchmark/mteb/pull/503),
  [`9dd4cd9`](https://github.com/embeddings-benchmark/mteb/commit/9dd4cd97654a10804f5b479e394e6f0fc43c32cb))


## v1.7.37 (2024-04-27)

### Bug Fixes

- Add LegalBench datasets - 2 ([#571](https://github.com/embeddings-benchmark/mteb/pull/571),
  [`202e0ed`](https://github.com/embeddings-benchmark/mteb/commit/202e0edd67813f076cf601b48f6294419eaa7aea))


## v1.7.36 (2024-04-26)

### Bug Fixes

- Languages is a sorted-list now ([#578](https://github.com/embeddings-benchmark/mteb/pull/578),
  [`adcc8c6`](https://github.com/embeddings-benchmark/mteb/commit/adcc8c6101a9ccdc8e678cbd591d2153d1d56f44))

### Refactoring

- Fix linting ([#578](https://github.com/embeddings-benchmark/mteb/pull/578),
  [`adcc8c6`](https://github.com/embeddings-benchmark/mteb/commit/adcc8c6101a9ccdc8e678cbd591d2153d1d56f44))

- Languages is a sorted-list now ([#578](https://github.com/embeddings-benchmark/mteb/pull/578),
  [`adcc8c6`](https://github.com/embeddings-benchmark/mteb/commit/adcc8c6101a9ccdc8e678cbd591d2153d1d56f44))


## v1.7.35 (2024-04-26)

### Bug Fixes

- Add GreekCivicsQA and results ([#570](https://github.com/embeddings-benchmark/mteb/pull/570),
  [`94c3b9a`](https://github.com/embeddings-benchmark/mteb/commit/94c3b9a29618d9c1cf059ae72f2e9119ea0e0273))


## v1.7.34 (2024-04-26)

### Bug Fixes

- Add DBpedia dataset ([#501](https://github.com/embeddings-benchmark/mteb/pull/501),
  [`c49de83`](https://github.com/embeddings-benchmark/mteb/commit/c49de833fc11544e3623267e87df7c50ed7104cf))


## v1.7.33 (2024-04-26)

### Bug Fixes

- Add Big Patent Classification dataset
  ([#497](https://github.com/embeddings-benchmark/mteb/pull/497),
  [`5baeb90`](https://github.com/embeddings-benchmark/mteb/commit/5baeb9009e26159e6951003e2799a69146031565))

- Don't change bibtex citation to "" for historic datasets
  ([#550](https://github.com/embeddings-benchmark/mteb/pull/550),
  [`49353d6`](https://github.com/embeddings-benchmark/mteb/commit/49353d6e8c809c75de353d74a69c116551299aa7))

- Update taskmetadata in adding a dataset example
  ([#550](https://github.com/embeddings-benchmark/mteb/pull/550),
  [`49353d6`](https://github.com/embeddings-benchmark/mteb/commit/49353d6e8c809c75de353d74a69c116551299aa7))

### Chores

- Add new datasets without filled metadata to whitelist
  ([#550](https://github.com/embeddings-benchmark/mteb/pull/550),
  [`49353d6`](https://github.com/embeddings-benchmark/mteb/commit/49353d6e8c809c75de353d74a69c116551299aa7))

- Delete unused file ([#550](https://github.com/embeddings-benchmark/mteb/pull/550),
  [`49353d6`](https://github.com/embeddings-benchmark/mteb/commit/49353d6e8c809c75de353d74a69c116551299aa7))

- Move historic datasets to test file
  ([#550](https://github.com/embeddings-benchmark/mteb/pull/550),
  [`49353d6`](https://github.com/embeddings-benchmark/mteb/commit/49353d6e8c809c75de353d74a69c116551299aa7))

- Update datasets without filled metadata
  ([#550](https://github.com/embeddings-benchmark/mteb/pull/550),
  [`49353d6`](https://github.com/embeddings-benchmark/mteb/commit/49353d6e8c809c75de353d74a69c116551299aa7))


## v1.7.32 (2024-04-25)

### Bug Fixes

- Add LegalBench datasets ([#515](https://github.com/embeddings-benchmark/mteb/pull/515),
  [`53429e1`](https://github.com/embeddings-benchmark/mteb/commit/53429e1d50327a553b1030fdfa8a889b1723dd49))


## v1.7.31 (2024-04-25)

### Bug Fixes

- Add bibtext citation ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Converted german clustering task to fast
  ([#528](https://github.com/embeddings-benchmark/mteb/pull/528),
  [`1e44b98`](https://github.com/embeddings-benchmark/mteb/commit/1e44b98535eb612551a5d8f68378b45e3a51670c))

- Make MLSUM a multilingual task ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Make MLSUM multilingual ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Update languages ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

### Chores

- Add linting ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Add minilm results and comment ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Add more results ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Add points ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Add result ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Fix metadata ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

- Update metadata ([#556](https://github.com/embeddings-benchmark/mteb/pull/556),
  [`b02f481`](https://github.com/embeddings-benchmark/mteb/commit/b02f48146be608c558e1a742dd54196bc0031fc3))

### Documentation

- Added points ([#528](https://github.com/embeddings-benchmark/mteb/pull/528),
  [`1e44b98`](https://github.com/embeddings-benchmark/mteb/commit/1e44b98535eb612551a5d8f68378b45e3a51670c))


## v1.7.30 (2024-04-25)

### Bug Fixes

- Change MultiHateClassification to use dataset_transform
  ([#558](https://github.com/embeddings-benchmark/mteb/pull/558),
  [`2a3cd79`](https://github.com/embeddings-benchmark/mteb/commit/2a3cd79352860669623f096aa0125738f40c6a65))


## v1.7.29 (2024-04-25)

### Bug Fixes

- Add Malayalam news classification ([#485](https://github.com/embeddings-benchmark/mteb/pull/485),
  [`aea61f5`](https://github.com/embeddings-benchmark/mteb/commit/aea61f5cbed5e77ef81775f036f7d542671b8be1))

- Add Tamil news classification ([#484](https://github.com/embeddings-benchmark/mteb/pull/484),
  [`5a8ca99`](https://github.com/embeddings-benchmark/mteb/commit/5a8ca99a70670208fe75db335d0c8285bb2357cc))


## v1.7.28 (2024-04-25)

### Bug Fixes

- Telugu news classification ([#557](https://github.com/embeddings-benchmark/mteb/pull/557),
  [`e3b1ba7`](https://github.com/embeddings-benchmark/mteb/commit/e3b1ba722dadf4c51f405e3cd2271ae35e7601cb))


## v1.7.27 (2024-04-24)

### Bug Fixes

- Add IndicQA dataset in Retrieval task
  ([#547](https://github.com/embeddings-benchmark/mteb/pull/547),
  [`6eeb596`](https://github.com/embeddings-benchmark/mteb/commit/6eeb596a4e19459386aed62bca96e48e68fcaab0))

- Fix minor bugs in FollowIR implementation; bump datasets revision id to the fixed version
  ([#555](https://github.com/embeddings-benchmark/mteb/pull/555),
  [`7b6d2ec`](https://github.com/embeddings-benchmark/mteb/commit/7b6d2ecc72c5af19d5fdcee71a0f166043bbb7a0))


## v1.7.26 (2024-04-24)

### Bug Fixes

- Fill metadata for Danish datasets ([#549](https://github.com/embeddings-benchmark/mteb/pull/549),
  [`be8eced`](https://github.com/embeddings-benchmark/mteb/commit/be8eced8cf327c60c2d815a7793386e0dbb4183b))

- Move dalaj to swedish and fill metadata
  ([#549](https://github.com/embeddings-benchmark/mteb/pull/549),
  [`be8eced`](https://github.com/embeddings-benchmark/mteb/commit/be8eced8cf327c60c2d815a7793386e0dbb4183b))

### Chores

- Add bibtex ([#549](https://github.com/embeddings-benchmark/mteb/pull/549),
  [`be8eced`](https://github.com/embeddings-benchmark/mteb/commit/be8eced8cf327c60c2d815a7793386e0dbb4183b))

- Fill metadata for BornholmBitextMining
  ([#549](https://github.com/embeddings-benchmark/mteb/pull/549),
  [`be8eced`](https://github.com/embeddings-benchmark/mteb/commit/be8eced8cf327c60c2d815a7793386e0dbb4183b))


## v1.7.25 (2024-04-24)

### Bug Fixes

- Add Urdu Latin Sentiment Classification Dataset
  ([#535](https://github.com/embeddings-benchmark/mteb/pull/535),
  [`aed0c75`](https://github.com/embeddings-benchmark/mteb/commit/aed0c758408d8f4f638b0848aac4ebea988b20be))


## v1.7.24 (2024-04-24)

### Bug Fixes

- Added the first thai classification dataset
  ([#538](https://github.com/embeddings-benchmark/mteb/pull/538),
  [`5f5935c`](https://github.com/embeddings-benchmark/mteb/commit/5f5935c93baaff217684da160a010b2205e7a075))


## v1.7.23 (2024-04-24)

### Bug Fixes

- Add Financial Phrasebank dataset ([#499](https://github.com/embeddings-benchmark/mteb/pull/499),
  [`c79a509`](https://github.com/embeddings-benchmark/mteb/commit/c79a509b0f17bdd4e402126109cb8280fd5fdac5))


## v1.7.22 (2024-04-24)

### Bug Fixes

- AddingGeorgian Sentiment Classification
  ([#534](https://github.com/embeddings-benchmark/mteb/pull/534),
  [`31d3b57`](https://github.com/embeddings-benchmark/mteb/commit/31d3b5794a92430b6f2049e3c224cf89a3ba24a6))

- Update missing results ([#527](https://github.com/embeddings-benchmark/mteb/pull/527),
  [`7b595d7`](https://github.com/embeddings-benchmark/mteb/commit/7b595d7ba1d60423a882af328725ad641008db2e))

- Update previous tasks and scores using new subsampling function
  ([#527](https://github.com/embeddings-benchmark/mteb/pull/527),
  [`7b595d7`](https://github.com/embeddings-benchmark/mteb/commit/7b595d7ba1d60423a882af328725ad641008db2e))

- Update ToxicChatClassification languages
  ([#527](https://github.com/embeddings-benchmark/mteb/pull/527),
  [`7b595d7`](https://github.com/embeddings-benchmark/mteb/commit/7b595d7ba1d60423a882af328725ad641008db2e))

### Chores

- Add linting ([#527](https://github.com/embeddings-benchmark/mteb/pull/527),
  [`7b595d7`](https://github.com/embeddings-benchmark/mteb/commit/7b595d7ba1d60423a882af328725ad641008db2e))

- Add points ([#527](https://github.com/embeddings-benchmark/mteb/pull/527),
  [`7b595d7`](https://github.com/embeddings-benchmark/mteb/commit/7b595d7ba1d60423a882af328725ad641008db2e))

### Documentation

- Updated ordering of languages
  ([`6faae86`](https://github.com/embeddings-benchmark/mteb/commit/6faae8658f47d5bebc79a665529740b472519856))


## v1.7.21 (2024-04-24)

### Bug Fixes

- Add Armenian Paraphrase Dataset ([#537](https://github.com/embeddings-benchmark/mteb/pull/537),
  [`37bf08e`](https://github.com/embeddings-benchmark/mteb/commit/37bf08ebd59fedeead29ced1293095a25564447e))

- Add Multilingual Hate Speech detection task
  ([#439](https://github.com/embeddings-benchmark/mteb/pull/439),
  [`eee7175`](https://github.com/embeddings-benchmark/mteb/commit/eee71755a403143f21c2fd09518742c689a36681))

- Added encoder interfaces ([#469](https://github.com/embeddings-benchmark/mteb/pull/469),
  [`7625b1d`](https://github.com/embeddings-benchmark/mteb/commit/7625b1d15cdc55248ecc2cc9873732c4dd3f9cc0))

- Added Sentiment Analysis Bengali Dataset
  ([#536](https://github.com/embeddings-benchmark/mteb/pull/536),
  [`ba9bcaa`](https://github.com/embeddings-benchmark/mteb/commit/ba9bcaa2dc976de407b05da86176d8929ee5e96e))

- First Turkish Retrieval dataset ([#533](https://github.com/embeddings-benchmark/mteb/pull/533),
  [`6607f2f`](https://github.com/embeddings-benchmark/mteb/commit/6607f2fc80620c0e6e841e5fb7d3007b68116f04))

### Continuous Integration

- Added missing dependency ([#469](https://github.com/embeddings-benchmark/mteb/pull/469),
  [`7625b1d`](https://github.com/embeddings-benchmark/mteb/commit/7625b1d15cdc55248ecc2cc9873732c4dd3f9cc0))

- Enable linting on all PRs ([#469](https://github.com/embeddings-benchmark/mteb/pull/469),
  [`7625b1d`](https://github.com/embeddings-benchmark/mteb/commit/7625b1d15cdc55248ecc2cc9873732c4dd3f9cc0))

- Fix create tasks table loop by sorting languages
  ([#541](https://github.com/embeddings-benchmark/mteb/pull/541),
  [`ed16cbf`](https://github.com/embeddings-benchmark/mteb/commit/ed16cbffd78886b9accb26feebd50105a3c933c7))

### Documentation

- Added points ([#469](https://github.com/embeddings-benchmark/mteb/pull/469),
  [`7625b1d`](https://github.com/embeddings-benchmark/mteb/commit/7625b1d15cdc55248ecc2cc9873732c4dd3f9cc0))

- Added script for creating point ([#469](https://github.com/embeddings-benchmark/mteb/pull/469),
  [`7625b1d`](https://github.com/embeddings-benchmark/mteb/commit/7625b1d15cdc55248ecc2cc9873732c4dd3f9cc0))


## v1.7.20 (2024-04-24)

### Bug Fixes

- Add Clustering dataset for Indic languages
  ([#532](https://github.com/embeddings-benchmark/mteb/pull/532),
  [`dc9ba24`](https://github.com/embeddings-benchmark/mteb/commit/dc9ba24bfa48774e488df671bc2fb6df0080c2b3))

### Documentation

- LB tab instructions ([#531](https://github.com/embeddings-benchmark/mteb/pull/531),
  [`e4e627d`](https://github.com/embeddings-benchmark/mteb/commit/e4e627deef6099d61d062a4297562bc4e7904ef9))


## v1.7.19 (2024-04-24)

### Bug Fixes

- Add yelp review dataset ([#478](https://github.com/embeddings-benchmark/mteb/pull/478),
  [`5458f2b`](https://github.com/embeddings-benchmark/mteb/commit/5458f2b81d5e0e15de32eaa9003ae07024339f33))

- Added Hindi sentiment analysis dataset
  ([#491](https://github.com/embeddings-benchmark/mteb/pull/491),
  [`4ed19ce`](https://github.com/embeddings-benchmark/mteb/commit/4ed19ce4e48d01ef3f8b4d27f20c8b8a6fed45d9))

- Added Kannada News Classification Dataset
  ([#492](https://github.com/embeddings-benchmark/mteb/pull/492),
  [`e1b6c1c`](https://github.com/embeddings-benchmark/mteb/commit/e1b6c1c366add45edfea6e4de3aafcee44212d4b))


## v1.7.18 (2024-04-24)

### Bug Fixes

- Add Nepali news classification dataset
  ([#502](https://github.com/embeddings-benchmark/mteb/pull/502),
  [`dd3f0a5`](https://github.com/embeddings-benchmark/mteb/commit/dd3f0a502f1980f969e959cedc2d201560891186))

### Documentation

- Added create task table ([#525](https://github.com/embeddings-benchmark/mteb/pull/525),
  [`5974e05`](https://github.com/embeddings-benchmark/mteb/commit/5974e058abaf0a6d7ccfbd246d535530e974a700))


## v1.7.17 (2024-04-23)

### Bug Fixes

- Add Indic STS benchmark dataset ([#524](https://github.com/embeddings-benchmark/mteb/pull/524),
  [`1f26615`](https://github.com/embeddings-benchmark/mteb/commit/1f2661513c697246ffc816d5f0dcf40a02c1db8e))

### Chores

- Move files to correct folder ([#526](https://github.com/embeddings-benchmark/mteb/pull/526),
  [`5370b44`](https://github.com/embeddings-benchmark/mteb/commit/5370b44435e63a9c00e76db51d01812824b559d0))


## v1.7.16 (2024-04-23)

### Bug Fixes

- Add get_tasks ([#522](https://github.com/embeddings-benchmark/mteb/pull/522),
  [`ede97a1`](https://github.com/embeddings-benchmark/mteb/commit/ede97a15fc6c6dd74ccef4407daa57b5d3088915))

### Documentation

- Added points ([#522](https://github.com/embeddings-benchmark/mteb/pull/522),
  [`ede97a1`](https://github.com/embeddings-benchmark/mteb/commit/ede97a15fc6c6dd74ccef4407daa57b5d3088915))


## v1.7.15 (2024-04-23)

### Bug Fixes

- Update current dict and do not remove other splits
  ([#521](https://github.com/embeddings-benchmark/mteb/pull/521),
  [`26e7cbd`](https://github.com/embeddings-benchmark/mteb/commit/26e7cbdd2c97e43ebcac7f8eb3a2779fd302c59f))


## v1.7.14 (2024-04-23)

### Bug Fixes

- Add Indic language identification dataset in Classification category
  ([#514](https://github.com/embeddings-benchmark/mteb/pull/514),
  [`e1833d5`](https://github.com/embeddings-benchmark/mteb/commit/e1833d545824cc27be45d93a5906682df794ad57))


## v1.7.13 (2024-04-23)

### Bug Fixes

- Fast Clustering ([#481](https://github.com/embeddings-benchmark/mteb/pull/481),
  [`8d454bd`](https://github.com/embeddings-benchmark/mteb/commit/8d454bdbcf44ecc89de859bd67903e53052659e2))

### Documentation

- Added points ([#481](https://github.com/embeddings-benchmark/mteb/pull/481),
  [`8d454bd`](https://github.com/embeddings-benchmark/mteb/commit/8d454bdbcf44ecc89de859bd67903e53052659e2))


## v1.7.12 (2024-04-23)

### Bug Fixes

- Stratify subsample fixes ([#518](https://github.com/embeddings-benchmark/mteb/pull/518),
  [`439ba93`](https://github.com/embeddings-benchmark/mteb/commit/439ba935bbb1017334b0bcc8e765df1f54933182))

### Chores

- Unify subsampling with stratify ([#513](https://github.com/embeddings-benchmark/mteb/pull/513),
  [`10922bb`](https://github.com/embeddings-benchmark/mteb/commit/10922bbb61ddeff810508489ed5ca04c1e8263ae))


## v1.7.11 (2024-04-23)

### Bug Fixes

- Add Arxiv classification dataset ([#479](https://github.com/embeddings-benchmark/mteb/pull/479),
  [`bb56724`](https://github.com/embeddings-benchmark/mteb/commit/bb567240d6363cfa77bd5c318703e9b022e54593))

- Add rerankers to mteb, following `CrossEncoder`
  ([#457](https://github.com/embeddings-benchmark/mteb/pull/457),
  [`23fe8bd`](https://github.com/embeddings-benchmark/mteb/commit/23fe8bdc99c326a766c811adb27bac828e8bcbc0))

### Documentation

- Added points for pr ([#457](https://github.com/embeddings-benchmark/mteb/pull/457),
  [`23fe8bd`](https://github.com/embeddings-benchmark/mteb/commit/23fe8bdc99c326a766c811adb27bac828e8bcbc0))


## v1.7.10 (2024-04-23)

### Bug Fixes

- Punjabi news classification ([#500](https://github.com/embeddings-benchmark/mteb/pull/500),
  [`eda2020`](https://github.com/embeddings-benchmark/mteb/commit/eda2020f59f96ae7b3e18f6a06c789aad71062c0))

### Continuous Integration

- Hotfix for points ([#510](https://github.com/embeddings-benchmark/mteb/pull/510),
  [`032ca25`](https://github.com/embeddings-benchmark/mteb/commit/032ca25d88f246ad12811583384b56261cf6ecb9))


## v1.7.9 (2024-04-23)

### Bug Fixes

- Add IndicSentimentClassification dataset for Indic languages
  ([#489](https://github.com/embeddings-benchmark/mteb/pull/489),
  [`61ffadf`](https://github.com/embeddings-benchmark/mteb/commit/61ffadf8ed19737de175a1a082ee9f9a7452fa0f))


## v1.7.8 (2024-04-23)

### Bug Fixes

- Add french squad ([#460](https://github.com/embeddings-benchmark/mteb/pull/460),
  [`0e24482`](https://github.com/embeddings-benchmark/mteb/commit/0e24482292757da1ebd82ea1135ef46c3e2583bf))


## v1.7.7 (2024-04-23)

### Bug Fixes

- Add Gujarati news classification ([#493](https://github.com/embeddings-benchmark/mteb/pull/493),
  [`3a59a4d`](https://github.com/embeddings-benchmark/mteb/commit/3a59a4dbdb4a4b4d20bddb64670cf209135116a6))


## v1.7.6 (2024-04-22)

### Bug Fixes

- Update WikiClusteringP2P with 10 new languages
  ([#486](https://github.com/embeddings-benchmark/mteb/pull/486),
  [`100e160`](https://github.com/embeddings-benchmark/mteb/commit/100e160337321814187ad4abc5018c0adf54782c))


## v1.7.5 (2024-04-22)

### Bug Fixes

- ToxicChatClassification task added to multilingual
  ([#480](https://github.com/embeddings-benchmark/mteb/pull/480),
  [`eb1959d`](https://github.com/embeddings-benchmark/mteb/commit/eb1959da84762ccd465e5a4d6352cb926391af61))

### Continuous Integration

- Only create table in ci when pushing to main
  ([#473](https://github.com/embeddings-benchmark/mteb/pull/473),
  [`738076d`](https://github.com/embeddings-benchmark/mteb/commit/738076de0c5115331eb908dc61386b7681e915e9))

- Use if-else instead for points table creation
  ([#477](https://github.com/embeddings-benchmark/mteb/pull/477),
  [`317ee44`](https://github.com/embeddings-benchmark/mteb/commit/317ee44073c93946ba693e35114a8b62f3dcee4d))


## v1.7.4 (2024-04-21)

### Bug Fixes

- Add Siswati News Classification ([#474](https://github.com/embeddings-benchmark/mteb/pull/474),
  [`9201dff`](https://github.com/embeddings-benchmark/mteb/commit/9201dffc46b32a5dc70437c9aff175041c7f7cf8))


## v1.7.3 (2024-04-21)

### Bug Fixes

- Add IN22-Gen dataset under BitextMining task
  ([#451](https://github.com/embeddings-benchmark/mteb/pull/451),
  [`d8ab2e7`](https://github.com/embeddings-benchmark/mteb/commit/d8ab2e718418fc7c375abcdd63996120f882ae0a))


## v1.7.2 (2024-04-21)

### Bug Fixes

- Add GreekLegalCodeClassification dataset
  ([#456](https://github.com/embeddings-benchmark/mteb/pull/456),
  [`0c53579`](https://github.com/embeddings-benchmark/mteb/commit/0c53579db92318c115fc79c34f08864c3657ef30))


## v1.7.1 (2024-04-21)

### Bug Fixes

- Add Filipino Classification Dataset
  ([#472](https://github.com/embeddings-benchmark/mteb/pull/472),
  [`611f56a`](https://github.com/embeddings-benchmark/mteb/commit/611f56a26ade11186ec4d58d13a00c6cd3c3eede))

### Continuous Integration

- Fixes issue when there is no point updates
  ([`22ca0e4`](https://github.com/embeddings-benchmark/mteb/commit/22ca0e4e941b14d62d543d866fb4daecacdd2805))


## v1.7.0 (2024-04-20)

### Chores

- Add points ([#470](https://github.com/embeddings-benchmark/mteb/pull/470),
  [`153bbb7`](https://github.com/embeddings-benchmark/mteb/commit/153bbb73965020edd70843ab043ebef319a099b2))

- Update license ([#470](https://github.com/embeddings-benchmark/mteb/pull/470),
  [`153bbb7`](https://github.com/embeddings-benchmark/mteb/commit/153bbb73965020edd70843ab043ebef319a099b2))

### Documentation

- Add points table ([#468](https://github.com/embeddings-benchmark/mteb/pull/468),
  [`e22c0c8`](https://github.com/embeddings-benchmark/mteb/commit/e22c0c8153c921940a3588548247e6a3b1742a0d))

- Add points table 2 ([#471](https://github.com/embeddings-benchmark/mteb/pull/471),
  [`fde4876`](https://github.com/embeddings-benchmark/mteb/commit/fde4876a2f0298fb5e5baefdf77e269103af2534))

- Added script for creating point ([#468](https://github.com/embeddings-benchmark/mteb/pull/468),
  [`e22c0c8`](https://github.com/embeddings-benchmark/mteb/commit/e22c0c8153c921940a3588548247e6a3b1742a0d))

### Features

- Add Greek sentiment classification ([#470](https://github.com/embeddings-benchmark/mteb/pull/470),
  [`153bbb7`](https://github.com/embeddings-benchmark/mteb/commit/153bbb73965020edd70843ab043ebef319a099b2))

- Add GreekSentimentClassifciation task
  ([#470](https://github.com/embeddings-benchmark/mteb/pull/470),
  [`153bbb7`](https://github.com/embeddings-benchmark/mteb/commit/153bbb73965020edd70843ab043ebef319a099b2))

- Add metadata ([#470](https://github.com/embeddings-benchmark/mteb/pull/470),
  [`153bbb7`](https://github.com/embeddings-benchmark/mteb/commit/153bbb73965020edd70843ab043ebef319a099b2))


## v1.6.38 (2024-04-20)

### Bug Fixes

- Add Zulu/Isizulu News Classification
  ([#466](https://github.com/embeddings-benchmark/mteb/pull/466),
  [`d9383b7`](https://github.com/embeddings-benchmark/mteb/commit/d9383b7a289f228889e1d225580b132374e76754))

### Continuous Integration

- Added doc linting rules to ensure google style docstrings.
  ([#461](https://github.com/embeddings-benchmark/mteb/pull/461),
  [`a8c0703`](https://github.com/embeddings-benchmark/mteb/commit/a8c07032fbfecd3730c7a6b1f90391bebcf22ecd))

### Documentation

- Added points ([#461](https://github.com/embeddings-benchmark/mteb/pull/461),
  [`a8c0703`](https://github.com/embeddings-benchmark/mteb/commit/a8c07032fbfecd3730c7a6b1f90391bebcf22ecd))


## v1.6.37 (2024-04-20)

### Bug Fixes

- Add Bambara Sentiment Classification
  ([#463](https://github.com/embeddings-benchmark/mteb/pull/463),
  [`f128a37`](https://github.com/embeddings-benchmark/mteb/commit/f128a3721de23eccb5ece7d87096d88c2a3a4fe0))

### Documentation

- Coordination contributions for mmteb by auto validating points
  ([#462](https://github.com/embeddings-benchmark/mteb/pull/462),
  [`6121822`](https://github.com/embeddings-benchmark/mteb/commit/6121822c2f588d3995af41479b5172c07be06680))

- Fix broken link in README ([#458](https://github.com/embeddings-benchmark/mteb/pull/458),
  [`57eb5c3`](https://github.com/embeddings-benchmark/mteb/commit/57eb5c37f9510ac9e06c9cbcfcacd4da7cb2241d))


## v1.6.36 (2024-04-19)

### Bug Fixes

- Set the min Pydantic version to 2 ([#453](https://github.com/embeddings-benchmark/mteb/pull/453),
  [`92e089e`](https://github.com/embeddings-benchmark/mteb/commit/92e089eaca953067eb52fea95ac2370edea5f83a))


## v1.6.35 (2024-04-19)

### Bug Fixes

- Fix split name of AlloprofRetrieval and SyntecRetrieval
  ([#449](https://github.com/embeddings-benchmark/mteb/pull/449),
  [`734e946`](https://github.com/embeddings-benchmark/mteb/commit/734e946fcfb0a58354b8fb1410950ac77c8bded6))

- Update eval split key ([#449](https://github.com/embeddings-benchmark/mteb/pull/449),
  [`734e946`](https://github.com/embeddings-benchmark/mteb/commit/734e946fcfb0a58354b8fb1410950ac77c8bded6))

- Update revision for alloprof retrieval
  ([#449](https://github.com/embeddings-benchmark/mteb/pull/449),
  [`734e946`](https://github.com/embeddings-benchmark/mteb/commit/734e946fcfb0a58354b8fb1410950ac77c8bded6))

- Update SyntecRetrieval metadata and loading fn
  ([#449](https://github.com/embeddings-benchmark/mteb/pull/449),
  [`734e946`](https://github.com/embeddings-benchmark/mteb/commit/734e946fcfb0a58354b8fb1410950ac77c8bded6))

### Chores

- Add linting ([#449](https://github.com/embeddings-benchmark/mteb/pull/449),
  [`734e946`](https://github.com/embeddings-benchmark/mteb/commit/734e946fcfb0a58354b8fb1410950ac77c8bded6))

- Add results of run_mteb_french script
  ([#449](https://github.com/embeddings-benchmark/mteb/pull/449),
  [`734e946`](https://github.com/embeddings-benchmark/mteb/commit/734e946fcfb0a58354b8fb1410950ac77c8bded6))

### Continuous Integration

- Correct jsonl validation ([#448](https://github.com/embeddings-benchmark/mteb/pull/448),
  [`22ef608`](https://github.com/embeddings-benchmark/mteb/commit/22ef608a2537e35fafc752fa53df4ef40876c344))


## v1.6.34 (2024-04-19)

### Bug Fixes

- Add AbsTaskInstructionRetrieval from the FollowIR paper
  ([#408](https://github.com/embeddings-benchmark/mteb/pull/408),
  [`b700f94`](https://github.com/embeddings-benchmark/mteb/commit/b700f946a15871d93b6ad31dfefc4ba8c862e52c))


## v1.6.33 (2024-04-19)

### Bug Fixes

- Add Persian Food Sentiment Classification
  ([#447](https://github.com/embeddings-benchmark/mteb/pull/447),
  [`5aac728`](https://github.com/embeddings-benchmark/mteb/commit/5aac7289c0d5f78f6dcc5f5f0fdfd4565f81058e))

### Chores

- Validate jsonl points files ([#441](https://github.com/embeddings-benchmark/mteb/pull/441),
  [`8dccdc0`](https://github.com/embeddings-benchmark/mteb/commit/8dccdc07e2bf4093be5792d8ebcb549d61670c38))


## v1.6.32 (2024-04-19)

### Bug Fixes

- Add Bulgarian Online Store Review Dataset
  ([#445](https://github.com/embeddings-benchmark/mteb/pull/445),
  [`8109e2d`](https://github.com/embeddings-benchmark/mteb/commit/8109e2dcaa97cd42821c0e871d19c507027f7c4c))

- Add Kurdish sentiment ([#446](https://github.com/embeddings-benchmark/mteb/pull/446),
  [`61dee74`](https://github.com/embeddings-benchmark/mteb/commit/61dee74f531a9bd5f10531fea1ba61c70bc7b9a9))


## v1.6.31 (2024-04-19)

### Bug Fixes

- Italian cola ([#444](https://github.com/embeddings-benchmark/mteb/pull/444),
  [`16438ce`](https://github.com/embeddings-benchmark/mteb/commit/16438ceccfdfe5a50e64d5df32d599e8301efeb1))

- Merge Italian cola (#348) ([#444](https://github.com/embeddings-benchmark/mteb/pull/444),
  [`16438ce`](https://github.com/embeddings-benchmark/mteb/commit/16438ceccfdfe5a50e64d5df32d599e8301efeb1))

### Documentation

- Added points ([#444](https://github.com/embeddings-benchmark/mteb/pull/444),
  [`16438ce`](https://github.com/embeddings-benchmark/mteb/commit/16438ceccfdfe5a50e64d5df32d599e8301efeb1))


## v1.6.30 (2024-04-19)

### Bug Fixes

- Add Wikiclustering benchmark for multiple languages:)) (#376)
  ([#443](https://github.com/embeddings-benchmark/mteb/pull/443),
  [`c2c6d52`](https://github.com/embeddings-benchmark/mteb/commit/c2c6d527d0787a8b052eaf5aca9f32ea1128af11))

### Chores

- Remove useless sub directory ([#442](https://github.com/embeddings-benchmark/mteb/pull/442),
  [`ce10347`](https://github.com/embeddings-benchmark/mteb/commit/ce103474350cb41388746c4db10fd60f81e1a32f))


## v1.6.29 (2024-04-19)

### Bug Fixes

- Add Uyghur Sentiment Classification
  ([#430](https://github.com/embeddings-benchmark/mteb/pull/430),
  [`ac4315b`](https://github.com/embeddings-benchmark/mteb/commit/ac4315b8ccd5daf7d72a8ad4e13f15bfc221d1b8))


## v1.6.28 (2024-04-19)

### Bug Fixes

- Add Bulgarian Sentiment Classification Dataset
  ([#429](https://github.com/embeddings-benchmark/mteb/pull/429),
  [`ea9c386`](https://github.com/embeddings-benchmark/mteb/commit/ea9c3869b1e52b8efedc93975cd6cb52e5ffe7f8))

### Documentation

- Added new scoring system ([#438](https://github.com/embeddings-benchmark/mteb/pull/438),
  [`0bb2d7a`](https://github.com/embeddings-benchmark/mteb/commit/0bb2d7ac3a502c48b9e64d0ae907ae99efa6a6d2))

- Correct review points ([#437](https://github.com/embeddings-benchmark/mteb/pull/437),
  [`f3abf71`](https://github.com/embeddings-benchmark/mteb/commit/f3abf717031d7b65eea2ef67519eb398ffe0a713))

- Updated review points ([#437](https://github.com/embeddings-benchmark/mteb/pull/437),
  [`f3abf71`](https://github.com/embeddings-benchmark/mteb/commit/f3abf717031d7b65eea2ef67519eb398ffe0a713))


## v1.6.27 (2024-04-19)

### Bug Fixes

- Add SlovakSentimentClassification dataset
  ([#433](https://github.com/embeddings-benchmark/mteb/pull/433),
  [`4988867`](https://github.com/embeddings-benchmark/mteb/commit/4988867f5f579c278354cedd373c46782b56376a))

### Documentation

- Updated review points ([#435](https://github.com/embeddings-benchmark/mteb/pull/435),
  [`160a607`](https://github.com/embeddings-benchmark/mteb/commit/160a60796a038ac7b7792afdbd82bccbf7c7e7f7))


## v1.6.26 (2024-04-19)

### Bug Fixes

- Add Romanian STS dataset ([#427](https://github.com/embeddings-benchmark/mteb/pull/427),
  [`0678036`](https://github.com/embeddings-benchmark/mteb/commit/0678036c2b3cd949c383269f3a56d2062dfb6c60))


## v1.6.25 (2024-04-18)

### Bug Fixes

- Update dataset source name for GermanDPR
  ([#428](https://github.com/embeddings-benchmark/mteb/pull/428),
  [`9ee6bbe`](https://github.com/embeddings-benchmark/mteb/commit/9ee6bbefdd835f6dd4c09be659675353f4301df5))

### Documentation

- Add missing points from PR reviews ([#425](https://github.com/embeddings-benchmark/mteb/pull/425),
  [`d3c16ed`](https://github.com/embeddings-benchmark/mteb/commit/d3c16edc6e0b53cee0588d2921fb632660a3a793))

- Add point for bug fix ([#428](https://github.com/embeddings-benchmark/mteb/pull/428),
  [`9ee6bbe`](https://github.com/embeddings-benchmark/mteb/commit/9ee6bbefdd835f6dd4c09be659675353f4301df5))


## v1.6.24 (2024-04-18)

### Bug Fixes

- Add Maltese Sentiment Classification Dataset
  ([#423](https://github.com/embeddings-benchmark/mteb/pull/423),
  [`8276549`](https://github.com/embeddings-benchmark/mteb/commit/82765497fc90ff31e7e4b8ccc6485287f659b74b))


## v1.6.23 (2024-04-18)

### Bug Fixes

- Add KorSTS and KLUE-STS datasets ([#414](https://github.com/embeddings-benchmark/mteb/pull/414),
  [`4135e9b`](https://github.com/embeddings-benchmark/mteb/commit/4135e9bc0ffe42777f559fbc7226d4bdfaf1fbed))

### Refactoring

- Ruff linting ([#414](https://github.com/embeddings-benchmark/mteb/pull/414),
  [`4135e9b`](https://github.com/embeddings-benchmark/mteb/commit/4135e9bc0ffe42777f559fbc7226d4bdfaf1fbed))


## v1.6.22 (2024-04-18)

### Bug Fixes

- Add support for the LongEmbed benchmark (#393)
  ([#421](https://github.com/embeddings-benchmark/mteb/pull/421),
  [`d21f25e`](https://github.com/embeddings-benchmark/mteb/commit/d21f25e0096264b68a4b3f7a72a0a158cb3c1fd5))


## v1.6.21 (2024-04-18)

### Bug Fixes

- Add Croatian Sentiment Classification
  ([#416](https://github.com/embeddings-benchmark/mteb/pull/416),
  [`e7c0362`](https://github.com/embeddings-benchmark/mteb/commit/e7c0362cd70edcb4fa99f332fdb9baf2ca50ffe6))

- EstQA is now properly formulated ([#418](https://github.com/embeddings-benchmark/mteb/pull/418),
  [`ae0da50`](https://github.com/embeddings-benchmark/mteb/commit/ae0da506b5a3e20024f5ae853fd06d234c7815f3))


## v1.6.20 (2024-04-18)

### Bug Fixes

- Add italian HateSpeech dataset ([#420](https://github.com/embeddings-benchmark/mteb/pull/420),
  [`4ce7f35`](https://github.com/embeddings-benchmark/mteb/commit/4ce7f35ce2e43eb5659e67abb32379cb99d4641f))

- Add italian HateSpeech dataset (#385)
  ([#420](https://github.com/embeddings-benchmark/mteb/pull/420),
  [`4ce7f35`](https://github.com/embeddings-benchmark/mteb/commit/4ce7f35ce2e43eb5659e67abb32379cb99d4641f))

### Documentation

- Update contributor information ([#417](https://github.com/embeddings-benchmark/mteb/pull/417),
  [`36de85e`](https://github.com/embeddings-benchmark/mteb/commit/36de85eb17976d595f3e9430f8cdc5bfc5dfa146))


## v1.6.19 (2024-04-18)

### Bug Fixes

- Add custom load dataset function ([#405](https://github.com/embeddings-benchmark/mteb/pull/405),
  [`c549af2`](https://github.com/embeddings-benchmark/mteb/commit/c549af2cc076f6b6474c36013adca485eedaa841))

- Add custom load dataset function for MLSUM tasks
  ([#405](https://github.com/embeddings-benchmark/mteb/pull/405),
  [`c549af2`](https://github.com/embeddings-benchmark/mteb/commit/c549af2cc076f6b6474c36013adca485eedaa841))

- Fix linter ([#405](https://github.com/embeddings-benchmark/mteb/pull/405),
  [`c549af2`](https://github.com/embeddings-benchmark/mteb/commit/c549af2cc076f6b6474c36013adca485eedaa841))

### Documentation

- Added points ([#405](https://github.com/embeddings-benchmark/mteb/pull/405),
  [`c549af2`](https://github.com/embeddings-benchmark/mteb/commit/c549af2cc076f6b6474c36013adca485eedaa841))


## v1.6.18 (2024-04-18)

### Bug Fixes

- Added Hungarian Roma Tales bitext task and Romani Bible clustering task
  ([#396](https://github.com/embeddings-benchmark/mteb/pull/396),
  [`6f9d19f`](https://github.com/embeddings-benchmark/mteb/commit/6f9d19f607c19b492cc5236855a80f68650910cc))

### Code Style

- Use rename_columns instead of rename_column
  ([#396](https://github.com/embeddings-benchmark/mteb/pull/396),
  [`6f9d19f`](https://github.com/embeddings-benchmark/mteb/commit/6f9d19f607c19b492cc5236855a80f68650910cc))


## v1.6.17 (2024-04-18)

### Bug Fixes

- Id clickbait ([#411](https://github.com/embeddings-benchmark/mteb/pull/411),
  [`a0fa60b`](https://github.com/embeddings-benchmark/mteb/commit/a0fa60b32335e31c0aa6dbb2861d14c3c3ce06e8))


## v1.6.16 (2024-04-17)

### Bug Fixes

- Add BengaliHateSpeechClassification
  ([#398](https://github.com/embeddings-benchmark/mteb/pull/398),
  [`2772ec8`](https://github.com/embeddings-benchmark/mteb/commit/2772ec8ef3051778c39c29fada964f4f95898c3e))

### Documentation

- Added missing point for pr
  ([`32d8979`](https://github.com/embeddings-benchmark/mteb/commit/32d89793825417ac7fb94fd7f8f3fac1d60b6b29))

- Added missing point for pr ([#394](https://github.com/embeddings-benchmark/mteb/pull/394),
  [`a2d4705`](https://github.com/embeddings-benchmark/mteb/commit/a2d470572219aa5afab32c4f276db01db7d921d5))


## v1.6.15 (2024-04-17)

### Bug Fixes

- Add Macedonian Tweet Sentiment Classification Task
  ([#392](https://github.com/embeddings-benchmark/mteb/pull/392),
  [`206f2c9`](https://github.com/embeddings-benchmark/mteb/commit/206f2c9e82417da69c6d20688bd7eabc6cff9403))


## v1.6.14 (2024-04-17)

### Bug Fixes

- Add Dutch Book Review Sentiment Classification Task
  ([#388](https://github.com/embeddings-benchmark/mteb/pull/388),
  [`10d50b6`](https://github.com/embeddings-benchmark/mteb/commit/10d50b6908c304e1a293888b51bdcc62f90142ea))

### Code Style

- Running linting ([#394](https://github.com/embeddings-benchmark/mteb/pull/394),
  [`ff3cbfc`](https://github.com/embeddings-benchmark/mteb/commit/ff3cbfc29e89c8cd1c4f3bddbadf0a222be0d75a))

- This should fail linting ([#394](https://github.com/embeddings-benchmark/mteb/pull/394),
  [`ff3cbfc`](https://github.com/embeddings-benchmark/mteb/commit/ff3cbfc29e89c8cd1c4f3bddbadf0a222be0d75a))

### Continuous Integration

- Ensure that linting fails when files are not linted
  ([#394](https://github.com/embeddings-benchmark/mteb/pull/394),
  [`ff3cbfc`](https://github.com/embeddings-benchmark/mteb/commit/ff3cbfc29e89c8cd1c4f3bddbadf0a222be0d75a))

- Update linting to check if project is linted
  ([#394](https://github.com/embeddings-benchmark/mteb/pull/394),
  [`ff3cbfc`](https://github.com/embeddings-benchmark/mteb/commit/ff3cbfc29e89c8cd1c4f3bddbadf0a222be0d75a))


## v1.6.13 (2024-04-17)

### Bug Fixes

- Added HunSum2 dataset with results ([#384](https://github.com/embeddings-benchmark/mteb/pull/384),
  [`c65e1d7`](https://github.com/embeddings-benchmark/mteb/commit/c65e1d70a8379056b40db31b3cbe59a699e811ba))


## v1.6.12 (2024-04-17)

### Bug Fixes

- Added EstQA and Eesti Valentsikorpus datasets
  ([#382](https://github.com/embeddings-benchmark/mteb/pull/382),
  [`282d421`](https://github.com/embeddings-benchmark/mteb/commit/282d4219e15f271e87785c30f84d42675a4e5a05))


## v1.6.11 (2024-04-16)

### Bug Fixes

- Add Arabic Restaurant Reviews ([#369](https://github.com/embeddings-benchmark/mteb/pull/369),
  [`dc82dc5`](https://github.com/embeddings-benchmark/mteb/commit/dc82dc53d33528445543b86eab1267e3cda05b8f))

- Add neuclir ([#350](https://github.com/embeddings-benchmark/mteb/pull/350),
  [`e773f64`](https://github.com/embeddings-benchmark/mteb/commit/e773f641cc9291945137a397c0928665a37a550f))

- Moved metadata calcuation to abstask and added minor updates to metadata
  ([#350](https://github.com/embeddings-benchmark/mteb/pull/350),
  [`e773f64`](https://github.com/embeddings-benchmark/mteb/commit/e773f641cc9291945137a397c0928665a37a550f))

### Documentation

- Added points
  ([`dc82dc5`](https://github.com/embeddings-benchmark/mteb/commit/dc82dc53d33528445543b86eab1267e3cda05b8f))

- Formatted points tables
  ([`dc82dc5`](https://github.com/embeddings-benchmark/mteb/commit/dc82dc53d33528445543b86eab1267e3cda05b8f))


## v1.6.10 (2024-04-15)

### Bug Fixes

- ADD JSTS Dataset for STS TASK ([#359](https://github.com/embeddings-benchmark/mteb/pull/359),
  [`7ce0601`](https://github.com/embeddings-benchmark/mteb/commit/7ce060123da6ebb844f4ef7cb3e6226669252e3b))


## v1.6.9 (2024-04-15)

### Bug Fixes

- Add Cantonese Openrice Classification task
  ([#370](https://github.com/embeddings-benchmark/mteb/pull/370),
  [`91cf99e`](https://github.com/embeddings-benchmark/mteb/commit/91cf99eb471e472dceea2b57b65bad48d45fb3bb))

### Documentation

- Update mmteb points for slvnwhrl ([#373](https://github.com/embeddings-benchmark/mteb/pull/373),
  [`b03573b`](https://github.com/embeddings-benchmark/mteb/commit/b03573b20ba615b0b74408c5a49610aab5b3b56e))


## v1.6.8 (2024-04-15)

### Bug Fixes

- Add German False friend dataset ([#374](https://github.com/embeddings-benchmark/mteb/pull/374),
  [`9ef2246`](https://github.com/embeddings-benchmark/mteb/commit/9ef22460196cdc38cdaa1f77bb6b53a759d9c7ec))

- Added false friends en de dataset to pair classification
  ([#374](https://github.com/embeddings-benchmark/mteb/pull/374),
  [`9ef2246`](https://github.com/embeddings-benchmark/mteb/commit/9ef22460196cdc38cdaa1f77bb6b53a759d9c7ec))

- Ran linting ([#374](https://github.com/embeddings-benchmark/mteb/pull/374),
  [`9ef2246`](https://github.com/embeddings-benchmark/mteb/commit/9ef22460196cdc38cdaa1f77bb6b53a759d9c7ec))

- Updated to task metadata of false friend
  ([#374](https://github.com/embeddings-benchmark/mteb/pull/374),
  [`9ef2246`](https://github.com/embeddings-benchmark/mteb/commit/9ef22460196cdc38cdaa1f77bb6b53a759d9c7ec))

### Chores

- Delete contributing.md ([#372](https://github.com/embeddings-benchmark/mteb/pull/372),
  [`bc6a1e3`](https://github.com/embeddings-benchmark/mteb/commit/bc6a1e3f1dea2c7a95d8cdd9ff297687cc37a0aa))

- Move and update contributing.md ([#372](https://github.com/embeddings-benchmark/mteb/pull/372),
  [`bc6a1e3`](https://github.com/embeddings-benchmark/mteb/commit/bc6a1e3f1dea2c7a95d8cdd9ff297687cc37a0aa))

### Documentation

- Updated link in readme
  ([`4cfd94b`](https://github.com/embeddings-benchmark/mteb/commit/4cfd94bbcda17be0f3769d539e582fec5f67f6c1))


## v1.6.7 (2024-04-15)

### Bug Fixes

- Add Japanese Social Network Posts Sentiment Classification Dataset
  ([#358](https://github.com/embeddings-benchmark/mteb/pull/358),
  [`6433c87`](https://github.com/embeddings-benchmark/mteb/commit/6433c8750c677fdac2be16d9917b4fd28a17cc8f))


## v1.6.6 (2024-04-15)

### Bug Fixes

- Add code search (edit of #345) ([#371](https://github.com/embeddings-benchmark/mteb/pull/371),
  [`d1152e8`](https://github.com/embeddings-benchmark/mteb/commit/d1152e8fdac87d96b676702d0088804cc389925c))

- Adding CodeSearchNet ([#371](https://github.com/embeddings-benchmark/mteb/pull/371),
  [`d1152e8`](https://github.com/embeddings-benchmark/mteb/commit/d1152e8fdac87d96b676702d0088804cc389925c))

- Minor fixes to codesearch metadata ([#371](https://github.com/embeddings-benchmark/mteb/pull/371),
  [`d1152e8`](https://github.com/embeddings-benchmark/mteb/commit/d1152e8fdac87d96b676702d0088804cc389925c))


## v1.6.5 (2024-04-15)

### Bug Fixes

- Add 3 tasks for Vietnamese ([#364](https://github.com/embeddings-benchmark/mteb/pull/364),
  [`d054221`](https://github.com/embeddings-benchmark/mteb/commit/d054221087a88c8ebeca3503b337bc0bdecb5a37))


## v1.6.4 (2024-04-15)

### Bug Fixes

- Remove Mr. TyDi Datasets ([#363](https://github.com/embeddings-benchmark/mteb/pull/363),
  [`b015148`](https://github.com/embeddings-benchmark/mteb/commit/b01514892248b68ce7a92ee0e1727c24fe016398))


## v1.6.3 (2024-04-14)

### Bug Fixes

- Add Japanese Question Answering Dataset (JaQuAD) dataset
  ([#352](https://github.com/embeddings-benchmark/mteb/pull/352),
  [`ae6adf4`](https://github.com/embeddings-benchmark/mteb/commit/ae6adf447d50953f8f2c99e06781e75b3a06514e))

### Documentation

- Updated examples in docs (adding a dataset)
  ([#353](https://github.com/embeddings-benchmark/mteb/pull/353),
  [`9b63f50`](https://github.com/embeddings-benchmark/mteb/commit/9b63f50694196666b75c5ccc0bd9ce8dc3206fa7))

- Updated examples in documentation ([#353](https://github.com/embeddings-benchmark/mteb/pull/353),
  [`9b63f50`](https://github.com/embeddings-benchmark/mteb/commit/9b63f50694196666b75c5ccc0bd9ce8dc3206fa7))

- Updated examples in documentation ([#351](https://github.com/embeddings-benchmark/mteb/pull/351),
  [`6832cf0`](https://github.com/embeddings-benchmark/mteb/commit/6832cf02b34d7959986dd36d18bf496a5b699f74))

- Updated taskmetadata to better describe edge cases
  ([#353](https://github.com/embeddings-benchmark/mteb/pull/353),
  [`9b63f50`](https://github.com/embeddings-benchmark/mteb/commit/9b63f50694196666b75c5ccc0bd9ce8dc3206fa7))


## v1.6.2 (2024-04-12)

### Bug Fixes

- Added Hindi discourse dataset ([#346](https://github.com/embeddings-benchmark/mteb/pull/346),
  [`a55ae5f`](https://github.com/embeddings-benchmark/mteb/commit/a55ae5f5b9c722d861faf5ff8c33c5dd4cb4f598))

- Removing bitextmining tasks from fr script
  ([#341](https://github.com/embeddings-benchmark/mteb/pull/341),
  [`86ad02d`](https://github.com/embeddings-benchmark/mteb/commit/86ad02d2d991b5786c9cad7922713f2e36c4831d))

### Documentation

- Added missing points for #214 ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added point for #197 ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added points for #116 ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added points for #137 polish ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added points for #210 (korean) ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added points for #224 ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added points for #27 (spanish) ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))

- Added points for previous submissions
  ([#344](https://github.com/embeddings-benchmark/mteb/pull/344),
  [`9dbf500`](https://github.com/embeddings-benchmark/mteb/commit/9dbf500c7bb3d427345549f7698e15482cf024c7))


## v1.6.1 (2024-04-11)

### Bug Fixes

- Added json files to pyproject.toml ([#340](https://github.com/embeddings-benchmark/mteb/pull/340),
  [`17c809d`](https://github.com/embeddings-benchmark/mteb/commit/17c809d219f95b8570ab044a60b7c56d0ac5b92c))

- Missing json and updated tests to not run in editable mode
  ([#340](https://github.com/embeddings-benchmark/mteb/pull/340),
  [`17c809d`](https://github.com/embeddings-benchmark/mteb/commit/17c809d219f95b8570ab044a60b7c56d0ac5b92c))

### Continuous Integration

- Avoid using -e when installing for tests
  ([#340](https://github.com/embeddings-benchmark/mteb/pull/340),
  [`17c809d`](https://github.com/embeddings-benchmark/mteb/commit/17c809d219f95b8570ab044a60b7c56d0ac5b92c))

### Documentation

- Update mmteb ([#338](https://github.com/embeddings-benchmark/mteb/pull/338),
  [`bee4244`](https://github.com/embeddings-benchmark/mteb/commit/bee424410a62b4602db70062a869302ae6faf068))


## v1.6.0 (2024-04-10)

### Bug Fixes

- Add trusting of remote code to remove warning
  ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Added corrections from review ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Added formatting ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Added initial language code suggestion
  ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Changed folder structure to iso 639-3 codes
  ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Reran linter after merge ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Trust remote code the flores dataset
  ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Updated all language tags ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Updated languages for newly added datasets
  ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

### Documentation

- Added point for language rewrite ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Added points for new annotations ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

- Update points.md ([#337](https://github.com/embeddings-benchmark/mteb/pull/337),
  [`2a3f9e6`](https://github.com/embeddings-benchmark/mteb/commit/2a3f9e6b5db55b7708d2441c78c2dcd47b3ef154))

- Updated task metadata description ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))

### Features

- Added new language code standard ([#326](https://github.com/embeddings-benchmark/mteb/pull/326),
  [`f0daece`](https://github.com/embeddings-benchmark/mteb/commit/f0daece8d8c6491a9d06153763f0d76c2773e251))


## v1.5.6 (2024-04-10)

### Bug Fixes

- Added medical qa dataset ([#333](https://github.com/embeddings-benchmark/mteb/pull/333),
  [`80acc3e`](https://github.com/embeddings-benchmark/mteb/commit/80acc3e39ee6fc27458275b3c2a25e0e13550c44))

### Documentation

- Add points and affiliation for MartinBernstorff
  ([#335](https://github.com/embeddings-benchmark/mteb/pull/335),
  [`2903cb4`](https://github.com/embeddings-benchmark/mteb/commit/2903cb4a738bdae0a5863059c85de97a17c95074))

- Update points.md ([#335](https://github.com/embeddings-benchmark/mteb/pull/335),
  [`2903cb4`](https://github.com/embeddings-benchmark/mteb/commit/2903cb4a738bdae0a5863059c85de97a17c95074))


## v1.5.5 (2024-04-09)

### Bug Fixes

- Improve logging when the revision is None
  ([#329](https://github.com/embeddings-benchmark/mteb/pull/329),
  [`404587b`](https://github.com/embeddings-benchmark/mteb/commit/404587b81d63d666cb121d0273ce0be1d0e70526))


## v1.5.4 (2024-04-08)

### Bug Fixes

- Added missing dataset_transform to multitask task
  ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

- Correctly fix pawsX ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

- Fixes the PawsX datasets ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

- Flores clustering ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

- Mulitple dataset fixes ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

- Multiple dataset fixes ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

- Remove time of run (as it does not relate to the model itself). Time of run should be on the
  dataset results ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))

### Documentation

- Updated points ([#328](https://github.com/embeddings-benchmark/mteb/pull/328),
  [`84408f7`](https://github.com/embeddings-benchmark/mteb/commit/84408f7bcd422fc59febe19818877299168b08b8))


## v1.5.3 (2024-04-08)

### Bug Fixes

- Added English news classification dataset
  ([#323](https://github.com/embeddings-benchmark/mteb/pull/323),
  [`4d21807`](https://github.com/embeddings-benchmark/mteb/commit/4d21807a161aab452ede725144ce4f4c802a8da9))

### Documentation

- Added point for SEB ([#318](https://github.com/embeddings-benchmark/mteb/pull/318),
  [`ca64fc7`](https://github.com/embeddings-benchmark/mteb/commit/ca64fc71ffd1eeecba97bc9a346f28748df5fca7))

- Added points for seb ([#318](https://github.com/embeddings-benchmark/mteb/pull/318),
  [`ca64fc7`](https://github.com/embeddings-benchmark/mteb/commit/ca64fc71ffd1eeecba97bc9a346f28748df5fca7))

- Small fixes in readme.md ([#317](https://github.com/embeddings-benchmark/mteb/pull/317),
  [`ede12c8`](https://github.com/embeddings-benchmark/mteb/commit/ede12c8705ca42b38ed2dc167acaa4250b816a34))


## v1.5.2 (2024-04-04)

### Bug Fixes

- Minor fixes to metadata ([#315](https://github.com/embeddings-benchmark/mteb/pull/315),
  [`e0eddf9`](https://github.com/embeddings-benchmark/mteb/commit/e0eddf9f0436bda44ffa71134e651d3c70829dc0))

- Updated wrong metadata ([#315](https://github.com/embeddings-benchmark/mteb/pull/315),
  [`e0eddf9`](https://github.com/embeddings-benchmark/mteb/commit/e0eddf9f0436bda44ffa71134e651d3c70829dc0))


## v1.5.1 (2024-04-03)

### Bug Fixes

- Added suggestions from the review ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Added tests for checking datasets ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Applied formatter ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Fixed hf_hub_name for WikiCitiesClustering
  ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Reuploaded scandeval datasets ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Sped up async test for whether datasets exist
  ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Updated hf references and revisions to multiple datasets
  ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

- Updated revisions ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))

### Features

- Added tests which validated that datasets are available
  ([#307](https://github.com/embeddings-benchmark/mteb/pull/307),
  [`8d804f4`](https://github.com/embeddings-benchmark/mteb/commit/8d804f4956ea66b68b25c14ab05bc5ad2e102ff7))


## v1.5.0 (2024-04-02)

### Features

- Allow extending the load_dataset parameters in custom tasks inheriting AbsTask
  ([#299](https://github.com/embeddings-benchmark/mteb/pull/299),
  [`953780d`](https://github.com/embeddings-benchmark/mteb/commit/953780dd680309cbae78f6543c005965ad9caf01))

### Testing

- Adding very high level test ([#299](https://github.com/embeddings-benchmark/mteb/pull/299),
  [`953780d`](https://github.com/embeddings-benchmark/mteb/commit/953780dd680309cbae78f6543c005965ad9caf01))


## v1.4.1 (2024-04-01)

### Bug Fixes

- Fixed hf_hub_name for WikiCitiesClustering
  ([#305](https://github.com/embeddings-benchmark/mteb/pull/305),
  [`b447235`](https://github.com/embeddings-benchmark/mteb/commit/b447235a18d2fb7261c6cb6cbb214e0bbd15452c))

- Hf_hub_name for WikiCitiesClustering
  ([#305](https://github.com/embeddings-benchmark/mteb/pull/305),
  [`b447235`](https://github.com/embeddings-benchmark/mteb/commit/b447235a18d2fb7261c6cb6cbb214e0bbd15452c))


## v1.4.0 (2024-04-01)

### Continuous Integration

- Added windows to test suite ([#292](https://github.com/embeddings-benchmark/mteb/pull/292),
  [`fc0e105`](https://github.com/embeddings-benchmark/mteb/commit/fc0e105e737352e0afd3a144717e7129654f8c90))

### Features

- Added windows support by replacing pytrec-eval with pytrec-eval-terrier
  ([#292](https://github.com/embeddings-benchmark/mteb/pull/292),
  [`fc0e105`](https://github.com/embeddings-benchmark/mteb/commit/fc0e105e737352e0afd3a144717e7129654f8c90))

- Changed to pytrec-eval-terrier to add support for windows installs
  ([#292](https://github.com/embeddings-benchmark/mteb/pull/292),
  [`fc0e105`](https://github.com/embeddings-benchmark/mteb/commit/fc0e105e737352e0afd3a144717e7129654f8c90))


## v1.3.4 (2024-04-01)

### Bug Fixes

- Update MindSmallReranking.py to have the correct hf reference
  ([#303](https://github.com/embeddings-benchmark/mteb/pull/303),
  [`102e24e`](https://github.com/embeddings-benchmark/mteb/commit/102e24e093c4414e3ecef49cb9d0ba94995f2eb3))


## v1.3.3 (2024-03-31)

### Bug Fixes

- Fixed bug introduced in TatoebaBitextMining causing it to use a different dataset
  ([#297](https://github.com/embeddings-benchmark/mteb/pull/297),
  [`d0549a3`](https://github.com/embeddings-benchmark/mteb/commit/d0549a35f9697663a3e4bc6bdbe5990977d5c2e2))

- Fixed mispecified rev. id for datasets
  ([#298](https://github.com/embeddings-benchmark/mteb/pull/298),
  [`e1ae0d3`](https://github.com/embeddings-benchmark/mteb/commit/e1ae0d367eda70e6be31f0a43e20aed7cf75fedd))

- Fixed wrong rev. id for ToxicConversationsClassification
  ([#298](https://github.com/embeddings-benchmark/mteb/pull/298),
  [`e1ae0d3`](https://github.com/embeddings-benchmark/mteb/commit/e1ae0d367eda70e6be31f0a43e20aed7cf75fedd))

- Fixed wrong rev. id with RedditClusteringP2P
  ([#298](https://github.com/embeddings-benchmark/mteb/pull/298),
  [`e1ae0d3`](https://github.com/embeddings-benchmark/mteb/commit/e1ae0d367eda70e6be31f0a43e20aed7cf75fedd))

### Continuous Integration

- Removed changelog ([#290](https://github.com/embeddings-benchmark/mteb/pull/290),
  [`6821d23`](https://github.com/embeddings-benchmark/mteb/commit/6821d2324f41fed2bb686eedf074e4dc6803ece6))

### Documentation

- Added information related to the automatic release
  ([#290](https://github.com/embeddings-benchmark/mteb/pull/290),
  [`6821d23`](https://github.com/embeddings-benchmark/mteb/commit/6821d2324f41fed2bb686eedf074e4dc6803ece6))

- Minor additions to contributing guidelines
  ([#290](https://github.com/embeddings-benchmark/mteb/pull/290),
  [`6821d23`](https://github.com/embeddings-benchmark/mteb/commit/6821d2324f41fed2bb686eedf074e4dc6803ece6))

- Removed test-parallel from docs ([#290](https://github.com/embeddings-benchmark/mteb/pull/290),
  [`6821d23`](https://github.com/embeddings-benchmark/mteb/commit/6821d2324f41fed2bb686eedf074e4dc6803ece6))


## v1.3.2 (2024-03-29)

### Bug Fixes

- Added tasks from SEB ([#287](https://github.com/embeddings-benchmark/mteb/pull/287),
  [`39cff49`](https://github.com/embeddings-benchmark/mteb/commit/39cff490157ae87d1cf62c77022f325be729bf04))

- Ran linting ([#287](https://github.com/embeddings-benchmark/mteb/pull/287),
  [`39cff49`](https://github.com/embeddings-benchmark/mteb/commit/39cff490157ae87d1cf62c77022f325be729bf04))

### Documentation

- Fix link ([#287](https://github.com/embeddings-benchmark/mteb/pull/287),
  [`39cff49`](https://github.com/embeddings-benchmark/mteb/commit/39cff490157ae87d1cf62c77022f325be729bf04))

- Update links in README.md ([#296](https://github.com/embeddings-benchmark/mteb/pull/296),
  [`76056b5`](https://github.com/embeddings-benchmark/mteb/commit/76056b5ba92dcbfe32d629897ab6d5db3a0861c4))


## v1.3.1 (2024-03-26)

### Bug Fixes

- Updated version in transition to semantic release ci
  ([`238ab82`](https://github.com/embeddings-benchmark/mteb/commit/238ab825e9b221c363589eed89273481e058c50f))


## v1.3.0 (2024-03-26)

### Bug Fixes

- Added missing init files
  ([`b1c78c1`](https://github.com/embeddings-benchmark/mteb/commit/b1c78c1121fbd488d62d1385d6f000d3f3b46ef4))

- Added sizes to the metadata
  ([`a16eb07`](https://github.com/embeddings-benchmark/mteb/commit/a16eb07da1d1a6d8380683e9fa11df3244fae87b))

- Added sizes to the metadata ([#276](https://github.com/embeddings-benchmark/mteb/pull/276),
  [`cd4a012`](https://github.com/embeddings-benchmark/mteb/commit/cd4a012271463b89db7a8ec9ca298a975805988d))

- Dead link in readme
  ([`ecbb776`](https://github.com/embeddings-benchmark/mteb/commit/ecbb776fba460c531f09e7b0ce986f075f2b665a))

- Updated form to be parsed correctly
  ([`a16eb07`](https://github.com/embeddings-benchmark/mteb/commit/a16eb07da1d1a6d8380683e9fa11df3244fae87b))

- Updated form to be parsed correctly
  ([#276](https://github.com/embeddings-benchmark/mteb/pull/276),
  [`cd4a012`](https://github.com/embeddings-benchmark/mteb/commit/cd4a012271463b89db7a8ec9ca298a975805988d))

- Updated form to be parsed correctly
  ([`c0dc49a`](https://github.com/embeddings-benchmark/mteb/commit/c0dc49a6b99f4d8136b7ec46c49563d7e1b866db))

### Build System

- **deps**: Update pyproject.toml ([#260](https://github.com/embeddings-benchmark/mteb/pull/260),
  [`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

- **deps**: Update TaskMetadata.py and pyproject.toml
  ([#260](https://github.com/embeddings-benchmark/mteb/pull/260),
  [`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

### Chores

- Delete ddisco.py, ddisco.test.tsv and ddisco.train.tsv
  ([`d46d0f5`](https://github.com/embeddings-benchmark/mteb/commit/d46d0f5281d5a9036501aca437e9c76480dc8885))

### Continuous Integration

- Added in newer workflow
  ([`023e881`](https://github.com/embeddings-benchmark/mteb/commit/023e8817f108a76718fc37f7c8937e000de56786))

- Added tests ([#282](https://github.com/embeddings-benchmark/mteb/pull/282),
  [`6675bb8`](https://github.com/embeddings-benchmark/mteb/commit/6675bb8668ff17ca8cf3cce2703f3ebf17795bfc))

- Avoid specifying tests folder as it causes issuew ith tests
  ([#246](https://github.com/embeddings-benchmark/mteb/pull/246),
  [`0048878`](https://github.com/embeddings-benchmark/mteb/commit/0048878deba9f57147c3696dcb89ade098c90376))

- Disable changelog
  ([`b7d3cde`](https://github.com/embeddings-benchmark/mteb/commit/b7d3cde561200264d74e592a678f0dea2eb68129))

- Moved release to the correct folder
  ([`7f56c1a`](https://github.com/embeddings-benchmark/mteb/commit/7f56c1a7d2eb2fab6eb028291d85054727c650d1))

- Moved release to the correct folder
  ([`b4fa85a`](https://github.com/embeddings-benchmark/mteb/commit/b4fa85a51374b78b47789557c5467700b859eba5))

- Removed unec. args for test ci ([#246](https://github.com/embeddings-benchmark/mteb/pull/246),
  [`0048878`](https://github.com/embeddings-benchmark/mteb/commit/0048878deba9f57147c3696dcb89ade098c90376))

- Renamed test job and workflow ([#282](https://github.com/embeddings-benchmark/mteb/pull/282),
  [`6675bb8`](https://github.com/embeddings-benchmark/mteb/commit/6675bb8668ff17ca8cf3cce2703f3ebf17795bfc))

- Update ci to use make commands ([#246](https://github.com/embeddings-benchmark/mteb/pull/246),
  [`0048878`](https://github.com/embeddings-benchmark/mteb/commit/0048878deba9f57147c3696dcb89ade098c90376))

- Updated ci to use make install ([#260](https://github.com/embeddings-benchmark/mteb/pull/260),
  [`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

- Updated ci to use make install
  ([`8a758bc`](https://github.com/embeddings-benchmark/mteb/commit/8a758bce00a6bc64dd4f0dab98f5bc3e0c683f46))

### Documentation

- Add dataset schemas ([#255](https://github.com/embeddings-benchmark/mteb/pull/255),
  [`c3ce1ac`](https://github.com/embeddings-benchmark/mteb/commit/c3ce1ac8ac92baf9a7481c30d476b45e3ec36786))

- Add development installation instructions
  ([#246](https://github.com/embeddings-benchmark/mteb/pull/246),
  [`0048878`](https://github.com/embeddings-benchmark/mteb/commit/0048878deba9f57147c3696dcb89ade098c90376))

- Typos in readme ([#268](https://github.com/embeddings-benchmark/mteb/pull/268),
  [`aa9234c`](https://github.com/embeddings-benchmark/mteb/commit/aa9234cc24f6dd3408961895d092ee019551fab2))

- Update AbsTaskClassification.py document schema for classification task
  ([#255](https://github.com/embeddings-benchmark/mteb/pull/255),
  [`c3ce1ac`](https://github.com/embeddings-benchmark/mteb/commit/c3ce1ac8ac92baf9a7481c30d476b45e3ec36786))

- Updated make file with new dependencies
  ([#246](https://github.com/embeddings-benchmark/mteb/pull/246),
  [`0048878`](https://github.com/embeddings-benchmark/mteb/commit/0048878deba9f57147c3696dcb89ade098c90376))

### Features

- Bump version again
  ([`294ab91`](https://github.com/embeddings-benchmark/mteb/commit/294ab910f6aa4099c0de8c9f91dbee38efd91aab))

- Bump version again
  ([`acf68c7`](https://github.com/embeddings-benchmark/mteb/commit/acf68c799133d390baba15cdf87f81c844c5a682))

- Updating version
  ([`caee2e9`](https://github.com/embeddings-benchmark/mteb/commit/caee2e9451999633476fb3305fb3fdc928ec9f0b))

### Refactoring

- Add metadata basemodel ([#260](https://github.com/embeddings-benchmark/mteb/pull/260),
  [`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

- Add TaskMetadata and first example ([#260](https://github.com/embeddings-benchmark/mteb/pull/260),
  [`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

- Rename description to metadata dict
  ([#260](https://github.com/embeddings-benchmark/mteb/pull/260),
  [`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

### Testing

- Update test_all_abstasks.py ([#249](https://github.com/embeddings-benchmark/mteb/pull/249),
  [`236614a`](https://github.com/embeddings-benchmark/mteb/commit/236614add5de4e848bc0f1db8ad997f451ae2906))


## v1.2.0 (2024-03-06)

### Bug Fixes

- Formatting of xpqa dataset ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Get scores from label column ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Nested corpus ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Typo in data loading ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Update revision id ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

### Code Style

- Add missing eof empty line ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

### Features

- Add miracl as retrieval task ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Add miracl reranking task for spanish
  ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Add revision ids for hf datasets ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Add stses task ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Add two clustering datasets ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Add xmarket es dataset ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Allow multlingual reranking tasks ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Import classes ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Make miraclreranking multilingual ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Make xmarket retrieval task multilingual
  ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Mintaka and xpqa retrieval tasks ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Update revision hash ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Use hf repo with all reranking langs
  ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

### Refactoring

- Add constant for language ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Add revision to data loading ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Cmon man ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Flores dataset ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Get lang from description ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Go back to monolingual tasks ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Loading logic ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Make mintaka into multilingual task
  ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Make miracl retrieval multilingual ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Make xpqa retrieval multilingual ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Multilingual task import ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Remove patool ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Remove unused import ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Rename miraclretrieval ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Rename xmarket ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Try out multilingual task ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Turn spanish tasks multilingual ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Use description for language ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))

- Use multilingual dataset ([#227](https://github.com/embeddings-benchmark/mteb/pull/227),
  [`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))


## v1.1.2 (2024-02-16)

### Bug Fixes

- Double import; xmarket name ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Gerdalir dataset ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Lang from en to de ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Pass parallel_retrieval kwarg to use DenseRetrievalParallelExactSearch
  ([`19b8f66`](https://github.com/embeddings-benchmark/mteb/commit/19b8f6619f07dfd95860f43f9376af230978f447))

- Remove debugging print statement
  ([`d292d93`](https://github.com/embeddings-benchmark/mteb/commit/d292d937ceedb5d137537b2c25a0f135d1bb91b9))

- Use test split in MIRACL ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

### Chores

- Add ir datasets to requirements ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Solve merge conflict ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

### Features

- Add german stsbenchmarksts task ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Add miracl reranking task for german
  ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Add revision id ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Update revision id ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Update revision id of wikicitiesclustering task
  ([`fb90c02`](https://github.com/embeddings-benchmark/mteb/commit/fb90c022e11834ae6605f5bbb0a79af701793a96))

### Refactoring

- Cleanup task ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Limit queries to 10k ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Remove WikiCLIR ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Update description of task with limit
  ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

- Update revision id after changes in scores
  ([#214](https://github.com/embeddings-benchmark/mteb/pull/214),
  [`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))


## v1.1.1 (2023-09-20)

### Bug Fixes

- Add missing task-langs attribute ([#152](https://github.com/embeddings-benchmark/mteb/pull/152),
  [`bc22909`](https://github.com/embeddings-benchmark/mteb/commit/bc22909c49284efb0df1d997ac23806694424a94))

- Msmarco-v2 uses dev.tsv, not dev1.tsv
  ([`6908d21`](https://github.com/embeddings-benchmark/mteb/commit/6908d21cfce644140bd70df47df0452c551ee0d0))

- Replaced prints with logging statements
  ([`d7ca378`](https://github.com/embeddings-benchmark/mteb/commit/d7ca3784451873042fb8a3fc2cdf4406f1ab465a))

### Chores

- Removed accidental commits
  ([`d7ca378`](https://github.com/embeddings-benchmark/mteb/commit/d7ca3784451873042fb8a3fc2cdf4406f1ab465a))


## v1.1.0 (2023-07-31)

### Bug Fixes

- Added functionality to raise error
  ([`acb0f59`](https://github.com/embeddings-benchmark/mteb/commit/acb0f59435ee660c266490bfa1db22bd5f19d1d5))

- Removed no as a language
  ([`acb0f59`](https://github.com/embeddings-benchmark/mteb/commit/acb0f59435ee660c266490bfa1db22bd5f19d1d5))

- Updated names
  ([`acb0f59`](https://github.com/embeddings-benchmark/mteb/commit/acb0f59435ee660c266490bfa1db22bd5f19d1d5))


## v1.0.2 (2023-03-28)


## v1.0.1 (2022-11-29)


## v1.0.0 (2022-10-17)


## v0.9.0 (2022-10-13)

- Initial Release
