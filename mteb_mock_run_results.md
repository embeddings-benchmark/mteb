# Last-prompt encoder live mock verification

Both reports were generated on 2026-10-26 using MTEB main source
`3edfd0ab5a7ce92d5198b59392bc1f2d112a9b2b` (2.26.24) imported with
the existing MTEB 2.24.14 evaluation environment dependencies. Each model
passed all 28 compatible text tasks. The server reported vLLM 0.30.0.
These checks validate integration, not historical evaluation runtime identity.

## qwen

# MTEB Mock-Run Results for `Qwen/Qwen3.5-4B-e5v-last-prompt`

| Task | Modality | Pass | Reason |
| --- | --- | --- | --- |
| MockMultilingualBitextMiningTask | text | ✓ | - |
| MockMultilingualParallelBitextMiningTask | text | ✓ | - |
| MockMultilingualClassificationTask | text | ✓ | - |
| MockMultilingualClusteringTask | text | ✓ | - |
| MockMultilingualClusteringFastTask | text | ✓ | - |
| MockMultilingualPairClassificationTask | text | ✓ | - |
| MockMultilingualRerankingTask | text | ✓ | - |
| MockMultilingualRetrievalTask | text | ✓ | - |
| MockMultilingualSTSTask | text | ✓ | - |
| MockMultilingualMultilabelClassification | text | ✓ | - |
| MockMultilingualSummarizationTask | text | ✓ | - |
| MockMultilingualInstructionRetrieval | text | ✓ | - |
| MockMultilingualInstructionReranking | text | ✓ | - |
| MockBitextMiningTask | text | ✓ | - |
| MockClassificationTask | text | ✓ | - |
| MockRegressionTask | text | ✓ | - |
| MockClusteringTask | text | ✓ | - |
| LegacyMockClusteringFastTask | text | ✓ | - |
| MockPairClassificationTask | text | ✓ | - |
| MockRerankingTask | text | ✓ | - |
| MockRetrievalTask | text | ✓ | - |
| MockSTSTask | text | ✓ | - |
| MockMultilabelClassification | text | ✓ | - |
| MockSummarizationTask | text | ✓ | - |
| MockInstructionRetrieval | text | ✓ | - |
| MockInstructionReranking | text | ✓ | - |
| MockRetrievalDialogTask | text | ✓ | - |
| MockTextZeroShotClassification | text | ✓ | - |

## Summary by Modality

| Pass      | Modality | Failures |
| --------- | -------- | -------- |
| ✓ (28/28) | text     |  |
| skipped   | image    |  |
| skipped   | audio    |  |
| skipped   | video    |  |

## nuextract

# MTEB Mock-Run Results for `numind/NuExtract3-revised-last-prompt`

| Task | Modality | Pass | Reason |
| --- | --- | --- | --- |
| MockMultilingualBitextMiningTask | text | ✓ | - |
| MockMultilingualParallelBitextMiningTask | text | ✓ | - |
| MockMultilingualClassificationTask | text | ✓ | - |
| MockMultilingualClusteringTask | text | ✓ | - |
| MockMultilingualClusteringFastTask | text | ✓ | - |
| MockMultilingualPairClassificationTask | text | ✓ | - |
| MockMultilingualRerankingTask | text | ✓ | - |
| MockMultilingualRetrievalTask | text | ✓ | - |
| MockMultilingualSTSTask | text | ✓ | - |
| MockMultilingualMultilabelClassification | text | ✓ | - |
| MockMultilingualSummarizationTask | text | ✓ | - |
| MockMultilingualInstructionRetrieval | text | ✓ | - |
| MockMultilingualInstructionReranking | text | ✓ | - |
| MockBitextMiningTask | text | ✓ | - |
| MockClassificationTask | text | ✓ | - |
| MockRegressionTask | text | ✓ | - |
| MockClusteringTask | text | ✓ | - |
| LegacyMockClusteringFastTask | text | ✓ | - |
| MockPairClassificationTask | text | ✓ | - |
| MockRerankingTask | text | ✓ | - |
| MockRetrievalTask | text | ✓ | - |
| MockSTSTask | text | ✓ | - |
| MockMultilabelClassification | text | ✓ | - |
| MockSummarizationTask | text | ✓ | - |
| MockInstructionRetrieval | text | ✓ | - |
| MockInstructionReranking | text | ✓ | - |
| MockRetrievalDialogTask | text | ✓ | - |
| MockTextZeroShotClassification | text | ✓ | - |

## Summary by Modality

| Pass      | Modality | Failures |
| --------- | -------- | -------- |
| ✓ (28/28) | text     |  |
| skipped   | image    |  |
| skipped   | audio    |  |
| skipped   | video    |  |
