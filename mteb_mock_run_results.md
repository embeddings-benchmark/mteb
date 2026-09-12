**Native MTEB integration with the installed ngnn-encoder 0.1.0 wheel and actual anonymous Hugging Face downloads. Provider embeddings are synthetic; no new paid OpenAI calls or model-quality claim.**

HF artifact commit: `bab7d30011438e52f22be540067c73ca37f462eb`. Canonical MTEB revision remains `d6969c26400944d4f5200ebddfdc04a083fd7b75`.

The previous [real-provider verification](https://github.com/steffen181/frozen-ngnn-api-modesl/blob/738ce65556d355b130f79e501bdc80af6bfd9f9c/verification/live_mock_run.md) covers the unchanged inference core and remains available separately.

# MTEB Mock-Run Results for `steffen-negabo/ngnn-general-encoder-v1`

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