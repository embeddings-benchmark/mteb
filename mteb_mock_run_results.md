**Real OpenAI provider verification with the public frozen NGNN compressor.**

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

Submission revision: `d6969c26400944d4f5200ebddfdc04a083fd7b75`.
The live run used local evaluation label `ngnn_general_encoder_singleton_e55ba679_20260906`.
Only the registration identifier was renamed to the immutable public source
commit for the results repository revision format. Inference and weights
are unchanged; see [revision mapping and hashes](https://github.com/steffen181/frozen-ngnn-api-modesl/blob/main/verification/revision_alias.json).
