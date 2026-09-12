# `tencent/EVIE-8B`

# MTEB Mock-Run Results for `tencent/EVIE-8B`

| Task | Modality | Pass | Reason |
| --- | --- | --- | --- |
| MockMultilingualBitextMiningTask | text | ✓ | - |
| MockMultilingualParallelBitextMiningTask | text | ✓ | - |
| MockMultilingualClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockMultilingualClusteringTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockMultilingualClusteringFastTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockMultilingualPairClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilingualRerankingTask | text | ✓ | - |
| MockMultilingualRetrievalTask | text | ✓ | - |
| MockMultilingualSTSTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilingualMultilabelClassification | text | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockMultilingualSummarizationTask | text | ✗ | Dynamo failed to run FX node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(..., size=(15, 4096)), FakeTensor(..., size=(14, 2, 4096))), **{}): got RuntimeError('Expected size for first two dimensions of batch2 tensor to be: [14, 4096] but got: [14, 2].')  from user code:    File "/root/evie/mteb/mteb/similarity_functions.py", line 149, in _cos_sim_core     return a_norm @ b_norm.transpose(0, 1)  Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"  |
| MockMultilingualInstructionRetrieval | text | ✓ | - |
| MockMultilingualInstructionReranking | text | ✓ | - |
| MockBitextMiningTask | text | ✓ | - |
| MockClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockRegressionTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LinearRegression. |
| MockClusteringTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| LegacyMockClusteringFastTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockPairClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockRerankingTask | text | ✓ | - |
| MockRetrievalTask | text | ✓ | - |
| MockSTSTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilabelClassification | text | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockSummarizationTask | text | ✗ | Dynamo failed to run FX node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(..., size=(15, 4096)), FakeTensor(..., size=(14, 2, 4096))), **{}): got RuntimeError('Expected size for first two dimensions of batch2 tensor to be: [14, 4096] but got: [14, 2].')  from user code:    File "/root/evie/mteb/mteb/similarity_functions.py", line 149, in _cos_sim_core     return a_norm @ b_norm.transpose(0, 1)  Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"  |
| MockInstructionRetrieval | text | ✓ | - |
| MockInstructionReranking | text | ✓ | - |
| MockRetrievalDialogTask | text | ✓ | - |
| MockTextZeroShotClassification | text | ✓ | - |
| MockAny2AnyRetrievalI2T | image, text | ✓ | - |
| MockAny2AnyRetrievalT2I | image, text | ✓ | - |
| MockVisionCentricQA | image, text | ✓ | - |
| MockImageClassification | image | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockImageClustering | image | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockImageTextPairClassification | image, text | ✗ | mat1 and mat2 shapes cannot be multiplied (1x307200 and 61440x1) |
| MockVisualSTS | image | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockZeroShotClassification | image, text | ✗ | The size of tensor a (2) must match the size of tensor b (4096) at non-singleton dimension 0 |
| MockImageMultilabelClassification | image | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockMultilingualImageClassification | image | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockMultilingualImageTextPairClassification | image, text | ✗ | mat1 and mat2 shapes cannot be multiplied (1x307200 and 61440x1) |
| MockMultilingualVisionCentricQA | image, text | ✓ | - |
| MockImageClusteringFastTask | image | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockImageRegressionTask | image | ✗ | Found array with dim 3, while dim <= 2 is required by LinearRegression. |
| MockPairImageClassificationTask | image | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |

## Summary by Modality

| Pass      | Modality | Failures |
| --------- | -------- | -------- |
| ✗ (17/35) | text     | MockMultilingualClassificationTask, MockMultilingualClusteringTask, MockMultilingualClusteringFastTask, MockMultilingualPairClassificationTask, MockMultilingualSTSTask, MockMultilingualMultilabelClassification, MockMultilingualSummarizationTask, MockClassificationTask, MockRegressionTask, MockClusteringTask, LegacyMockClusteringFastTask, MockPairClassificationTask, MockSTSTask, MockMultilabelClassification, MockSummarizationTask, MockImageTextPairClassification, MockZeroShotClassification, MockMultilingualImageTextPairClassification |
| ✗ (4/15)  | image    | MockImageClassification, MockImageClustering, MockImageTextPairClassification, MockVisualSTS, MockZeroShotClassification, MockImageMultilabelClassification, MockMultilingualImageClassification, MockMultilingualImageTextPairClassification, MockImageClusteringFastTask, MockImageRegressionTask, MockPairImageClassificationTask |
| skipped   | audio    |  |
| skipped   | video    |  |

# `tencent/EVIE-4.5B`

# MTEB Mock-Run Results for `tencent/EVIE-4.5B`

| Task | Modality | Pass | Reason |
| --- | --- | --- | --- |
| MockMultilingualBitextMiningTask | text | ✓ | - |
| MockMultilingualParallelBitextMiningTask | text | ✓ | - |
| MockMultilingualClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockMultilingualClusteringTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockMultilingualClusteringFastTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockMultilingualPairClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilingualRerankingTask | text | ✓ | - |
| MockMultilingualRetrievalTask | text | ✓ | - |
| MockMultilingualSTSTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilingualMultilabelClassification | text | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockMultilingualSummarizationTask | text | ✗ | Dynamo failed to run FX node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(..., size=(15, s75)), FakeTensor(..., size=(14, 2, s36))), **{}): got RuntimeError('Expected size for first two dimensions of batch2 tensor to be: [14, s75] but got: [14, 2].')  from user code:    File "/root/evie/mteb/mteb/similarity_functions.py", line 149, in _cos_sim_core     return a_norm @ b_norm.transpose(0, 1)  Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"  |
| MockMultilingualInstructionRetrieval | text | ✓ | - |
| MockMultilingualInstructionReranking | text | ✓ | - |
| MockBitextMiningTask | text | ✓ | - |
| MockClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockRegressionTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LinearRegression. |
| MockClusteringTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| LegacyMockClusteringFastTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockPairClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockRerankingTask | text | ✓ | - |
| MockRetrievalTask | text | ✓ | - |
| MockSTSTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilabelClassification | text | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockSummarizationTask | text | ✗ | Dynamo failed to run FX node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(..., size=(15, s75)), FakeTensor(..., size=(14, 2, s36))), **{}): got RuntimeError('Expected size for first two dimensions of batch2 tensor to be: [14, s75] but got: [14, 2].')  from user code:    File "/root/evie/mteb/mteb/similarity_functions.py", line 149, in _cos_sim_core     return a_norm @ b_norm.transpose(0, 1)  Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"  |
| MockInstructionRetrieval | text | ✓ | - |
| MockInstructionReranking | text | ✓ | - |
| MockRetrievalDialogTask | text | ✓ | - |
| MockTextZeroShotClassification | text | ✓ | - |
| MockAny2AnyRetrievalI2T | image, text | ✓ | - |
| MockAny2AnyRetrievalT2I | image, text | ✓ | - |
| MockVisionCentricQA | image, text | ✓ | - |
| MockImageClassification | image | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockImageClustering | image | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockImageTextPairClassification | image, text | ✗ | mat1 and mat2 shapes cannot be multiplied (1x153600 and 30720x1) |
| MockVisualSTS | image | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockZeroShotClassification | image, text | ✗ | The size of tensor a (2) must match the size of tensor b (2048) at non-singleton dimension 0 |
| MockImageMultilabelClassification | image | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockMultilingualImageClassification | image | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockMultilingualImageTextPairClassification | image, text | ✗ | mat1 and mat2 shapes cannot be multiplied (1x153600 and 30720x1) |
| MockMultilingualVisionCentricQA | image, text | ✓ | - |
| MockImageClusteringFastTask | image | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockImageRegressionTask | image | ✗ | Found array with dim 3, while dim <= 2 is required by LinearRegression. |
| MockPairImageClassificationTask | image | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |

## Summary by Modality

| Pass      | Modality | Failures |
| --------- | -------- | -------- |
| ✗ (17/35) | text     | MockMultilingualClassificationTask, MockMultilingualClusteringTask, MockMultilingualClusteringFastTask, MockMultilingualPairClassificationTask, MockMultilingualSTSTask, MockMultilingualMultilabelClassification, MockMultilingualSummarizationTask, MockClassificationTask, MockRegressionTask, MockClusteringTask, LegacyMockClusteringFastTask, MockPairClassificationTask, MockSTSTask, MockMultilabelClassification, MockSummarizationTask, MockImageTextPairClassification, MockZeroShotClassification, MockMultilingualImageTextPairClassification |
| ✗ (4/15)  | image    | MockImageClassification, MockImageClustering, MockImageTextPairClassification, MockVisualSTS, MockZeroShotClassification, MockImageMultilabelClassification, MockMultilingualImageClassification, MockMultilingualImageTextPairClassification, MockImageClusteringFastTask, MockImageRegressionTask, MockPairImageClassificationTask |
| skipped   | audio    |  |
| skipped   | video    |  |
