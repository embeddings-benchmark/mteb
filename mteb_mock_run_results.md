# MTEB Mock-Run Results for `nanovdr/ColNanoVDR-Q-Ettin150M-ColVec4B-640-ML`

| Task | Modality | Pass | Reason |
| --- | --- | --- | --- |
| MockMultilingualBitextMiningTask | text | ✓ | - |
| MockMultilingualParallelBitextMiningTask | text | ✓ | - |
| MockMultilingualClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockMultilingualClusteringTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockMultilingualClusteringFastTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockMultilingualPairClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilingualRerankingTask | text | ✗ | 'image' |
| MockMultilingualRetrievalTask | text | ✗ | 'image' |
| MockMultilingualSTSTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilingualMultilabelClassification | text | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockMultilingualSummarizationTask | text | ✗ | RuntimeError when making fake tensor call   Explanation: Dynamo failed to run FX node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(..., size=(5, 640)), FakeTensor(..., size=(4, 2, 640))), **{}): got RuntimeError('Expected size for first two dimensions of batch2 tensor to be: [4, 640] but got: [4, 2].')   Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.     Developer debug context:    For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html  from user code:    File "/scratch/elec/t412-humanmotion/liuz16/colnanovdr/mteb_work/mteb/mteb/similarity_functions.py", line 149, in _cos_sim_core     return a_norm @ b_norm.transpose(0, 1)  Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"  |
| MockMultilingualInstructionRetrieval | text | ✗ | 'image' |
| MockMultilingualInstructionReranking | text | ✗ | 'image' |
| MockBitextMiningTask | text | ✓ | - |
| MockClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LogisticRegression. |
| MockRegressionTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by LinearRegression. |
| MockClusteringTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| LegacyMockClusteringFastTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by MiniBatchKMeans. |
| MockPairClassificationTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockRerankingTask | text | ✗ | 'image' |
| MockRetrievalTask | text | ✗ | 'image' |
| MockSTSTask | text | ✗ | Found array with dim 3, while dim <= 2 is required by check_pairwise_arrays. |
| MockMultilabelClassification | text | ✗ | Found array with dim 3, while dim <= 2 is required by KNeighborsClassifier. |
| MockSummarizationTask | text | ✗ | RuntimeError when making fake tensor call   Explanation: Dynamo failed to run FX node with fake tensors: call_function <built-in function matmul>(*(FakeTensor(..., size=(5, 640)), FakeTensor(..., size=(4, 2, 640))), **{}): got RuntimeError('Expected size for first two dimensions of batch2 tensor to be: [4, 640] but got: [4, 2].')   Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.     Developer debug context:    For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html  from user code:    File "/scratch/elec/t412-humanmotion/liuz16/colnanovdr/mteb_work/mteb/mteb/similarity_functions.py", line 149, in _cos_sim_core     return a_norm @ b_norm.transpose(0, 1)  Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"  |
| MockInstructionRetrieval | text | ✗ | 'image' |
| MockInstructionReranking | text | ✗ | 'image' |
| MockRetrievalDialogTask | text | ✗ | 'image' |
| MockTextZeroShotClassification | text | ✓ | - |
| MockAny2AnyRetrievalI2T | image, text | ✗ | ColNanoVDR only supports text queries, but task 'MockAny2AnyRetrievalI2T' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockAny2AnyRetrievalT2I | image, text | ✓ | - |
| MockVisionCentricQA | image, text | ✗ | ColNanoVDR only supports text queries, but task 'MockVisionCentricQA' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockImageClassification | image | ✗ | ColNanoVDR only supports text queries, but task 'MockImageClassification' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockImageClustering | image | ✗ | ColNanoVDR only supports text queries, but task 'MockImageClustering' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockImageTextPairClassification | image, text | ✗ | ColNanoVDR only supports text queries, but task 'MockImageTextPairClassification' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockVisualSTS | image | ✗ | ColNanoVDR only supports text queries, but task 'MockVisualSTS' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockZeroShotClassification | image, text | ✗ | ColNanoVDR only supports text queries, but task 'MockZeroShotClassification' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockImageMultilabelClassification | image | ✗ | ColNanoVDR only supports text queries, but task 'MockImageMultilabelClassification' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockMultilingualImageClassification | image | ✗ | ColNanoVDR only supports text queries, but task 'MockMultilingualImageClassification' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockMultilingualImageTextPairClassification | image, text | ✗ | ColNanoVDR only supports text queries, but task 'MockMultilingualImageTextPairClassification' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockMultilingualVisionCentricQA | image, text | ✗ | ColNanoVDR only supports text queries, but task 'MockMultilingualVisionCentricQA' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockImageClusteringFastTask | image | ✗ | ColNanoVDR only supports text queries, but task 'MockImageClusteringFastTask' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockImageRegressionTask | image | ✗ | ColNanoVDR only supports text queries, but task 'MockImageRegressionTask' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |
| MockPairImageClassificationTask | image | ✗ | ColNanoVDR only supports text queries, but task 'MockPairImageClassificationTask' provides image inputs for queries. ColNanoVDR is a text-query -> image-document retrieval model and does not support image-query or image-classification tasks. |

## Summary by Modality

| Pass     | Modality | Failures |
| -------- | -------- | -------- |
| ✗ (5/35) | text     | MockMultilingualClassificationTask, MockMultilingualClusteringTask, MockMultilingualClusteringFastTask, MockMultilingualPairClassificationTask, MockMultilingualRerankingTask, MockMultilingualRetrievalTask, MockMultilingualSTSTask, MockMultilingualMultilabelClassification, MockMultilingualSummarizationTask, MockMultilingualInstructionRetrieval, MockMultilingualInstructionReranking, MockClassificationTask, MockRegressionTask, MockClusteringTask, LegacyMockClusteringFastTask, MockPairClassificationTask, MockRerankingTask, MockRetrievalTask, MockSTSTask, MockMultilabelClassification, MockSummarizationTask, MockInstructionRetrieval, MockInstructionReranking, MockRetrievalDialogTask, MockAny2AnyRetrievalI2T, MockVisionCentricQA, MockImageTextPairClassification, MockZeroShotClassification, MockMultilingualImageTextPairClassification, MockMultilingualVisionCentricQA |
| ✗ (1/15) | image    | MockAny2AnyRetrievalI2T, MockVisionCentricQA, MockImageClassification, MockImageClustering, MockImageTextPairClassification, MockVisualSTS, MockZeroShotClassification, MockImageMultilabelClassification, MockMultilingualImageClassification, MockMultilingualImageTextPairClassification, MockMultilingualVisionCentricQA, MockImageClusteringFastTask, MockImageRegressionTask, MockPairImageClassificationTask |
| skipped  | audio    |  |
| skipped  | video    |  |