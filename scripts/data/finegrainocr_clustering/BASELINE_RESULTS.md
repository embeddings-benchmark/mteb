# FineGrainOCR image-text clustering baselines

These experiments use the processed 4,919-row test set described in
`BUILD_SUMMARY.json`. They ran locally on an arm64 macOS CPU with no CUDA or MPS
accelerator.

## Protocol

- Seed: 42
- Repeats: 10
- Rows sampled with replacement per repeat: 16,384
- Gold classes / K-means clusters: 256
- MiniBatch K-means batch size: 512
- Initialization: k-means++, one initialization
- Metrics: V-measure and adjusted mutual information (AMI)

This matches the current MTEB `AbsTaskClustering` defaults. `MTEB add` is the
raw image-plus-text fusion used by the repository's CLIP and SigLIP wrappers.
For normalized fusion, each modality is L2-normalized before weighting and the
fused vector is normalized again. The named weight is the image weight.

## Results

| Model | Experiment | V-measure | AMI |
|---|---|---:|---:|
| `openai/clip-vit-base-patch32` | Random Gaussian control | 0.1466 ± 0.0106 | 0.0696 ± 0.0088 |
| | Image only, MTEB output | 0.7263 ± 0.0044 | 0.6421 ± 0.0054 |
| | Text only, MTEB output | 0.6957 ± 0.0039 | 0.6019 ± 0.0042 |
| | Image + text, MTEB add | **0.7664 ± 0.0028** | **0.6962 ± 0.0036** |
| | Normalized fusion, 25% image | 0.7149 ± 0.0030 | 0.6278 ± 0.0032 |
| | Normalized fusion, 50% image | 0.7603 ± 0.0022 | 0.6882 ± 0.0023 |
| | Normalized fusion, 75% image | 0.7608 ± 0.0036 | 0.6889 ± 0.0048 |
| `google/siglip-base-patch16-256` | Random Gaussian control | 0.1278 ± 0.0044 | 0.0502 ± 0.0057 |
| | Image only, MTEB output | 0.7912 ± 0.0039 | 0.7269 ± 0.0048 |
| | Text only, MTEB output | 0.6948 ± 0.0049 | 0.6060 ± 0.0051 |
| | Image + text, MTEB add | 0.7681 ± 0.0048 | 0.6991 ± 0.0058 |
| | Normalized fusion, 25% image | 0.7170 ± 0.0048 | 0.6341 ± 0.0053 |
| | Normalized fusion, 50% image | 0.7800 ± 0.0039 | 0.7149 ± 0.0048 |
| | Normalized fusion, 75% image | **0.8065 ± 0.0039** | **0.7476 ± 0.0049** |

Pinned model revisions:

- CLIP ViT-B/32: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- SigLIP B/16-256: `b078df89e446d623010d890864d4207fe6399f61`

## Interpretation

The chance controls are far below every real representation. Both modalities
are individually predictive, but neither is saturated. CLIP's standard raw
additive fusion improves V-measure by 0.040 over image-only and 0.071 over
text-only. SigLIP image-only is stronger, and raw addition is hurt by its mean
text-vector norm (17.28) exceeding its image-vector norm (14.49). After
normalizing the modalities, 75% image / 25% text improves V-measure by 0.015
over image-only. The benchmark therefore measures both complementary modality
signal and the quality of a model's fusion strategy.
