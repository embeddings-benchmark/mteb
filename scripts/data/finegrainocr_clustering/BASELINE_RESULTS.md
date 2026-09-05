# FineGrainOCR image-text clustering baselines

These experiments use the processed 18,389-row test set described in
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
| `openai/clip-vit-base-patch32` | Random Gaussian control | 0.0669 ± 0.0041 | 0.0174 ± 0.0020 |
| | Image only, MTEB output | 0.7033 ± 0.0050 | 0.6116 ± 0.0062 |
| | Text only, MTEB output | 0.6649 ± 0.0040 | 0.5624 ± 0.0044 |
| | Image + text, MTEB add | **0.7503 ± 0.0032** | **0.6753 ± 0.0046** |
| | Normalized fusion, 25% image | 0.6880 ± 0.0046 | 0.5925 ± 0.0055 |
| | Normalized fusion, 50% image | 0.7442 ± 0.0037 | 0.6682 ± 0.0045 |
| | Normalized fusion, 75% image | 0.7428 ± 0.0053 | 0.6650 ± 0.0066 |
| `google/siglip-base-patch16-256` | Random Gaussian control | 0.0599 ± 0.0045 | 0.0128 ± 0.0025 |
| | Image only, MTEB output | 0.7800 ± 0.0039 | 0.7121 ± 0.0050 |
| | Text only, MTEB output | 0.6741 ± 0.0045 | 0.5792 ± 0.0050 |
| | Image + text, MTEB add | 0.7559 ± 0.0048 | 0.6846 ± 0.0057 |
| | Normalized fusion, 25% image | 0.7028 ± 0.0040 | 0.6163 ± 0.0051 |
| | Normalized fusion, 50% image | 0.7719 ± 0.0053 | 0.7060 ± 0.0065 |
| | Normalized fusion, 75% image | **0.7966 ± 0.0034** | **0.7361 ± 0.0041** |

Pinned model revisions:

- CLIP ViT-B/32: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- SigLIP B/16-256: `b078df89e446d623010d890864d4207fe6399f61`

## Interpretation

The chance controls are far below every real representation. Both modalities
are individually predictive, but neither is saturated. CLIP's standard raw
additive fusion improves V-measure by 0.047 over image-only and 0.085 over
text-only. SigLIP image-only is stronger, and raw addition is hurt by its mean
text-vector norm (17.30) exceeding its image-vector norm (14.48). After
normalizing the modalities, 75% image / 25% text improves V-measure by 0.017
over image-only. The benchmark therefore measures both complementary modality
signal and the quality of a model's fusion strategy.
