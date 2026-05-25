# MVEB Task Selection — scope: `audio-video`

MVEB — full A+V+T encoders (pe-av, ebind-av, omni, +)

## Pre-selection filters

- Source MVEB(extended): **184** tasks
- After scope filter (`audio-video`): **183** (-0)
- After AV-variant preference: **120** (-63)
- After annotation-provenance filter: **106** (-14)
- After saturation/floor filter (best≤0.93, spread≥0.05, n≥3): **92** (-14)

- Must-include tasks in scope: **0** (bypass annotation and saturation filters)

### Dropped — non-AV variant (AV variant exists)

- `AVEDatasetVideoClassification` — non-AV variant dropped; family 'AVEDataset' has AV variants
- `AVEDatasetVideoClustering` — non-AV variant dropped; family 'AVEDataset' has AV variants
- `AVEDatasetVideoZeroShot` — non-AV variant dropped; family 'AVEDataset' has AV variants
- `AVEDatasetVPairClassification` — non-AV variant dropped; family 'AVEDataset' has AV variants
- `AVMemeVideoClassification` — non-AV variant dropped; family 'AVMeme' has AV variants
- `AVMemeVideoZeroShot` — non-AV variant dropped; family 'AVMeme' has AV variants
- `HumanAnimalCartoonV` — non-AV variant dropped; family 'HumanAnimalCartoon' has AV variants
- `HumanAnimalCartoonZeroShot` — non-AV variant dropped; family 'HumanAnimalCartoon' has AV variants
- `HumanAnimalCartoonVPairClassification` — non-AV variant dropped; family 'HumanAnimalCartoon' has AV variants
- `Kinetics400V` — non-AV variant dropped; family 'Kinetics400' has AV variants
- `Kinetics400ZeroShot` — non-AV variant dropped; family 'Kinetics400' has AV variants
- `Kinetics600V` — non-AV variant dropped; family 'Kinetics600' has AV variants
- `Kinetics600VZeroShot` — non-AV variant dropped; family 'Kinetics600' has AV variants
- `Kinetics700V` — non-AV variant dropped; family 'Kinetics700' has AV variants
- `Kinetics700VZeroShot` — non-AV variant dropped; family 'Kinetics700' has AV variants
- `MELDVideoClassification` — non-AV variant dropped; family 'MELD' has AV variants
- `MELDEmotionVideoClustering` — non-AV variant dropped; family 'MELD' has AV variants
- `MELDSpeakerVideoClustering` — non-AV variant dropped; family 'MELD' has AV variants
- `MELDVideoZeroShot` — non-AV variant dropped; family 'MELD' has AV variants
- `MELDVPairClassification` — non-AV variant dropped; family 'MELD' has AV variants
- `MusicAVQACLSVideoClassification` — non-AV variant dropped; family 'MusicAVQACLS' has AV variants
- `MusicAVQACLSVideoClustering` — non-AV variant dropped; family 'MusicAVQACLS' has AV variants
- `MusicAVQACLSVideoZeroShot` — non-AV variant dropped; family 'MusicAVQACLS' has AV variants
- `RAVDESSVClassification` — non-AV variant dropped; family 'RAVDESS' has AV variants
- `RAVDESSVideoClustering` — non-AV variant dropped; family 'RAVDESS' has AV variants
- `RAVDESSVZeroShot` — non-AV variant dropped; family 'RAVDESS' has AV variants
- `RAVDESSAVVPairClassification` — non-AV variant dropped; family 'RAVDESS' has AV variants
- `UCF101VideoClassification` — non-AV variant dropped; family 'UCF101' has AV variants
- `UCF101VideoClustering` — non-AV variant dropped; family 'UCF101' has AV variants
- `UCF101VideoZeroShotClassification` — non-AV variant dropped; family 'UCF101' has AV variants
- ... and 33 more

### Dropped — annotation provenance

- `DiDeMoT2VARetrieval` — uses audio but 'DiDeMo' has visual-only labels
- `MSRVTTT2VA` — uses audio but 'MSRVTT' has visual-only labels
- `Panda70MT2VARetrieval` — uses audio but 'Panda70M' has visual-only labels
- `DiDeMoVA2TRetrieval` — uses audio but 'DiDeMo' has visual-only labels
- `MSRVTTVA2T` — uses audio but 'MSRVTT' has visual-only labels
- `Panda70MVA2TRetrieval` — uses audio but 'Panda70M' has visual-only labels
- `DiDeMoAT2VRetrieval` — uses audio but 'DiDeMo' has visual-only labels
- `DiDeMoV2ARetrieval` — uses audio but 'DiDeMo' has visual-only labels
- `MSRVTTV2A` — uses audio but 'MSRVTT' has visual-only labels
- `DiDeMoA2VRetrieval` — uses audio but 'DiDeMo' has visual-only labels
- `DiDeMoVT2ARetrieval` — uses audio but 'DiDeMo' has visual-only labels
- `MSRVTTVT2A` — uses audio but 'MSRVTT' has visual-only labels
- `MSRVTTA2V` — uses audio but 'MSRVTT' has visual-only labels
- `MSRVTTAT2V` — uses audio but 'MSRVTT' has visual-only labels

### Dropped — saturated, floor, or low-support

- `Diving48Classification.V2` — floor (spread=0.029 < 0.05)
- `AVEDatasetZeroShot` — saturated (best=0.940 > 0.93)
- `VinogroundPairClassification` — floor (spread=0.031 < 0.05)
- `AVEDatasetVAPairClassification` — saturated (best=0.963 > 0.93)
- `TUNABenchT2VRetrieval` — saturated (best=0.986 > 0.93)
- `TUNABenchV2TRetrieval` — saturated (best=0.983 > 0.93)
- `Shot2Story20KT2VARetrieval` — saturated (best=0.995 > 0.93)
- `VGGSoundAVT2VARetrieval` — saturated (best=0.991 > 0.93)
- `Shot2Story20KVA2TRetrieval` — saturated (best=0.990 > 0.93)
- `VGGSoundAVVA2TRetrieval` — saturated (best=0.987 > 0.93)
- `Shot2Story20KAT2VRetrieval` — saturated (best=0.994 > 0.93)
- `VATEXAT2VRetrieval` — saturated (best=0.931 > 0.93)
- `VGGSoundAVAT2VRetrieval` — saturated (best=0.987 > 0.93)
- `Shot2Story20KVT2ARetrieval` — saturated (best=0.933 > 0.93)

# MVEB Task Selection Analysis

## Overview
- **Source pool**: MVEB(extended) with 184 tasks
- **Working pool**: 92 tasks
- **Goal**: Select non-redundant tasks while preserving coverage

## Selection Rules

1. **Retrieval direction preference**: For task families with both V2T and T2V, prefer T2V (text-to-video)
2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task
3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage

## Protected Tasks (Unique Coverage): 0

## Task Families (Same Source Dataset)

### Other (22 tasks)
- AVQAVideoAudioCentricQA (VideoCentricQA, vat2t)
- AVQAVideoCentricQA (VideoCentricQA, vt2t)
- AVSpeakerBenchPairClassification (VideoPairClassification, v2v)
- AVSpeakerBenchVideoAudioCentricQA (VideoCentricQA, vat2t)
- AVSpeakerBenchVideoCentricQA (VideoCentricQA, vt2t)
- DailyOmniVideoAudioCentricQA (VideoCentricQA, vat2t)
- DailyOmniVideoCentricQA (VideoCentricQA, vt2t)
- EgoSchemaVideoCentricQA (VideoCentricQA, vt2t)
- MusicAVQAVAPairClassification (VideoPairClassification, va2va)
- MusicAVQAVPairClassification (VideoPairClassification, v2v)
- NExTQAVideoCentricQA (VideoCentricQA, vt2t)
- OmniVideoBenchVideoAudioCentricQA (VideoCentricQA, vat2t)
- OmniVideoBenchVideoCentricQA (VideoCentricQA, vt2t)
- PerceptionTestVideoAudioCentricQA (VideoCentricQA, vat2t)
- PerceptionTestVideoCentricQA (VideoCentricQA, vt2t)
- SomethingSomethingV2Classification (VideoClassification, v2c)
- VideoConPairClassification (VideoPairClassification, v2t)
- VideoMMEShortVideoAudioCentricQA (VideoCentricQA, vat2t)
- VideoMMEShortVideoCentricQA (VideoCentricQA, vt2t)
- VinogroundPairClassification (VideoPairClassification, v2v)
- WorldQAVideoAudioCentricQA (VideoCentricQA, vat2t)
- WorldQAVideoCentricQA (VideoCentricQA, vt2t)

### MELD (10 tasks)
- MELDAudioVideoClassification (VideoClassification, va2c)
- MELDAudioVideoZeroShot (VideoZeroshotClassification, va2t)
- MELDEmotionAudioVideoClustering (VideoClustering, va2c)
- MELDEmotionVideoClustering (VideoClustering, v2c)
- MELDSpeakerAudioVideoClustering (VideoClustering, va2c)
- MELDSpeakerVideoClustering (VideoClustering, v2c)
- MELDVAPairClassification (VideoPairClassification, va2va)
- MELDVPairClassification (VideoPairClassification, v2v)
- MELDVideoClassification (VideoClassification, v2c)
- MELDVideoZeroShot (VideoZeroshotClassification, v2t)

### AVMemeExam (10 tasks)
- AVMemeExamA2VRetrieval (Any2AnyRetrieval, a2v)
- AVMemeExamAT2VRetrieval (Any2AnyRetrieval, at2v)
- AVMemeExamT2VARetrieval (Any2AnyRetrieval, t2va)
- AVMemeExamT2VRetrieval (Any2AnyRetrieval, t2v)
- AVMemeExamV2ARetrieval (Any2AnyRetrieval, v2a)
- AVMemeExamV2TRetrieval (Any2AnyRetrieval, v2t)
- AVMemeExamVA2TRetrieval (Any2AnyRetrieval, va2t)
- AVMemeExamVT2ARetrieval (Any2AnyRetrieval, vt2a)
- AVMemeExamVideoAudioCentricQA (VideoCentricQA, vat2t)
- AVMemeExamVideoCentricQA (VideoCentricQA, vt2t)

### AVEDataset (8 tasks)
- AVEDatasetAudioVideoClustering (VideoClustering, va2c)
- AVEDatasetClassification (VideoClassification, va2c)
- AVEDatasetVAPairClassification (VideoPairClassification, va2va)
- AVEDatasetVPairClassification (VideoPairClassification, v2v)
- AVEDatasetVideoClassification (VideoClassification, v2c)
- AVEDatasetVideoClustering (VideoClustering, v2c)
- AVEDatasetVideoZeroShot (VideoZeroshotClassification, v2t)
- AVEDatasetZeroShot (VideoZeroshotClassification, va2t)

### RAVDESS (8 tasks)
- RAVDESSAVClassification (VideoClassification, va2c)
- RAVDESSAVClustering (VideoClustering, va2c)
- RAVDESSAVVAPairClassification (VideoPairClassification, va2va)
- RAVDESSAVVPairClassification (VideoPairClassification, v2v)
- RAVDESSAVZeroShot (VideoZeroshotClassification, va2t)
- RAVDESSVClassification (VideoClassification, v2c)
- RAVDESSVZeroShot (VideoZeroshotClassification, v2t)
- RAVDESSVideoClustering (VideoClustering, v2c)

### WorldSense (8 tasks)
- WorldSense1MinDomainAudioVideoClustering (VideoClustering, va2c)
- WorldSense1MinDomainVideoClustering (VideoClustering, v2c)
- WorldSense1MinVideoAudioCentricQA (VideoCentricQA, vat2t)
- WorldSense1MinVideoCentricQA (VideoCentricQA, vt2t)
- WorldSenseAudioVideoClassification (VideoClassification, va2c)
- WorldSenseAudioVideoZeroShot (VideoZeroshotClassification, va2t)
- WorldSenseVideoClassification (VideoClassification, v2c)
- WorldSenseVideoZeroShot (VideoZeroshotClassification, v2t)

### AudioCapsAV (8 tasks)
- AudioCapsAVA2VRetrieval (Any2AnyRetrieval, a2v)
- AudioCapsAVAT2VRetrieval (Any2AnyRetrieval, at2v)
- AudioCapsAVT2VARetrieval (Any2AnyRetrieval, t2va)
- AudioCapsAVT2VRetrieval (Any2AnyRetrieval, t2v)
- AudioCapsAVV2ARetrieval (Any2AnyRetrieval, v2a)
- AudioCapsAVV2TRetrieval (Any2AnyRetrieval, v2t)
- AudioCapsAVVA2TRetrieval (Any2AnyRetrieval, va2t)
- AudioCapsAVVT2ARetrieval (Any2AnyRetrieval, vt2a)

### DiDeMo (8 tasks)
- DiDeMoA2VRetrieval (Any2AnyRetrieval, a2v)
- DiDeMoAT2VRetrieval (Any2AnyRetrieval, at2v)
- DiDeMoT2VARetrieval (Any2AnyRetrieval, t2va)
- DiDeMoT2VRetrieval (Any2AnyRetrieval, t2v)
- DiDeMoV2ARetrieval (Any2AnyRetrieval, v2a)
- DiDeMoV2TRetrieval (Any2AnyRetrieval, v2t)
- DiDeMoVA2TRetrieval (Any2AnyRetrieval, va2t)
- DiDeMoVT2ARetrieval (Any2AnyRetrieval, vt2a)

### MSRVTT (8 tasks)
- MSRVTTA2V (Any2AnyRetrieval, a2v)
- MSRVTTAT2V (Any2AnyRetrieval, at2v)
- MSRVTTT2V (Any2AnyRetrieval, t2v)
- MSRVTTT2VA (Any2AnyRetrieval, t2va)
- MSRVTTV2A (Any2AnyRetrieval, v2a)
- MSRVTTV2T (Any2AnyRetrieval, v2t)
- MSRVTTVA2T (Any2AnyRetrieval, va2t)
- MSRVTTVT2A (Any2AnyRetrieval, vt2a)

### Shot2Story20K (8 tasks)
- Shot2Story20KA2VRetrieval (Any2AnyRetrieval, a2v)
- Shot2Story20KAT2VRetrieval (Any2AnyRetrieval, at2v)
- Shot2Story20KT2VARetrieval (Any2AnyRetrieval, t2va)
- Shot2Story20KT2VRetrieval (Any2AnyRetrieval, t2v)
- Shot2Story20KV2ARetrieval (Any2AnyRetrieval, v2a)
- Shot2Story20KV2TRetrieval (Any2AnyRetrieval, v2t)
- Shot2Story20KVA2TRetrieval (Any2AnyRetrieval, va2t)
- Shot2Story20KVT2ARetrieval (Any2AnyRetrieval, vt2a)

### VALOR32K (8 tasks)
- VALOR32KA2VRetrieval (Any2AnyRetrieval, a2v)
- VALOR32KAT2VRetrieval (Any2AnyRetrieval, at2v)
- VALOR32KT2VARetrieval (Any2AnyRetrieval, t2va)
- VALOR32KT2VRetrieval (Any2AnyRetrieval, t2v)
- VALOR32KV2ARetrieval (Any2AnyRetrieval, v2a)
- VALOR32KV2TRetrieval (Any2AnyRetrieval, v2t)
- VALOR32KVA2TRetrieval (Any2AnyRetrieval, va2t)
- VALOR32KVT2ARetrieval (Any2AnyRetrieval, vt2a)

### VATEX (8 tasks)
- VATEXA2VRetrieval (Any2AnyRetrieval, a2v)
- VATEXAT2VRetrieval (Any2AnyRetrieval, at2v)
- VATEXT2VARetrieval (Any2AnyRetrieval, t2va)
- VATEXT2VRetrieval (Any2AnyRetrieval, t2v)
- VATEXV2ARetrieval (Any2AnyRetrieval, v2a)
- VATEXV2TRetrieval (Any2AnyRetrieval, v2t)
- VATEXVA2TRetrieval (Any2AnyRetrieval, va2t)
- VATEXVT2ARetrieval (Any2AnyRetrieval, vt2a)

### VGGSoundAV (8 tasks)
- VGGSoundAVA2VRetrieval (Any2AnyRetrieval, a2v)
- VGGSoundAVAT2VRetrieval (Any2AnyRetrieval, at2v)
- VGGSoundAVT2VARetrieval (Any2AnyRetrieval, t2va)
- VGGSoundAVT2VRetrieval (Any2AnyRetrieval, t2v)
- VGGSoundAVV2ARetrieval (Any2AnyRetrieval, v2a)
- VGGSoundAVV2TRetrieval (Any2AnyRetrieval, v2t)
- VGGSoundAVVA2TRetrieval (Any2AnyRetrieval, va2t)
- VGGSoundAVVT2ARetrieval (Any2AnyRetrieval, vt2a)

### YouCook2 (8 tasks)
- YouCook2A2VRetrieval (Any2AnyRetrieval, a2v)
- YouCook2AT2VRetrieval (Any2AnyRetrieval, at2v)
- YouCook2T2VARetrieval (Any2AnyRetrieval, t2va)
- YouCook2T2VRetrieval (Any2AnyRetrieval, t2v)
- YouCook2V2ARetrieval (Any2AnyRetrieval, v2a)
- YouCook2V2TRetrieval (Any2AnyRetrieval, v2t)
- YouCook2VA2TRetrieval (Any2AnyRetrieval, va2t)
- YouCook2VT2ARetrieval (Any2AnyRetrieval, vt2a)

### HumanAnimalCartoon (6 tasks)
- HumanAnimalCartoonV (VideoClassification, v2c)
- HumanAnimalCartoonVA (VideoClassification, va2c)
- HumanAnimalCartoonVAPairClassification (VideoPairClassification, va2va)
- HumanAnimalCartoonVAZeroShot (VideoZeroshotClassification, va2t)
- HumanAnimalCartoonVPairClassification (VideoPairClassification, v2v)
- HumanAnimalCartoonZeroShot (VideoZeroshotClassification, v2t)

### MusicAVQACLS (6 tasks)
- MusicAVQACLSAudioVideoClassification (VideoClassification, va2c)
- MusicAVQACLSAudioVideoClustering (VideoClustering, va2c)
- MusicAVQACLSAudioVideoZeroShot (VideoZeroshotClassification, va2t)
- MusicAVQACLSVideoClassification (VideoClassification, v2c)
- MusicAVQACLSVideoClustering (VideoClustering, v2c)
- MusicAVQACLSVideoZeroShot (VideoZeroshotClassification, v2t)

### UCF101 (6 tasks)
- UCF101AudioVideoClustering (VideoClustering, va2c)
- UCF101VideoAudioClassification (VideoClassification, va2c)
- UCF101VideoAudioZeroShotClassification (VideoZeroshotClassification, va2t)
- UCF101VideoClassification (VideoClassification, v2c)
- UCF101VideoClustering (VideoClustering, v2c)
- UCF101VideoZeroShotClassification (VideoZeroshotClassification, va2t)

### AVMeme (4 tasks)
- AVMemeAudioVideoClassification (VideoClassification, va2c)
- AVMemeAudioVideoZeroShot (VideoZeroshotClassification, va2t)
- AVMemeVideoClassification (VideoClassification, v2c)
- AVMemeVideoZeroShot (VideoZeroshotClassification, v2t)

### Kinetics400 (4 tasks)
- Kinetics400V (VideoClassification, v2c)
- Kinetics400VA (VideoClassification, va2c)
- Kinetics400VAZeroShot (VideoZeroshotClassification, va2t)
- Kinetics400ZeroShot (VideoZeroshotClassification, v2t)

### Kinetics600 (4 tasks)
- Kinetics600V (VideoClassification, v2c)
- Kinetics600VA (VideoClassification, va2c)
- Kinetics600VAZeroShot (VideoZeroshotClassification, va2t)
- Kinetics600VZeroShot (VideoZeroshotClassification, v2t)

### Kinetics700 (4 tasks)
- Kinetics700V (VideoClassification, v2c)
- Kinetics700VA (VideoClassification, va2c)
- Kinetics700VAZeroShot (VideoZeroshotClassification, va2t)
- Kinetics700VZeroShot (VideoZeroshotClassification, v2t)

### VGGSound (4 tasks)
- VGGSoundV (VideoClassification, v2c)
- VGGSoundVA (VideoClassification, va2c)
- VGGSoundVideoAudioZeroshot (VideoZeroshotClassification, va2t)
- VGGSoundVideoZeroshot (VideoZeroshotClassification, v2t)

### Panda70M (4 tasks)
- Panda70MT2VARetrieval (Any2AnyRetrieval, t2va)
- Panda70MT2VRetrieval (Any2AnyRetrieval, t2v)
- Panda70MV2TRetrieval (Any2AnyRetrieval, v2t)
- Panda70MVA2TRetrieval (Any2AnyRetrieval, va2t)

### HMDB51 (3 tasks)
- HMDB51Classification (VideoClassification, v2c)
- HMDB51Clustering (VideoClustering, v2c)
- HMDB51ZeroShot (VideoZeroshotClassification, v2t)

### Breakfast (2 tasks)
- BreakfastClassification (VideoClassification, v2c)
- BreakfastZeroShot (VideoZeroshotClassification, v2t)

### ActivityNetCaptions (2 tasks)
- ActivityNetCaptionsT2VRetrieval (Any2AnyRetrieval, t2v)
- ActivityNetCaptionsV2TRetrieval (Any2AnyRetrieval, v2t)

### MSVD (2 tasks)
- MSVDT2VRetrieval (Any2AnyRetrieval, t2v)
- MSVDV2TRetrieval (Any2AnyRetrieval, v2t)

### TUNABench (2 tasks)
- TUNABenchT2VRetrieval (Any2AnyRetrieval, t2v)
- TUNABenchV2TRetrieval (Any2AnyRetrieval, v2t)

## Evaluation Time (MVEB Extended Working Pool)

| Model | Time | Tasks w/ data |
|-------|------|---------------|
| ebind-av (encord-team/ebind-audio-vision) | 77h 16m | 92/92 |
| pe-av-small (facebook/pe-av-small) | 93h 16m | 92/92 |
| LCO-Embedding-Omni-7B (LCO-Embedding/LCO-Embedding-Omni-7B) | 172h 16m | 92/92 |
| Qwen2.5-Omni-7B (Qwen/Qwen2.5-Omni-7B) | 148h 12m | 92/92 |

## Selection Results Summary

| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson | ebind-av | pe-av-small | LCO-Embedding-Omni-7B | Qwen2.5-Omni-7B |
|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|--- | --- | --- | ---|
| 0.95 | 35 | 18 | 7 | 3 | 0 | 2 | 3 | 2 | 16 | 8 | 0.9876 | 0.9985 | 30h 56m | 33h 14m | 59h 6m | 54h 11m |
| 0.93 | 29 | 13 | 6 | 3 | 0 | 1 | 4 | 2 | 16 | 8 | 0.9766 | 0.9985 | 22h 38m | 28h 27m | 46h 20m | 45h 39m |
| 0.9 | 26 | 11 | 7 | 2 | 0 | 2 | 2 | 2 | 16 | 8 | 0.9436 | 0.9947 | 22h 50m | 23h 36m | 41h 56m | 40h 52m |
| 0.88 | 25 | 11 | 6 | 2 | 0 | 2 | 2 | 2 | 16 | 8 | 0.9271 | 0.9934 | 19h 35m | 20h 47m | 37h 55m | 35h 23m |
| 0.87 | 25 | 11 | 6 | 2 | 0 | 2 | 2 | 2 | 16 | 8 | 0.9271 | 0.9934 | 19h 35m | 20h 47m | 37h 55m | 35h 23m |
| 0.85 | 23 | 10 | 6 | 2 | 0 | 2 | 2 | 1 | 16 | 8 | 0.9436 | 0.9955 | 16h 49m | 16h 40m | 31h 52m | 29h 22m |
| 0.84 | 20 | 9 | 6 | 1 | 0 | 1 | 1 | 2 | 16 | 8 | 0.9381 | 0.9965 | 14h 32m | 12h 21m | 22h 46m | 19h 55m |
| 0.83 | 20 | 9 | 5 | 1 | 0 | 1 | 2 | 2 | 16 | 8 | 0.9436 | 0.9964 | 12h 40m | 11h 58m | 21h 51m | 19h 10m |
| 0.82 | 19 | 8 | 5 | 1 | 0 | 1 | 3 | 1 | 16 | 8 | 0.9491 | 0.9963 | 12h 49m | 10h 43m | 21h 31m | 19h 11m |
| 0.81 | 19 | 8 | 5 | 1 | 0 | 1 | 3 | 1 | 16 | 8 | 0.9491 | 0.9963 | 12h 49m | 10h 43m | 21h 31m | 19h 11m |
| 0.8 | 18 | 7 | 5 | 1 | 0 | 1 | 3 | 1 | 16 | 8 | 0.9436 | 0.9968 | 12h 28m | 9h 40m | 19h 52m | 17h 32m |
| 0.7 | 13 | 6 | 2 | 1 | 0 | 1 | 2 | 1 | 16 | 8 | 0.8556 | 0.9877 | 4h 6m | 5h 46m | 11h 13m | 8h 55m |
| 0.6 | 13 | 6 | 2 | 1 | 0 | 1 | 2 | 1 | 16 | 8 | 0.8556 | 0.9877 | 4h 6m | 5h 46m | 11h 13m | 8h 55m |
| 0.5 | 13 | 6 | 2 | 1 | 0 | 1 | 2 | 1 | 16 | 8 | 0.8556 | 0.9877 | 4h 6m | 5h 46m | 11h 13m | 8h 55m |

*Working pool: 92 tasks, 16 langs, 8 doms*

*Spearman/Pearson: Correlation of average model scores between selected tasks and full MVEB(extended)*

## Threshold 0.95

**184 → 35 tasks** (57 removed)

### Remaining Tasks

#### Any2AnyRetrieval (18)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Shot2Story20KA2VRetrieval** - a2v, Web, Spoken
- **Shot2Story20KV2ARetrieval** - v2a, Web, Spoken
- **VALOR32KAT2VRetrieval** - at2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVA2TRetrieval** - va2t, Web, Spoken
- **VATEXA2VRetrieval** - a2v, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVVT2ARetrieval** - vt2a, Web, Spoken
- **YouCook2AT2VRetrieval** - at2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken
- **YouCook2VA2TRetrieval** - va2t, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **HMDB51Classification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (3)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music
- **UCF101AudioVideoClustering** - va2c, Web, Scene

#### VideoPairClassification (2)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music

#### VideoZeroshotClassification (3)
- **BreakfastZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 13 (was 13)
- Types: 6 (was 6)

## Threshold 0.93

**184 → 29 tasks** (63 removed)

### Remaining Tasks

#### Any2AnyRetrieval (13)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Shot2Story20KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken
- **YouCook2VA2TRetrieval** - va2t, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken

#### VideoClustering (3)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music
- **UCF101AudioVideoClustering** - va2c, Web, Scene

#### VideoPairClassification (1)
- **MusicAVQAVAPairClassification** - va2va, Music

#### VideoZeroshotClassification (4)
- **HumanAnimalCartoonVAZeroShot** - va2t, Web, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **VGGSoundVideoAudioZeroshot** - va2t, Web
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.9

**184 → 26 tasks** (66 removed)

### Remaining Tasks

#### Any2AnyRetrieval (11)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (2)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music

#### VideoPairClassification (2)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music

#### VideoZeroshotClassification (2)
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **VGGSoundVideoAudioZeroshot** - va2t, Web

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.88

**184 → 25 tasks** (67 removed)

### Remaining Tasks

#### Any2AnyRetrieval (11)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (2)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music

#### VideoPairClassification (2)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 13 (was 13)
- Types: 6 (was 6)

## Threshold 0.87

**184 → 25 tasks** (67 removed)

### Remaining Tasks

#### Any2AnyRetrieval (11)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (2)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music

#### VideoPairClassification (2)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 13 (was 13)
- Types: 6 (was 6)

## Threshold 0.85

**184 → 23 tasks** (69 removed)

### Remaining Tasks

#### Any2AnyRetrieval (10)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (2)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music

#### VideoPairClassification (2)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.84

**184 → 20 tasks** (72 removed)

### Remaining Tasks

#### Any2AnyRetrieval (9)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene

#### VideoZeroshotClassification (1)
- **HMDB51ZeroShot** - v2t, Scene

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 13 (was 13)
- Types: 6 (was 6)

## Threshold 0.83

**184 → 20 tasks** (72 removed)

### Remaining Tasks

#### Any2AnyRetrieval (9)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (5)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **UCF101VideoAudioZeroShotClassification** - va2t, Web, Scene

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 13 (was 13)
- Types: 6 (was 6)

## Threshold 0.82

**184 → 19 tasks** (73 removed)

### Remaining Tasks

#### Any2AnyRetrieval (8)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (5)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene

#### VideoZeroshotClassification (3)
- **HMDB51ZeroShot** - v2t, Scene
- **UCF101VideoAudioZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.81

**184 → 19 tasks** (73 removed)

### Remaining Tasks

#### Any2AnyRetrieval (8)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (5)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene

#### VideoZeroshotClassification (3)
- **HMDB51ZeroShot** - v2t, Scene
- **UCF101VideoAudioZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.8

**184 → 18 tasks** (74 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (5)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene

#### VideoZeroshotClassification (3)
- **HMDB51ZeroShot** - v2t, Scene
- **UCF101VideoAudioZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.7

**184 → 13 tasks** (79 removed)

### Remaining Tasks

#### Any2AnyRetrieval (6)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (2)
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.6

**184 → 13 tasks** (79 removed)

### Remaining Tasks

#### Any2AnyRetrieval (6)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (2)
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Threshold 0.5

**184 → 13 tasks** (79 removed)

### Remaining Tasks

#### Any2AnyRetrieval (6)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken

#### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

#### VideoClassification (2)
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene

#### VideoClustering (1)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 12 (was 13)
- Types: 6 (was 6)

## Recommended MVEB Task List (threshold=0.85)

**Total: 23 tasks**

### Any2AnyRetrieval (10)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **AudioCapsAVVT2ARetrieval** - vt2a, Encyclopaedic, Web
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVA2VRetrieval** - a2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken

### VideoCentricQA (1)
- **EgoSchemaVideoCentricQA** - vt2t, Web

### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700VA** - va2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

### VideoClustering (2)
- **MELDEmotionAudioVideoClustering** - va2c, Entertainment
- **MusicAVQACLSAudioVideoClustering** - va2c, Music

### VideoPairClassification (2)
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music

### VideoZeroshotClassification (2)
- **HMDB51ZeroShot** - v2t, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Code for benchmarks.py

```python
tasks=get_tasks(
    tasks=[
        # Any2AnyRetrieval (10)
        "AVMemeExamAT2VRetrieval",
        "ActivityNetCaptionsT2VRetrieval",
        "AudioCapsAVVA2TRetrieval",
        "AudioCapsAVVT2ARetrieval",
        "MSVDT2VRetrieval",
        "VALOR32KT2VARetrieval",
        "VATEXV2ARetrieval",
        "VATEXVA2TRetrieval",
        "VGGSoundAVA2VRetrieval",
        "YouCook2T2VARetrieval",
        # VideoCentricQA (1)
        "EgoSchemaVideoCentricQA",
        # VideoClassification (6)
        "AVEDatasetClassification",
        "AVMemeAudioVideoClassification",
        "BreakfastClassification",
        "Kinetics700VA",
        "RAVDESSAVClassification",
        "UCF101VideoAudioClassification",
        # VideoClustering (2)
        "MELDEmotionAudioVideoClustering",
        "MusicAVQACLSAudioVideoClustering",
        # VideoPairClassification (2)
        "HumanAnimalCartoonVAPairClassification",
        "MusicAVQAVAPairClassification",
        # VideoZeroshotClassification (2)
        "HMDB51ZeroShot",
        "WorldSenseAudioVideoZeroShot",
    ]
),
```
