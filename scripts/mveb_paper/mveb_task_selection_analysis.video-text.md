# MVEB Task Selection — scope: `video-text`

MVEB(text, video) — T+V encoders (xclip, UME-R1, ebind-points-vision, +)

## Pre-selection filters

- Source MVEB(extended): **184** tasks
- After scope filter (`video-text`): **82** (-101)
- After annotation-provenance filter: **82** (-0)
- After saturation/floor filter (best≤0.93, spread≥0.05, n≥3): **66** (-16)

- Must-include tasks in scope: **0** (bypass annotation and saturation filters)

### Dropped — saturated, floor, or low-support

- `Diving48Classification.V2` — floor (spread=0.029 < 0.05)
- `HumanAnimalCartoonV` — saturated (best=0.933 > 0.93)
- `UCF101VideoClassification` — saturated (best=0.943 > 0.93)
- `MELDEmotionVideoClustering` — floor (spread=0.012 < 0.05)
- `UCF101VideoClustering` — saturated (best=0.941 > 0.93)
- `HumanAnimalCartoonZeroShot` — saturated (best=0.938 > 0.93)
- `AVEDatasetVPairClassification` — saturated (best=0.970 > 0.93)
- `AVSpeakerBenchPairClassification` — saturated (best=0.992 > 0.93)
- `MELDVPairClassification` — floor (spread=0.041 < 0.05)
- `VinogroundPairClassification` — floor (spread=0.031 < 0.05)
- `Shot2Story20KT2VRetrieval` — saturated (best=0.991 > 0.93)
- `TUNABenchT2VRetrieval` — saturated (best=0.986 > 0.93)
- `VGGSoundAVT2VRetrieval` — saturated (best=0.988 > 0.93)
- `Shot2Story20KV2TRetrieval` — saturated (best=0.986 > 0.93)
- `TUNABenchV2TRetrieval` — saturated (best=0.983 > 0.93)
- `VGGSoundAVV2TRetrieval` — saturated (best=0.989 > 0.93)

# MVEB Task Selection Analysis

## Overview
- **Source pool**: MVEB(extended) with 184 tasks
- **Working pool**: 66 tasks
- **Goal**: Select non-redundant tasks while preserving coverage

## Selection Rules

1. **Retrieval direction preference**: For task families with both V2T and T2V, prefer T2V (text-to-video)
2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task
3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage

## Protected Tasks (Unique Coverage): 1

### Unique Category
- UCF101VideoZeroShotClassification (unique: va2t)

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
| ebind-av (encord-team/ebind-audio-vision) | 37h 28m | 66/66 |
| pe-av-small (facebook/pe-av-small) | 59h 53m | 66/66 |
| LCO-Embedding-Omni-7B (LCO-Embedding/LCO-Embedding-Omni-7B) | 119h 43m | 66/66 |
| Qwen2.5-Omni-7B (Qwen/Qwen2.5-Omni-7B) | 98h 26m | 66/66 |

## Selection Results Summary

| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson | ebind-av | pe-av-small | LCO-Embedding-Omni-7B | Qwen2.5-Omni-7B |
|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|--- | --- | --- | ---|
| 0.95 | 31 | 11 | 7 | 2 | 0 | 1 | 5 | 5 | 16 | 8 | 0.9381 | 0.9925 | 17h 35m | 27h 41m | 59h 27m | 42h 46m |
| 0.93 | 28 | 10 | 5 | 2 | 0 | 1 | 5 | 5 | 16 | 8 | 0.9161 | 0.9916 | 15h 17m | 25h 36m | 55h 9m | 38h 34m |
| 0.9 | 26 | 10 | 5 | 2 | 0 | 1 | 5 | 3 | 16 | 8 | 0.9546 | 0.9919 | 14h 35m | 20h 11m | 36h 13m | 33h 35m |
| 0.88 | 20 | 8 | 4 | 2 | 0 | 1 | 4 | 1 | 16 | 8 | 0.9601 | 0.9924 | 13h 9m | 16h 30m | 30h 33m | 29h 23m |
| 0.87 | 19 | 8 | 4 | 1 | 0 | 1 | 4 | 1 | 16 | 8 | 0.9766 | 0.9913 | 12h 48m | 15h 13m | 28h 7m | 26h 57m |
| 0.85 | 18 | 7 | 4 | 0 | 0 | 1 | 5 | 1 | 16 | 8 | 0.9326 | 0.9913 | 12h 42m | 14h 52m | 27h 9m | 25h 58m |
| 0.84 | 17 | 7 | 4 | 0 | 0 | 1 | 4 | 1 | 16 | 8 | 0.9381 | 0.9916 | 12h 4m | 14h 11m | 25h 40m | 24h 24m |
| 0.83 | 16 | 7 | 3 | 0 | 0 | 1 | 4 | 1 | 16 | 8 | 0.9161 | 0.9896 | 8h 24m | 12h 7m | 21h 26m | 19h 45m |
| 0.82 | 15 | 6 | 3 | 0 | 0 | 1 | 4 | 1 | 16 | 8 | 0.9271 | 0.9910 | 8h 22m | 12h 2m | 21h 15m | 19h 35m |
| 0.81 | 14 | 6 | 3 | 0 | 0 | 0 | 4 | 1 | 16 | 8 | 0.9271 | 0.9891 | 8h 1m | 11h 39m | 20h 24m | 18h 45m |
| 0.8 | 13 | 6 | 2 | 0 | 0 | 0 | 4 | 1 | 16 | 8 | 0.8886 | 0.9874 | 7h 59m | 11h 35m | 20h 15m | 18h 38m |
| 0.7 | 10 | 4 | 1 | 0 | 0 | 0 | 4 | 1 | 16 | 8 | 0.9106 | 0.9891 | 1h 37m | 5h 52m | 9h 42m | 6h 42m |
| 0.6 | 8 | 2 | 1 | 0 | 0 | 0 | 4 | 1 | 16 | 8 | 0.9106 | 0.9761 | 1h 8m | 3h 49m | 5h 58m | 3h 49m |
| 0.5 | 8 | 2 | 1 | 0 | 0 | 1 | 3 | 1 | 16 | 8 | 0.9436 | 0.9753 | 1h 34m | 4h 4m | 6h 34m | 4h 27m |

*Working pool: 66 tasks, 16 langs, 8 doms*

*Spearman/Pearson: Correlation of average model scores between selected tasks and full MVEB(extended)*

## Threshold 0.95

**184 → 31 tasks** (35 removed)

### Remaining Tasks

#### Any2AnyRetrieval (11)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (5)
- **DailyOmniVideoCentricQA** - vt2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web
- **WorldQAVideoCentricQA** - vt2t, Web

#### VideoClassification (7)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (2)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (5)
- **AVEDatasetVideoZeroShot** - v2t, Web, AudioScene
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 6 (was 6)

## Threshold 0.93

**184 → 28 tasks** (38 removed)

### Remaining Tasks

#### Any2AnyRetrieval (10)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (5)
- **DailyOmniVideoCentricQA** - vt2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web
- **WorldQAVideoCentricQA** - vt2t, Web

#### VideoClassification (5)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (2)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (5)
- **AVEDatasetVideoZeroShot** - v2t, Web, AudioScene
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 6 (was 6)

## Threshold 0.9

**184 → 26 tasks** (40 removed)

### Remaining Tasks

#### Any2AnyRetrieval (10)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **DiDeMoV2TRetrieval** - v2t, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (3)
- **DailyOmniVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web

#### VideoClassification (5)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (2)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (5)
- **AVEDatasetVideoZeroShot** - v2t, Web, AudioScene
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 6 (was 6)

## Threshold 0.88

**184 → 20 tasks** (46 removed)

### Remaining Tasks

#### Any2AnyRetrieval (8)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **DiDeMoV2TRetrieval** - v2t, Web, Spoken
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (4)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (2)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (4)
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 6 (was 6)

## Threshold 0.87

**184 → 19 tasks** (47 removed)

### Remaining Tasks

#### Any2AnyRetrieval (8)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **DiDeMoV2TRetrieval** - v2t, Web, Spoken
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (4)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (1)
- **RAVDESSVideoClustering** - v2c, Spoken

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (4)
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 6 (was 6)

## Threshold 0.85

**184 → 18 tasks** (48 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (4)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (5)
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 5 (was 6)

## Threshold 0.84

**184 → 17 tasks** (49 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (4)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 5 (was 6)

## Threshold 0.83

**184 → 16 tasks** (50 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (3)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 5 (was 6)

## Threshold 0.82

**184 → 15 tasks** (51 removed)

### Remaining Tasks

#### Any2AnyRetrieval (6)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (3)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 5 (was 6)

## Threshold 0.81

**184 → 14 tasks** (52 removed)

### Remaining Tasks

#### Any2AnyRetrieval (6)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (3)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 5 (was 6)
- Types: 4 (was 6)

## Threshold 0.8

**184 → 13 tasks** (53 removed)

### Remaining Tasks

#### Any2AnyRetrieval (6)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **VGGSoundV** - v2c, Web

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 5 (was 6)
- Types: 4 (was 6)

## Threshold 0.7

**184 → 10 tasks** (56 removed)

### Remaining Tasks

#### Any2AnyRetrieval (4)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **VATEXT2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (1)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 5 (was 6)
- Types: 4 (was 6)

## Threshold 0.6

**184 → 8 tasks** (58 removed)

### Remaining Tasks

#### Any2AnyRetrieval (2)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (1)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music

#### VideoZeroshotClassification (4)
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 5 (was 6)
- Types: 4 (was 6)

## Threshold 0.5

**184 → 8 tasks** (58 removed)

### Remaining Tasks

#### Any2AnyRetrieval (2)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web

#### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

#### VideoClassification (1)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music

#### VideoPairClassification (1)
- **RAVDESSAVVPairClassification** - v2v, Spoken

#### VideoZeroshotClassification (3)
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 6 (was 6)
- Types: 5 (was 6)

## Recommended MVEB Task List (threshold=0.85)

**Total: 18 tasks**

### Any2AnyRetrieval (7)
- **AVMemeExamT2VRetrieval** - t2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **MSVDV2TRetrieval** - v2t, Web, Spoken
- **Panda70MT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken

### VideoCentricQA (1)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web

### VideoClassification (4)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundV** - v2c, Web

### VideoPairClassification (1)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene

### VideoZeroshotClassification (5)
- **Kinetics400ZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **RAVDESSVZeroShot** - v2t, Spoken
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseVideoZeroShot** - v2t, Scene, AudioScene, Music, Entertainment

### Code for benchmarks.py

```python
tasks=get_tasks(
    tasks=[
        # Any2AnyRetrieval (7)
        "AVMemeExamT2VRetrieval",
        "ActivityNetCaptionsT2VRetrieval",
        "AudioCapsAVT2VRetrieval",
        "MSVDV2TRetrieval",
        "Panda70MT2VRetrieval",
        "VALOR32KT2VRetrieval",
        "VATEXT2VRetrieval",
        # VideoCentricQA (1)
        "OmniVideoBenchVideoCentricQA",
        # VideoClassification (4)
        "AVMemeVideoClassification",
        "BreakfastClassification",
        "Kinetics700V",
        "VGGSoundV",
        # VideoPairClassification (1)
        "HumanAnimalCartoonVPairClassification",
        # VideoZeroshotClassification (5)
        "Kinetics400ZeroShot",
        "MELDVideoZeroShot",
        "RAVDESSVZeroShot",
        "UCF101VideoZeroShotClassification",
        "WorldSenseVideoZeroShot",
    ]
),
```
