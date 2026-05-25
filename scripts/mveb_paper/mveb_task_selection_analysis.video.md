# MVEB Task Selection — scope: `video`

MVEB(video) — V-only encoders (vjepa2, etc.)

## Pre-selection filters

- Source MVEB(extended): **184** tasks
- After scope filter (`video`): **31** (-152)
- After annotation-provenance filter: **31** (-0)
- After saturation/floor filter (best≤0.93, spread≥0.05, n≥3): **22** (-9)

- Must-include tasks in scope: **0** (bypass annotation and saturation filters)

### Dropped — saturated, floor, or low-support

- `Diving48Classification.V2` — floor (spread=0.029 < 0.05)
- `HumanAnimalCartoonV` — saturated (best=0.933 > 0.93)
- `UCF101VideoClassification` — saturated (best=0.943 > 0.93)
- `MELDEmotionVideoClustering` — floor (spread=0.012 < 0.05)
- `UCF101VideoClustering` — saturated (best=0.941 > 0.93)
- `AVEDatasetVPairClassification` — saturated (best=0.970 > 0.93)
- `AVSpeakerBenchPairClassification` — saturated (best=0.992 > 0.93)
- `MELDVPairClassification` — floor (spread=0.041 < 0.05)
- `VinogroundPairClassification` — floor (spread=0.031 < 0.05)

# MVEB Task Selection Analysis

## Overview
- **Source pool**: MVEB(extended) with 184 tasks
- **Working pool**: 22 tasks
- **Goal**: Select non-redundant tasks while preserving coverage

## Selection Rules

1. **Retrieval direction preference**: For task families with both V2T and T2V, prefer T2V (text-to-video)
2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task
3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage

## Protected Tasks (Unique Coverage): 1

### Unique Language
- AVMemeVideoClassification (unique: bos)
- AVMemeVideoClassification (unique: bre)
- AVMemeVideoClassification (unique: deu)
- AVMemeVideoClassification (unique: fas)
- AVMemeVideoClassification (unique: fin)
- AVMemeVideoClassification (unique: fra)
- AVMemeVideoClassification (unique: hin)
- AVMemeVideoClassification (unique: ita)
- AVMemeVideoClassification (unique: jpn)
- AVMemeVideoClassification (unique: kor)
- AVMemeVideoClassification (unique: por)
- AVMemeVideoClassification (unique: rus)
- AVMemeVideoClassification (unique: spa)
- AVMemeVideoClassification (unique: tel)
- AVMemeVideoClassification (unique: zho)

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
| ebind-av (encord-team/ebind-audio-vision) | 20h 46m | 22/22 |
| pe-av-small (facebook/pe-av-small) | 21h 19m | 22/22 |
| LCO-Embedding-Omni-7B (LCO-Embedding/LCO-Embedding-Omni-7B) | 44h 23m | 22/22 |
| Qwen2.5-Omni-7B (Qwen/Qwen2.5-Omni-7B) | 43h 24m | 22/22 |

## Selection Results Summary

| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson | ebind-av | pe-av-small | LCO-Embedding-Omni-7B | Qwen2.5-Omni-7B |
|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|--- | --- | --- | ---|
| 0.99 | 15 | 0 | 10 | 3 | 0 | 2 | 0 | 0 | 16 | 6 | 0.7730 | 0.9826 | 18h 30m | 18h 41m | 39h 3m | 37h 44m |
| 0.98 | 15 | 0 | 10 | 3 | 0 | 2 | 0 | 0 | 16 | 6 | 0.7730 | 0.9826 | 18h 30m | 18h 41m | 39h 3m | 37h 44m |
| 0.97 | 14 | 0 | 9 | 3 | 0 | 2 | 0 | 0 | 16 | 6 | 0.7785 | 0.9811 | 12h 44m | 12h 16m | 24h 52m | 26h 56m |
| 0.96 | 14 | 0 | 10 | 2 | 0 | 2 | 0 | 0 | 16 | 6 | 0.7950 | 0.9807 | 13h 8m | 12h 11m | 24h 40m | 26h 54m |
| 0.95 | 14 | 0 | 9 | 3 | 0 | 2 | 0 | 0 | 16 | 5 | 0.7840 | 0.9800 | 12h 54m | 11h 57m | 24h 8m | 26h 32m |
| 0.93 | 12 | 0 | 7 | 3 | 0 | 2 | 0 | 0 | 16 | 5 | 0.6960 | 0.9785 | 8h 15m | 9h 19m | 18h 57m | 19h 24m |
| 0.9 | 12 | 0 | 7 | 3 | 0 | 2 | 0 | 0 | 16 | 5 | 0.6960 | 0.9785 | 8h 15m | 9h 19m | 18h 57m | 19h 24m |
| 0.88 | 10 | 0 | 6 | 2 | 0 | 2 | 0 | 0 | 16 | 5 | 0.7235 | 0.9733 | 4h 31m | 7h 10m | 14h 33m | 14h 35m |
| 0.87 | 9 | 0 | 6 | 1 | 0 | 2 | 0 | 0 | 16 | 6 | 0.6465 | 0.9682 | 3h 50m | 6h 8m | 12h 38m | 12h 38m |
| 0.85 | 9 | 0 | 6 | 0 | 0 | 3 | 0 | 0 | 16 | 6 | 0.7785 | 0.9753 | 4h 16m | 6h 22m | 13h 12m | 13h 14m |
| 0.84 | 9 | 0 | 6 | 0 | 0 | 3 | 0 | 0 | 16 | 6 | 0.7785 | 0.9753 | 4h 16m | 6h 22m | 13h 12m | 13h 14m |
| 0.83 | 9 | 0 | 6 | 0 | 0 | 3 | 0 | 0 | 16 | 6 | 0.7785 | 0.9753 | 4h 16m | 6h 22m | 13h 12m | 13h 14m |
| 0.82 | 7 | 0 | 3 | 1 | 0 | 3 | 0 | 0 | 16 | 6 | 0.6795 | 0.9721 | 2h 30m | 4h 31m | 9h 17m | 9h 21m |
| 0.81 | 6 | 0 | 3 | 1 | 0 | 2 | 0 | 0 | 16 | 6 | 0.7455 | 0.9731 | 2h 9m | 4h 9m | 8h 26m | 8h 31m |
| 0.8 | 5 | 0 | 2 | 1 | 0 | 2 | 0 | 0 | 16 | 6 | 0.7180 | 0.9580 | 2h 7m | 4h 4m | 8h 17m | 8h 25m |
| 0.7 | 4 | 0 | 2 | 1 | 0 | 1 | 0 | 0 | 16 | 6 | 0.7290 | 0.9305 | 1h 14m | 1h 10m | 2h 37m | 2h 44m |
| 0.6 | 4 | 0 | 2 | 1 | 0 | 1 | 0 | 0 | 16 | 6 | 0.7290 | 0.9305 | 1h 14m | 1h 10m | 2h 37m | 2h 44m |
| 0.5 | 4 | 0 | 2 | 1 | 0 | 1 | 0 | 0 | 16 | 6 | 0.7290 | 0.9305 | 1h 14m | 1h 10m | 2h 37m | 2h 44m |

*Working pool: 22 tasks, 16 langs, 6 doms*

*Spearman/Pearson: Correlation of average model scores between selected tasks and full MVEB(extended)*

## Threshold 0.99

**184 → 15 tasks** (7 removed)

### Remaining Tasks

#### VideoClassification (10)
- **AVEDatasetVideoClassification** - v2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics400V** - v2c, Web, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (3)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.98

**184 → 15 tasks** (7 removed)

### Remaining Tasks

#### VideoClassification (10)
- **AVEDatasetVideoClassification** - v2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics400V** - v2c, Web, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (3)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.97

**184 → 14 tasks** (8 removed)

### Remaining Tasks

#### VideoClassification (9)
- **AVEDatasetVideoClassification** - v2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (3)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.96

**184 → 14 tasks** (8 removed)

### Remaining Tasks

#### VideoClassification (10)
- **AVEDatasetVideoClassification** - v2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **MusicAVQACLSVideoClassification** - v2c, Music
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (2)
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.95

**184 → 14 tasks** (8 removed)

### Remaining Tasks

#### VideoClassification (9)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **MusicAVQACLSVideoClassification** - v2c, Music
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundV** - v2c, Web

#### VideoClustering (3)
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 5 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.93

**184 → 12 tasks** (10 removed)

### Remaining Tasks

#### VideoClassification (7)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **MusicAVQACLSVideoClassification** - v2c, Music

#### VideoClustering (3)
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 5 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.9

**184 → 12 tasks** (10 removed)

### Remaining Tasks

#### VideoClassification (7)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **MusicAVQACLSVideoClassification** - v2c, Music

#### VideoClustering (3)
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 5 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.88

**184 → 10 tasks** (12 removed)

### Remaining Tasks

#### VideoClassification (6)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **MusicAVQACLSVideoClassification** - v2c, Music

#### VideoClustering (2)
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainVideoClustering** - v2c, Scene, Web, Entertainment

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 5 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.87

**184 → 9 tasks** (13 removed)

### Remaining Tasks

#### VideoClassification (6)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **RAVDESSVideoClustering** - v2c, Spoken

#### VideoPairClassification (2)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.85

**184 → 9 tasks** (13 removed)

### Remaining Tasks

#### VideoClassification (6)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 2 (was 3)

## Threshold 0.84

**184 → 9 tasks** (13 removed)

### Remaining Tasks

#### VideoClassification (6)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 2 (was 3)

## Threshold 0.83

**184 → 9 tasks** (13 removed)

### Remaining Tasks

#### VideoClassification (6)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 2 (was 3)

## Threshold 0.82

**184 → 7 tasks** (15 removed)

### Remaining Tasks

#### VideoClassification (3)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **MELDSpeakerVideoClustering** - v2c, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.81

**184 → 6 tasks** (16 removed)

### Remaining Tasks

#### VideoClassification (3)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **MELDSpeakerVideoClustering** - v2c, Entertainment

#### VideoPairClassification (2)
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.8

**184 → 5 tasks** (17 removed)

### Remaining Tasks

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **MELDSpeakerVideoClustering** - v2c, Entertainment

#### VideoPairClassification (2)
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.7

**184 → 4 tasks** (18 removed)

### Remaining Tasks

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **MELDSpeakerVideoClustering** - v2c, Entertainment

#### VideoPairClassification (1)
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.6

**184 → 4 tasks** (18 removed)

### Remaining Tasks

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **MELDSpeakerVideoClustering** - v2c, Entertainment

#### VideoPairClassification (1)
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Threshold 0.5

**184 → 4 tasks** (18 removed)

### Remaining Tasks

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (1)
- **MELDSpeakerVideoClustering** - v2c, Entertainment

#### VideoPairClassification (1)
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Coverage
- Languages: 16 (was 16)
- Domains: 6 (was 6)
- Categories: 2 (was 2)
- Types: 3 (was 3)

## Recommended MVEB Task List (threshold=0.85)

**Total: 9 tasks**

### VideoClassification (6)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics600V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVPairClassification** - v2v, Music
- **RAVDESSAVVPairClassification** - v2v, Spoken

### Code for benchmarks.py

```python
tasks=get_tasks(
    tasks=[
        # VideoClassification (6)
        "AVMemeVideoClassification",
        "BreakfastClassification",
        "HMDB51Classification",
        "Kinetics600V",
        "MELDVideoClassification",
        "WorldSenseVideoClassification",
        # VideoPairClassification (3)
        "HumanAnimalCartoonVPairClassification",
        "MusicAVQAVPairClassification",
        "RAVDESSAVVPairClassification",
    ]
),
```
