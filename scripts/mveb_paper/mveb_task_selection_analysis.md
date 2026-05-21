# MVEB Task Selection Analysis

## Overview
- **Source pool**: MVEB(extended) with 184 tasks
- **Working pool**: 183 tasks
- **Goal**: Select non-redundant tasks while preserving coverage

## Selection Rules

1. **Retrieval direction preference**: For task families with both V2T and T2V, prefer T2V (text-to-video)
2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task
3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage

## Protected Tasks (Unique Coverage): 1

### Unique Language
- Diving48Classification.V2 (unique: zxx)

### Unique Domain
- Diving48Classification.V2 (unique: Sport)

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
| ebind-av (encord-team/ebind-audio-vision) | 122h 40m | 183/183 |
| pe-av-small (facebook/pe-av-small) | 159h 40m | 183/183 |
| LCO-Embedding-Omni-7B (LCO-Embedding/LCO-Embedding-Omni-7B) | 302h 39m | 183/183 |
| Qwen2.5-Omni-7B (Qwen/Qwen2.5-Omni-7B) | 267h 15m | 183/183 |

## Selection Results Summary

| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson | ebind-av | pe-av-small | LCO-Embedding-Omni-7B | Qwen2.5-Omni-7B |
|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|--- | --- | --- | ---|
| 0.95 | 55 | 34 | 8 | 3 | 0 | 3 | 3 | 4 | 17 | 9 | 0.9780 | 0.9978 | 46h 19m | 57h 33m | 104h 18m | 94h 27m |
| 0.93 | 47 | 27 | 7 | 3 | 0 | 2 | 4 | 4 | 17 | 9 | 0.9615 | 0.9974 | 41h 4m | 50h 5m | 88h 26m | 80h 15m |
| 0.9 | 38 | 19 | 9 | 1 | 0 | 3 | 3 | 3 | 17 | 9 | 0.9821 | 0.9974 | 39h 50m | 39h 34m | 76h 0m | 69h 53m |
| 0.88 | 35 | 18 | 7 | 1 | 0 | 3 | 4 | 2 | 17 | 9 | 0.9546 | 0.9958 | 28h 39m | 29h 26m | 55h 50m | 50h 0m |
| 0.87 | 32 | 16 | 7 | 1 | 0 | 2 | 4 | 2 | 17 | 9 | 0.9656 | 0.9973 | 26h 28m | 25h 46m | 48h 28m | 42h 43m |
| 0.85 | 27 | 13 | 7 | 1 | 0 | 1 | 3 | 2 | 17 | 9 | 0.9601 | 0.9963 | 21h 20m | 22h 14m | 40h 12m | 34h 33m |
| 0.84 | 23 | 10 | 6 | 1 | 0 | 2 | 2 | 2 | 17 | 9 | 0.9821 | 0.9965 | 13h 38m | 17h 17m | 30h 11m | 26h 26m |
| 0.83 | 22 | 10 | 6 | 1 | 0 | 1 | 2 | 2 | 17 | 9 | 0.9821 | 0.9965 | 13h 18m | 16h 55m | 29h 24m | 25h 37m |
| 0.82 | 21 | 9 | 6 | 1 | 0 | 1 | 2 | 2 | 17 | 9 | 0.9931 | 0.9970 | 13h 16m | 16h 50m | 29h 13m | 25h 27m |
| 0.81 | 21 | 9 | 7 | 1 | 0 | 0 | 2 | 2 | 17 | 9 | 0.9931 | 0.9973 | 13h 25m | 16h 34m | 28h 26m | 24h 53m |
| 0.8 | 21 | 9 | 7 | 1 | 0 | 0 | 2 | 2 | 17 | 8 | 0.9931 | 0.9965 | 13h 27m | 16h 16m | 27h 43m | 24h 29m |
| 0.7 | 14 | 7 | 3 | 0 | 0 | 1 | 1 | 2 | 17 | 8 | 0.9381 | 0.9924 | 5h 15m | 9h 56m | 15h 52m | 12h 12m |
| 0.6 | 13 | 7 | 2 | 0 | 0 | 1 | 1 | 2 | 17 | 7 | 0.9381 | 0.9908 | 4h 43m | 9h 31m | 14h 56m | 11h 33m |
| 0.5 | 13 | 7 | 2 | 0 | 0 | 1 | 1 | 2 | 17 | 7 | 0.9381 | 0.9908 | 4h 43m | 9h 31m | 14h 56m | 11h 33m |

*Working pool: 183 tasks, 17 langs, 9 doms*

*Spearman/Pearson: Correlation of average model scores between selected tasks and full MVEB(extended)*

## Threshold 0.95

**184 → 55 tasks** (128 removed)

### Remaining Tasks

#### Any2AnyRetrieval (34)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVA2VRetrieval** - a2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **DiDeMoVA2TRetrieval** - va2t, Web, Spoken
- **DiDeMoVT2ARetrieval** - vt2a, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSRVTTAT2V** - at2v, N/A
- **MSRVTTT2VA** - t2va, N/A
- **MSRVTTV2A** - v2a, N/A
- **MSRVTTVT2A** - vt2a, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MT2VARetrieval** - t2va, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VARetrieval** - t2va, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **Shot2Story20KVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KVT2ARetrieval** - vt2a, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KV2ARetrieval** - v2a, Web, Spoken
- **VALOR32KVA2TRetrieval** - va2t, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VATEXVT2ARetrieval** - vt2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2AT2VRetrieval** - at2v, Web, Spoken
- **YouCook2T2VARetrieval** - t2va, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

#### VideoCentricQA (4)
- **OmniVideoBenchVideoAudioCentricQA** - vat2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (8)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **Kinetics400VA** - va2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (3)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSAVClustering** - va2c, Spoken
- **UCF101AudioVideoClustering** - va2c, Web, Scene

#### VideoPairClassification (3)
- **AVSpeakerBenchPairClassification** - v2v, Spoken
- **MusicAVQAVAPairClassification** - va2va, Music
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (3)
- **HumanAnimalCartoonZeroShot** - v2t, Web, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.93

**184 → 47 tasks** (136 removed)

### Remaining Tasks

#### Any2AnyRetrieval (27)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVA2VRetrieval** - a2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **DiDeMoVA2TRetrieval** - va2t, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSRVTTV2A** - v2a, N/A
- **MSRVTTVT2A** - vt2a, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MT2VARetrieval** - t2va, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VARetrieval** - t2va, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **Shot2Story20KVA2TRetrieval** - va2t, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVA2TRetrieval** - va2t, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VATEXVT2ARetrieval** - vt2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

#### VideoCentricQA (4)
- **OmniVideoBenchVideoAudioCentricQA** - vat2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **Kinetics400VA** - va2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (3)
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSAVClustering** - va2c, Spoken
- **UCF101AudioVideoClustering** - va2c, Web, Scene

#### VideoPairClassification (2)
- **AVSpeakerBenchPairClassification** - v2v, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **BreakfastZeroShot** - v2t, Scene
- **HumanAnimalCartoonZeroShot** - v2t, Web, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 13 (was 14)
- Types: 6 (was 6)

## Threshold 0.9

**184 → 38 tasks** (145 removed)

### Remaining Tasks

#### Any2AnyRetrieval (19)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVA2VRetrieval** - a2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **DiDeMoVA2TRetrieval** - va2t, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MT2VARetrieval** - t2va, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

#### VideoCentricQA (3)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **Kinetics400VA** - va2c, Web, Scene
- **Kinetics700V** - v2c, Web, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (3)
- **AVSpeakerBenchPairClassification** - v2v, Spoken
- **HumanAnimalCartoonVAPairClassification** - va2va, Web, Scene
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (3)
- **BreakfastZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.88

**184 → 35 tasks** (148 removed)

### Remaining Tasks

#### Any2AnyRetrieval (18)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVA2VRetrieval** - a2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **DiDeMoVA2TRetrieval** - va2t, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MT2VARetrieval** - t2va, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **UCF101VideoAudioClassification** - va2c, Web, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (3)
- **AVSpeakerBenchPairClassification** - v2v, Spoken
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **BreakfastZeroShot** - v2t, Scene
- **HumanAnimalCartoonZeroShot** - v2t, Web, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.87

**184 → 32 tasks** (151 removed)

### Remaining Tasks

#### Any2AnyRetrieval (16)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **DiDeMoVA2TRetrieval** - va2t, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **UCF101VideoAudioClassification** - va2c, Web, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (2)
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **BreakfastZeroShot** - v2t, Scene
- **HumanAnimalCartoonZeroShot** - v2t, Web, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.85

**184 → 27 tasks** (156 removed)

### Remaining Tasks

#### Any2AnyRetrieval (13)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene
- **VGGSoundVA** - va2c, Web

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (1)
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (3)
- **BreakfastZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 13 (was 14)
- Types: 6 (was 6)

## Threshold 0.84

**184 → 23 tasks** (160 removed)

### Remaining Tasks

#### Any2AnyRetrieval (10)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (2)
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (2)
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.83

**184 → 22 tasks** (161 removed)

### Remaining Tasks

#### Any2AnyRetrieval (10)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (2)
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 13 (was 14)
- Types: 6 (was 6)

## Threshold 0.82

**184 → 21 tasks** (162 removed)

### Remaining Tasks

#### Any2AnyRetrieval (9)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (6)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (2)
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 13 (was 14)
- Types: 6 (was 6)

## Threshold 0.81

**184 → 21 tasks** (162 removed)

### Remaining Tasks

#### Any2AnyRetrieval (9)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **MusicAVQACLSAudioVideoClassification** - va2c, Music
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (1)
- **RAVDESSAVClustering** - va2c, Spoken

#### VideoZeroshotClassification (2)
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 9 (was 9)
- Categories: 12 (was 14)
- Types: 5 (was 6)

## Threshold 0.8

**184 → 21 tasks** (162 removed)

### Remaining Tasks

#### Any2AnyRetrieval (9)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVVA2TRetrieval** - va2t, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (7)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **MusicAVQACLSAudioVideoClassification** - va2c, Music
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene

#### VideoClustering (1)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music

#### VideoZeroshotClassification (2)
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 8 (was 9)
- Categories: 12 (was 14)
- Types: 5 (was 6)

## Threshold 0.7

**184 → 14 tasks** (169 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (3)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (1)
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 8 (was 9)
- Categories: 13 (was 14)
- Types: 5 (was 6)

## Threshold 0.6

**184 → 13 tasks** (170 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (1)
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 7 (was 9)
- Categories: 12 (was 14)
- Types: 5 (was 6)

## Threshold 0.5

**184 → 13 tasks** (170 removed)

### Remaining Tasks

#### Any2AnyRetrieval (7)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **MSRVTTA2V** - a2v, N/A
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken

#### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (2)
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport

#### VideoPairClassification (1)
- **RAVDESSAVVAPairClassification** - va2va, Spoken

#### VideoZeroshotClassification (1)
- **MELDVideoZeroShot** - v2t, Entertainment

### Coverage
- Languages: 17 (was 17)
- Domains: 7 (was 9)
- Categories: 12 (was 14)
- Types: 5 (was 6)

## Recommended MVEB Task List (threshold=0.85)

**Total: 27 tasks**

### Any2AnyRetrieval (13)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VARetrieval** - t2va, Web, Spoken
- **MSRVTTA2V** - a2v, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MVA2TRetrieval** - va2t, Web, Spoken
- **Shot2Story20KT2VRetrieval** - t2v, Web, Spoken
- **TUNABenchT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VGGSoundAVT2VARetrieval** - t2va, Web, Spoken
- **YouCook2VT2ARetrieval** - vt2a, Web, Spoken

### VideoCentricQA (2)
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

### VideoClassification (7)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **Diving48Classification.V2** - v2c, Sport
- **HMDB51Classification** - v2c, Scene
- **RAVDESSAVClassification** - va2c, Spoken
- **UCF101VideoAudioClassification** - va2c, Web, Scene
- **VGGSoundVA** - va2c, Web

### VideoClustering (1)
- **MusicAVQACLSVideoClustering** - v2c, Music

### VideoPairClassification (1)
- **VinogroundPairClassification** - v2v, Scene

### VideoZeroshotClassification (3)
- **BreakfastZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment

### Code for benchmarks.py

```python
tasks=get_tasks(
    tasks=[
        # Any2AnyRetrieval (13)
        "AVMemeExamAT2VRetrieval",
        "AudioCapsAVVA2TRetrieval",
        "DiDeMoT2VARetrieval",
        "MSRVTTA2V",
        "MSVDT2VRetrieval",
        "Panda70MVA2TRetrieval",
        "Shot2Story20KT2VRetrieval",
        "TUNABenchT2VRetrieval",
        "VALOR32KVT2ARetrieval",
        "VATEXT2VARetrieval",
        "VATEXV2ARetrieval",
        "VGGSoundAVT2VARetrieval",
        "YouCook2VT2ARetrieval",
        # VideoCentricQA (2)
        "OmniVideoBenchVideoCentricQA",
        "WorldSense1MinVideoAudioCentricQA",
        # VideoClassification (7)
        "AVEDatasetClassification",
        "AVMemeVideoClassification",
        "Diving48Classification.V2",
        "HMDB51Classification",
        "RAVDESSAVClassification",
        "UCF101VideoAudioClassification",
        "VGGSoundVA",
        # VideoClustering (1)
        "MusicAVQACLSVideoClustering",
        # VideoPairClassification (1)
        "VinogroundPairClassification",
        # VideoZeroshotClassification (3)
        "BreakfastZeroShot",
        "Kinetics600VAZeroShot",
        "MELDVideoZeroShot",
    ]
),
```
