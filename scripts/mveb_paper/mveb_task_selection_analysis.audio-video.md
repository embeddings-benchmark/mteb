# MVEB Task Selection — scope: `audio-video`

MVEB(text, audio, video) — full A+V+T encoders (pe-av, ebind-av, omni, +)

## Pre-selection filters

- Source MVEB(extended): **184** tasks
- After scope filter (`audio-video`): **183** (-0)
- After annotation-provenance filter: **169** (-14)
- After saturation/floor filter (best≤0.93, spread≥0.05, n≥5): **144** (-25)

- Must-include tasks in scope: **39** (bypass annotation and saturation filters)

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
- `HumanAnimalCartoonV` — saturated (best=0.933 > 0.93)
- `UCF101VideoClassification` — saturated (best=0.943 > 0.93)
- `MELDEmotionVideoClustering` — floor (spread=0.012 < 0.05)
- `UCF101VideoClustering` — saturated (best=0.941 > 0.93)
- `HumanAnimalCartoonZeroShot` — saturated (best=0.938 > 0.93)
- `AVEDatasetZeroShot` — saturated (best=0.940 > 0.93)
- `AVEDatasetVPairClassification` — saturated (best=0.970 > 0.93)
- `AVSpeakerBenchPairClassification` — saturated (best=0.992 > 0.93)
- `MELDVPairClassification` — floor (spread=0.041 < 0.05)
- `AVEDatasetVAPairClassification` — saturated (best=0.963 > 0.93)
- `Shot2Story20KT2VRetrieval` — saturated (best=0.991 > 0.93)
- `TUNABenchT2VRetrieval` — saturated (best=0.986 > 0.93)
- `VGGSoundAVT2VRetrieval` — saturated (best=0.988 > 0.93)
- `Shot2Story20KV2TRetrieval` — saturated (best=0.986 > 0.93)
- `TUNABenchV2TRetrieval` — saturated (best=0.983 > 0.93)
- `VGGSoundAVV2TRetrieval` — saturated (best=0.989 > 0.93)
- `Shot2Story20KT2VARetrieval` — saturated (best=0.995 > 0.93)
- `VGGSoundAVT2VARetrieval` — saturated (best=0.991 > 0.93)
- `Shot2Story20KVA2TRetrieval` — saturated (best=0.990 > 0.93)
- `VGGSoundAVVA2TRetrieval` — saturated (best=0.987 > 0.93)
- `Shot2Story20KAT2VRetrieval` — saturated (best=0.994 > 0.93)
- `VATEXAT2VRetrieval` — saturated (best=0.931 > 0.93)
- `VGGSoundAVAT2VRetrieval` — saturated (best=0.987 > 0.93)
- `Shot2Story20KVT2ARetrieval` — saturated (best=0.933 > 0.93)

### Must-include tasks (kept regardless)

- `AVEDatasetAudioVideoClustering`
- `AVEDatasetClassification`
- `AVEDatasetVideoClustering`
- `AVMemeAudioVideoClassification`
- `AVMemeExamAT2VRetrieval`
- `AVMemeVideoClassification`
- `ActivityNetCaptionsT2VRetrieval`
- `AudioCapsAVAT2VRetrieval`
- `AudioCapsAVVA2TRetrieval`
- `BreakfastClassification`
- `DailyOmniVideoAudioCentricQA`
- `DiDeMoT2VRetrieval`
- `EgoSchemaVideoCentricQA`
- `HMDB51Classification`
- `HMDB51ZeroShot`
- `HumanAnimalCartoonVPairClassification`
- `MELDVideoClassification`
- `MELDVideoZeroShot`
- `MSRVTTT2V`
- `MSRVTTV2T`
- `MSVDT2VRetrieval`
- `NExTQAVideoCentricQA`
- `OmniVideoBenchVideoCentricQA`
- `RAVDESSAVVAPairClassification`
- `RAVDESSVideoClustering`
- `SomethingSomethingV2Classification`
- `UCF101VideoZeroShotClassification`
- `VALOR32KA2VRetrieval`
- `VALOR32KT2VARetrieval`
- `VALOR32KVT2ARetrieval`
- `VATEXT2VRetrieval`
- `VGGSoundVA`
- `VideoMMEShortVideoCentricQA`
- `VinogroundPairClassification`
- `WorldSense1MinDomainAudioVideoClustering`
- `WorldSense1MinVideoAudioCentricQA`
- `WorldSenseAudioVideoZeroShot`
- `WorldSenseVideoClassification`
- `YouCook2T2VRetrieval`

# MVEB Task Selection Analysis

## Overview
- **Source pool**: MVEB(extended) with 184 tasks
- **Working pool**: 144 tasks
- **Goal**: Select non-redundant tasks while preserving coverage

## Selection Rules

1. **Retrieval direction preference**: For task families with both V2T and T2V, prefer T2V (text-to-video)
2. **Correlation-based redundancy**: Remove tasks with Spearman ρ > threshold to a retained task
3. **Coverage preservation**: Protect tasks with unique language/domain/type coverage

## Protected Tasks (Unique Coverage): 39

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
| ebind-av (encord-team/ebind-audio-vision) | 106h 39m | 144/144 |
| pe-av-small (facebook/pe-av-small) | 139h 38m | 144/144 |
| LCO-Embedding-Omni-7B (LCO-Embedding/LCO-Embedding-Omni-7B) | 258h 43m | 144/144 |
| Qwen2.5-Omni-7B (Qwen/Qwen2.5-Omni-7B) | 226h 58m | 144/144 |

## Selection Results Summary

| Threshold | Tasks | Retr | Class | Clust | MLC | Pair | ZS | QA | Langs | Doms | Spearman | Pearson | ebind-av | pe-av-small | LCO-Embedding-Omni-7B | Qwen2.5-Omni-7B |
|-----------|-------|------|-------|-------|-----|------|----|----|-------|------|----------|---------|--- | --- | --- | ---|
| 0.95 | 54 | 20 | 10 | 5 | 0 | 4 | 6 | 9 | 16 | 8 | 0.9821 | 0.9961 | 37h 26m | 53h 54m | 90h 57m | 79h 45m |
| 0.93 | 49 | 18 | 10 | 5 | 0 | 3 | 6 | 7 | 16 | 8 | 0.9821 | 0.9960 | 34h 36m | 45h 22m | 79h 1m | 69h 4m |
| 0.9 | 46 | 17 | 10 | 5 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9766 | 0.9959 | 24h 27m | 30h 16m | 56h 3m | 46h 37m |
| 0.88 | 44 | 16 | 9 | 5 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9656 | 0.9953 | 20h 27m | 27h 10m | 50h 11m | 40h 19m |
| 0.87 | 43 | 15 | 9 | 5 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9546 | 0.9955 | 20h 21m | 27h 3m | 49h 55m | 40h 4m |
| 0.85 | 42 | 14 | 9 | 5 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9546 | 0.9965 | 20h 0m | 26h 1m | 48h 19m | 38h 27m |
| 0.84 | 42 | 14 | 9 | 5 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9546 | 0.9965 | 20h 0m | 26h 1m | 48h 19m | 38h 27m |
| 0.83 | 42 | 14 | 9 | 5 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9546 | 0.9965 | 20h 0m | 26h 1m | 48h 19m | 38h 27m |
| 0.82 | 41 | 14 | 9 | 4 | 0 | 3 | 5 | 6 | 16 | 8 | 0.9546 | 0.9962 | 19h 38m | 24h 44m | 45h 53m | 36h 1m |
| 0.81 | 40 | 14 | 9 | 4 | 0 | 3 | 4 | 6 | 16 | 8 | 0.9381 | 0.9959 | 17h 1m | 22h 49m | 41h 36m | 31h 26m |
| 0.8 | 40 | 14 | 9 | 4 | 0 | 3 | 4 | 6 | 16 | 8 | 0.9381 | 0.9959 | 17h 1m | 22h 49m | 41h 36m | 31h 26m |
| 0.7 | 40 | 14 | 9 | 4 | 0 | 3 | 4 | 6 | 16 | 8 | 0.9381 | 0.9959 | 17h 1m | 22h 49m | 41h 36m | 31h 26m |
| 0.6 | 40 | 14 | 9 | 4 | 0 | 3 | 4 | 6 | 16 | 8 | 0.9381 | 0.9959 | 17h 1m | 22h 49m | 41h 36m | 31h 26m |
| 0.5 | 40 | 14 | 9 | 4 | 0 | 3 | 4 | 6 | 16 | 8 | 0.9381 | 0.9959 | 17h 1m | 22h 49m | 41h 36m | 31h 26m |

*Working pool: 144 tasks, 16 langs, 8 doms*

*Spearman/Pearson: Correlation of average model scores between selected tasks and full MVEB(extended)*

## Threshold 0.95

**184 → 54 tasks** (90 removed)

### Remaining Tasks

#### Any2AnyRetrieval (20)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MV2TRetrieval** - v2t, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **VGGSoundAVVT2ARetrieval** - vt2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (9)
- **AVSpeakerBenchVideoAudioCentricQA** - vat2t, Web
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoAudioCentricQA** - vat2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **PerceptionTestVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (10)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics400VA** - va2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (4)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **MusicAVQAVAPairClassification** - va2va, Music
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (6)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **Kinetics700VZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.93

**184 → 49 tasks** (95 removed)

### Remaining Tasks

#### Any2AnyRetrieval (18)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **Panda70MV2TRetrieval** - v2t, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (7)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoAudioCentricQA** - vat2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (10)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics400VA** - va2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (6)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **Kinetics700VZeroShot** - v2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.9

**184 → 46 tasks** (98 removed)

### Remaining Tasks

#### Any2AnyRetrieval (17)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **VATEXVA2TRetrieval** - va2t, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (10)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **Kinetics700V** - v2c, Web, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.88

**184 → 44 tasks** (100 removed)

### Remaining Tasks

#### Any2AnyRetrieval (16)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVT2VRetrieval** - t2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.87

**184 → 43 tasks** (101 removed)

### Remaining Tasks

#### Any2AnyRetrieval (15)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VARetrieval** - t2va, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.85

**184 → 42 tasks** (102 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.84

**184 → 42 tasks** (102 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.83

**184 → 42 tasks** (102 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.82

**184 → 41 tasks** (103 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (4)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.81

**184 → 40 tasks** (104 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (4)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **HMDB51ZeroShot** - v2t, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.8

**184 → 40 tasks** (104 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (4)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **HMDB51ZeroShot** - v2t, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.7

**184 → 40 tasks** (104 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (4)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **HMDB51ZeroShot** - v2t, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.6

**184 → 40 tasks** (104 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (4)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **HMDB51ZeroShot** - v2t, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Threshold 0.5

**184 → 40 tasks** (104 removed)

### Remaining Tasks

#### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

#### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

#### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

#### VideoClustering (4)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

#### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

#### VideoZeroshotClassification (4)
- **HMDB51ZeroShot** - v2t, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Coverage
- Languages: 16 (was 16)
- Domains: 8 (was 8)
- Categories: 14 (was 14)
- Types: 6 (was 6)

## Recommended MVEB Task List (threshold=0.85)

**Total: 42 tasks**

### Any2AnyRetrieval (14)
- **AVMemeExamAT2VRetrieval** - at2v, Web, Social
- **ActivityNetCaptionsT2VRetrieval** - t2v, Web, Spoken
- **AudioCapsAVAT2VRetrieval** - at2v, Encyclopaedic, Web
- **AudioCapsAVVA2TRetrieval** - va2t, Encyclopaedic, Web
- **DiDeMoT2VRetrieval** - t2v, Web, Spoken
- **MSRVTTT2V** - t2v, N/A
- **MSRVTTV2T** - v2t, N/A
- **MSVDT2VRetrieval** - t2v, Web, Spoken
- **VALOR32KA2VRetrieval** - a2v, Web, Spoken
- **VALOR32KT2VARetrieval** - t2va, Web, Spoken
- **VALOR32KVT2ARetrieval** - vt2a, Web, Spoken
- **VATEXT2VRetrieval** - t2v, Web, Spoken
- **VATEXV2ARetrieval** - v2a, Web, Spoken
- **YouCook2T2VRetrieval** - t2v, Web, Spoken

### VideoCentricQA (6)
- **DailyOmniVideoAudioCentricQA** - vat2t, Web
- **EgoSchemaVideoCentricQA** - vt2t, Web
- **NExTQAVideoCentricQA** - vt2t, Web
- **OmniVideoBenchVideoCentricQA** - vt2t, Web
- **VideoMMEShortVideoCentricQA** - vt2t, Web
- **WorldSense1MinVideoAudioCentricQA** - vat2t, Web

### VideoClassification (9)
- **AVEDatasetClassification** - va2c, Web, AudioScene
- **AVMemeAudioVideoClassification** - va2c, Web, Entertainment, Music
- **AVMemeVideoClassification** - v2c, Web, Entertainment, Music
- **BreakfastClassification** - v2c, Scene
- **HMDB51Classification** - v2c, Scene
- **MELDVideoClassification** - v2c, Entertainment
- **SomethingSomethingV2Classification** - v2c, Scene
- **VGGSoundVA** - va2c, Web
- **WorldSenseVideoClassification** - v2c, Scene, AudioScene, Music, Entertainment

### VideoClustering (5)
- **AVEDatasetAudioVideoClustering** - va2c, Spoken, Scene, Music
- **AVEDatasetVideoClustering** - v2c, Spoken, Scene, Music
- **MusicAVQACLSVideoClustering** - v2c, Music
- **RAVDESSVideoClustering** - v2c, Spoken
- **WorldSense1MinDomainAudioVideoClustering** - va2c, Scene, Web, Entertainment

### VideoPairClassification (3)
- **HumanAnimalCartoonVPairClassification** - v2v, Web, Scene
- **RAVDESSAVVAPairClassification** - va2va, Spoken
- **VinogroundPairClassification** - v2v, Scene

### VideoZeroshotClassification (5)
- **HMDB51ZeroShot** - v2t, Scene
- **Kinetics600VAZeroShot** - va2t, Web, Scene
- **MELDVideoZeroShot** - v2t, Entertainment
- **UCF101VideoZeroShotClassification** - va2t, Web, Scene
- **WorldSenseAudioVideoZeroShot** - va2t, Scene, AudioScene, Music, Entertainment

### Code for benchmarks.py

```python
tasks=get_tasks(
    tasks=[
        # Any2AnyRetrieval (14)
        "AVMemeExamAT2VRetrieval",
        "ActivityNetCaptionsT2VRetrieval",
        "AudioCapsAVAT2VRetrieval",
        "AudioCapsAVVA2TRetrieval",
        "DiDeMoT2VRetrieval",
        "MSRVTTT2V",
        "MSRVTTV2T",
        "MSVDT2VRetrieval",
        "VALOR32KA2VRetrieval",
        "VALOR32KT2VARetrieval",
        "VALOR32KVT2ARetrieval",
        "VATEXT2VRetrieval",
        "VATEXV2ARetrieval",
        "YouCook2T2VRetrieval",
        # VideoCentricQA (6)
        "DailyOmniVideoAudioCentricQA",
        "EgoSchemaVideoCentricQA",
        "NExTQAVideoCentricQA",
        "OmniVideoBenchVideoCentricQA",
        "VideoMMEShortVideoCentricQA",
        "WorldSense1MinVideoAudioCentricQA",
        # VideoClassification (9)
        "AVEDatasetClassification",
        "AVMemeAudioVideoClassification",
        "AVMemeVideoClassification",
        "BreakfastClassification",
        "HMDB51Classification",
        "MELDVideoClassification",
        "SomethingSomethingV2Classification",
        "VGGSoundVA",
        "WorldSenseVideoClassification",
        # VideoClustering (5)
        "AVEDatasetAudioVideoClustering",
        "AVEDatasetVideoClustering",
        "MusicAVQACLSVideoClustering",
        "RAVDESSVideoClustering",
        "WorldSense1MinDomainAudioVideoClustering",
        # VideoPairClassification (3)
        "HumanAnimalCartoonVPairClassification",
        "RAVDESSAVVAPairClassification",
        "VinogroundPairClassification",
        # VideoZeroshotClassification (5)
        "HMDB51ZeroShot",
        "Kinetics600VAZeroShot",
        "MELDVideoZeroShot",
        "UCF101VideoZeroShotClassification",
        "WorldSenseAudioVideoZeroShot",
    ]
),
```
