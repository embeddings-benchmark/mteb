
# AudioReranking

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 5

#### ESC50AudioReranking

ESC-50 environmental sound dataset adapted for audio reranking. Given a query audio of environmental sounds, rank 5 relevant audio samples higher than 16 irrelevant ones from different sound classes. Contains 200 queries across 50 environmental sound categories for robust evaluation.

**Dataset:** [`mteb/ESC50AudioReranking`](https://huggingface.co/datasets/mteb/ESC50AudioReranking) • **License:** cc-by-3.0 • [Learn more →](https://github.com/karolpiczak/ESC-50)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | map_at_1000 | zxx | AudioScene | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{piczak2015esc,
      author = {Piczak, Karol J},
      booktitle = {Proceedings of the 23rd ACM international conference on Multimedia},
      pages = {1015--1018},
      title = {ESC: Dataset for Environmental Sound Classification},
      year = {2015},
    }

    ```




#### FSDnoisy18kAudioReranking

FSDnoisy18k sound event dataset adapted for audio reranking. Given a query audio with potential label noise, rank 4 relevant audio samples higher than 16 irrelevant ones from different sound classes. Contains 200 queries across 20 sound event categories.

**Dataset:** [`mteb/FSDnoisy18kAudioReranking`](https://huggingface.co/datasets/mteb/FSDnoisy18kAudioReranking) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/record/2529934)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | map_at_1000 | eng | AudioScene | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{fonseca2019fsdnoisy18k,
      author = {Fonseca, Eduardo and Plakal, Manoj and Ellis, Daniel P. W. and Font, Frederic and Favory, Xavier and Serra, Xavier},
      booktitle = {ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      organization = {IEEE},
      pages = {21--25},
      title = {Learning Sound Event Classifiers from Web Audio with Noisy Labels},
      year = {2019},
    }

    ```




#### GTZANAudioReranking

GTZAN music genre dataset adapted for audio reranking. Given a query audio from one of 10 music genres, rank 3 relevant audio samples higher than 10 irrelevant ones from different genres. Contains 100 queries across 10 music genres for comprehensive evaluation.

**Dataset:** [`mteb/GTZANAudioReranking`](https://huggingface.co/datasets/mteb/GTZANAudioReranking) • **License:** not specified • [Learn more →](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | map_at_1000 | zxx | Music | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{1021072,
      author = {Tzanetakis, G. and Cook, P.},
      doi = {10.1109/TSA.2002.800560},
      journal = {IEEE Transactions on Speech and Audio Processing},
      keywords = {Humans;Music information retrieval;Instruments;Computer science;Multiple signal classification;Signal analysis;Pattern recognition;Feature extraction;Wavelet analysis;Cultural differences},
      number = {5},
      pages = {293-302},
      title = {Musical genre classification of audio signals},
      volume = {10},
      year = {2002},
    }

    ```




#### UrbanSound8KAudioReranking

UrbanSound8K urban sound dataset adapted for audio reranking. Given a query audio of urban sounds, rank 4 relevant audio samples higher than 16 irrelevant ones from different urban sound classes. Contains 200 queries across 10 urban sound categories for comprehensive evaluation.

**Dataset:** [`mteb/UrbanSound8KAudioReranking`](https://huggingface.co/datasets/mteb/UrbanSound8KAudioReranking) • **License:** cc-by-4.0 • [Learn more →](https://urbansounddataset.weebly.com/urbansound8k.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | map_at_1000 | zxx | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Salamon2014ADA,
      author = {Justin Salamon and Christopher Jacoby and Juan Pablo Bello},
      journal = {Proceedings of the 22nd ACM international conference on Multimedia},
      title = {A Dataset and Taxonomy for Urban Sound Research},
      url = {https://api.semanticscholar.org/CorpusID:207217115},
      year = {2014},
    }

    ```




#### VocalSoundAudioReranking

VocalSound dataset adapted for audio reranking. Given a query vocal sound from one of 6 categories, rank 4 relevant vocal samples higher than 16 irrelevant ones from different vocal sound types. Contains 198 queries across 6 vocal sound categories for robust evaluation.

**Dataset:** [`mteb/VocalSoundAudioReranking`](https://huggingface.co/datasets/mteb/VocalSoundAudioReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | map_at_1000 | eng | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{inproceedings,
      author = {Gong, Yuan and Yu, Jin and Glass, James},
      doi = {10.1109/ICASSP43922.2022.9746828},
      month = {05},
      pages = {151-155},
      title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
      year = {2022},
    }

    ```
