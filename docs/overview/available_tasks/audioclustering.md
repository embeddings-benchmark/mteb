
# AudioClustering

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 10

#### AmbientAcousticContextClustering

Clustering task based on a subset of the Ambient Acoustic Context dataset containing 1-second segments for workplace activities.

**Dataset:** [`AdnanElAssadi/ambient-acoustic-context-small`](https://huggingface.co/datasets/AdnanElAssadi/ambient-acoustic-context-small) • **License:** not specified • [Learn more →](https://dl.acm.org/doi/10.1145/3379503.3403535)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{10.1145/3379503.3403535,
      address = {New York, NY, USA},
      articleno = {33},
      author = {Park, Chunjong and Min, Chulhong and Bhattacharya, Sourav and Kawsar, Fahim},
      booktitle = {22nd International Conference on Human-Computer Interaction with Mobile Devices and Services},
      doi = {10.1145/3379503.3403535},
      isbn = {9781450375160},
      keywords = {Acoustic ambient context, Conversational agents},
      location = {Oldenburg, Germany},
      numpages = {9},
      publisher = {Association for Computing Machinery},
      series = {MobileHCI '20},
      title = {Augmenting Conversational Agents with Ambient Acoustic Contexts},
      url = {https://doi.org/10.1145/3379503.3403535},
      year = {2020},
    }

    ```




#### CREMA_DClustering

Emotion clustering task with audio data for 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad.

**Dataset:** [`silky1708/CREMA-D`](https://huggingface.co/datasets/silky1708/CREMA-D) • **License:** http://opendatacommons.org/licenses/odbl/1.0/ • [Learn more →](https://huggingface.co/datasets/silky1708/CREMA-D)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | eng | Speech | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @article{cao2014crema,
      author = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
      journal = {IEEE transactions on affective computing},
      number = {4},
      pages = {377--390},
      publisher = {IEEE},
      title = {Crema-d: Crowd-sourced emotional multimodal actors dataset},
      volume = {5},
      year = {2014},
    }

    ```




#### ESC50Clustering

The ESC-50 dataset contains 2,000 labeled environmental audio recordings evenly distributed across 50 classes (40 clips per class). These classes are organized into 5 broad categories: animal sounds, natural soundscapes & water sounds, human (non-speech) sounds, interior/domestic sounds, and exterior/urban noises. This task evaluates unsupervised clustering performance on environmental audio recordings.

**Dataset:** [`ashraq/esc50`](https://huggingface.co/datasets/ashraq/esc50) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ashraq/esc50)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | zxx | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{piczak2015dataset,
      author = {Piczak, Karol J.},
      booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
      date = {2015-10-13},
      doi = {10.1145/2733373.2806390},
      isbn = {978-1-4503-3459-4},
      location = {{Brisbane, Australia}},
      pages = {1015--1018},
      publisher = {{ACM Press}},
      title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
      url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
    }

    ```




#### GTZANGenreClustering

Music genre clustering task based on GTZAN dataset with 10 music genres.

**Dataset:** [`silky1708/GTZAN-Genre`](https://huggingface.co/datasets/silky1708/GTZAN-Genre) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/silky1708/GTZAN-Genre)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | zxx | Music | human-annotated | found |



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




#### MusicGenreClustering

Clustering music recordings in 9 different genres.

**Dataset:** [`mteb/music-genre`](https://huggingface.co/datasets/mteb/music-genre) • **License:** not specified • [Learn more →](https://www-ai.cs.tu-dortmund.de/audio.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | zxx | Music | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{homburg2005benchmark,
      author = {Homburg, Helge and Mierswa, Ingo and M{\"o}ller, B{\"u}lent and Morik, Katharina and Wurst, Michael},
      booktitle = {ISMIR},
      pages = {528--31},
      title = {A Benchmark Dataset for Audio Classification and Clustering.},
      volume = {2005},
      year = {2005},
    }

    ```




#### VehicleSoundClustering

Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class))

**Dataset:** [`DynamicSuperb/Vehicle_sounds_classification_dataset`](https://huggingface.co/datasets/DynamicSuperb/Vehicle_sounds_classification_dataset) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/DynamicSuperb/Vehicle_sounds_classification_dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | zxx | Scene | derived | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{inproceedings,
      author = {Bazilinskyy, Pavlo and Aa, Arne and Schoustra, Michael and Spruit, John and Staats, Laurens and van der Vlist, Klaas Jan and de Winter, Joost},
      month = {05},
      pages = {},
      title = {An auditory dataset of passing vehicles recorded with a smartphone},
      year = {2018},
    }

    ```




#### VoiceGenderClustering

Clustering audio recordings based on gender (male vs female).

**Dataset:** [`mmn3690/voice-gender-clustering`](https://huggingface.co/datasets/mmn3690/voice-gender-clustering) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mmn3690/voice-gender-clustering)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Chung18b,
      author = {Joon Son Chung and Arsha Nagrani and Andrew Zisserman},
      booktitle = {Proceedings of Interspeech},
      title = {VoxCeleb2: Deep Speaker Recognition},
      year = {2018},
    }

    ```




#### VoxCelebClustering

Clustering task based on the VoxCeleb dataset for sentiment analysis, clustering by positive/negative sentiment.

**Dataset:** [`DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb`](https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{shon2022sluenewbenchmarktasks,
      archiveprefix = {arXiv},
      author = {Suwon Shon and Ankita Pasad and Felix Wu and Pablo Brusco and Yoav Artzi and Karen Livescu and Kyu J. Han},
      eprint = {2111.10367},
      primaryclass = {cs.CL},
      title = {SLUE: New Benchmark Tasks for Spoken Language Understanding Evaluation on Natural Speech},
      url = {https://arxiv.org/abs/2111.10367},
      year = {2022},
    }

    ```




#### VoxPopuliAccentClustering

Clustering English speech samples by non-native accent from European Parliament recordings.

**Dataset:** [`mteb/voxpopuli-accent-clustering`](https://huggingface.co/datasets/mteb/voxpopuli-accent-clustering) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/facebook/voxpopuli)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{wang-etal-2021-voxpopuli,
      address = {Online},
      author = {Wang, Changhan  and
    Riviere, Morgane  and
    Lee, Ann  and
    Wu, Anne  and
    Talnikar, Chaitanya  and
    Haziza, Daniel  and
    Williamson, Mary  and
    Pino, Juan  and
    Dupoux, Emmanuel},
      booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
      doi = {10.18653/v1/2021.acl-long.80},
      month = aug,
      pages = {993--1003},
      publisher = {Association for Computational Linguistics},
      title = {{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
      url = {https://aclanthology.org/2021.acl-long.80},
      year = {2021},
    }

    ```




#### VoxPopuliGenderClustering

Subsampled Dataset for clustering speech samples by speaker gender (male/female) from European Parliament recordings.

**Dataset:** [`AdnanElAssadi/mini-voxpopuli`](https://huggingface.co/datasets/AdnanElAssadi/mini-voxpopuli) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/facebook/voxpopuli)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | deu, eng, fra, pol, spa | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{wang-etal-2021-voxpopuli,
      address = {Online},
      author = {Wang, Changhan  and
    Riviere, Morgane  and
    Lee, Ann  and
    Wu, Anne  and
    Talnikar, Chaitanya  and
    Haziza, Daniel  and
    Williamson, Mary  and
    Pino, Juan  and
    Dupoux, Emmanuel},
      booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
      doi = {10.18653/v1/2021.acl-long.80},
      month = aug,
      pages = {993--1003},
      publisher = {Association for Computational Linguistics},
      title = {{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
      url = {https://aclanthology.org/2021.acl-long.80},
      year = {2021},
    }

    ```
