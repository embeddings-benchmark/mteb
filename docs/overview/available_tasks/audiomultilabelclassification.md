
# AudioMultilabelClassification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 6

#### AudioSet

AudioSet consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos.

**Dataset:** [`agkphysics/AudioSet`](https://huggingface.co/datasets/agkphysics/AudioSet) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/agkphysics/AudioSet)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | lrap | eng | Music, Scene, Speech, Web | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{45857,
      address = {New Orleans, LA},
      author = {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
      booktitle = {Proc. IEEE ICASSP 2017},
      title = {Audio Set: An ontology and human-labeled dataset for audio events},
      year = {2017},
    }

    ```




#### AudioSetMini

AudioSet consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos. This is a mini version that is sampled from the original dataset.

**Dataset:** [`mteb/audioset`](https://huggingface.co/datasets/mteb/audioset) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/agkphysics/AudioSet)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | lrap | eng | Music, Scene, Speech, Web | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{45857,
      address = {New Orleans, LA},
      author = {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
      booktitle = {Proc. IEEE ICASSP 2017},
      title = {Audio Set: An ontology and human-labeled dataset for audio events},
      year = {2017},
    }

    ```




#### BirdSet

BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics

**Dataset:** [`DBD-research-group/BirdSet`](https://huggingface.co/datasets/DBD-research-group/BirdSet) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/DBD-research-group/BirdSet)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | zxx | Bioacoustics, Speech, Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{rauch2024birdsetlargescaledatasetaudio,
      archiveprefix = {arXiv},
      author = {Lukas Rauch and Raphael Schwinger and Moritz Wirth and René Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz},
      eprint = {2403.10380},
      primaryclass = {cs.SD},
      title = {BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics},
      url = {https://arxiv.org/abs/2403.10380},
      year = {2024},
    }

    ```




#### FSD2019Kaggle

Multilabel Audio Classification.

**Dataset:** [`confit/fsdkaggle2019-parquet`](https://huggingface.co/datasets/confit/fsdkaggle2019-parquet) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/confit/fsdkaggle2019-parquet)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Web | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @dataset{eduardo_fonseca_2020_3612637,
      author = {Eduardo Fonseca and
    Manoj Plakal and
    Frederic Font and
    Daniel P. W. Ellis and
    Xavier Serra},
      doi = {10.5281/zenodo.3612637},
      month = jan,
      publisher = {Zenodo},
      title = {FSDKaggle2019},
      url = {https://doi.org/10.5281/zenodo.3612637},
      version = {1.0},
      year = {2020},
    }

    ```




#### FSD50K

Multilabel Audio Classification on a subsampled version of FSD50K using 2048 samples

**Dataset:** [`mteb/fsd50k_mini`](https://huggingface.co/datasets/mteb/fsd50k_mini) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/fsd50k_mini)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Web | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{9645159,
      author = {Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
      doi = {10.1109/TASLP.2021.3133208},
      journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
      keywords = {Videos;Task analysis;Labeling;Vocabulary;Speech recognition;Ontologies;Benchmark testing;Audio dataset;sound event;recognition;classification;tagging;data collection;environmental sound},
      number = {},
      pages = {829-852},
      title = {FSD50K: An Open Dataset of Human-Labeled Sound Events},
      volume = {30},
      year = {2022},
    }

    ```




#### SIBFLEURS

Topic Classification for multilingual audio dataset. This dataset is a stratified and downsampled subset of the SIBFLEURS dataset, which is a collection of 1000+ hours of audio data in 100+ languages.

**Dataset:** [`mteb/sib-fleurs-multilingual-mini`](https://huggingface.co/datasets/mteb/sib-fleurs-multilingual-mini) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/WueNLP/sib-fleurs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | afr, amh, arb, asm, ast, ... (101) | Encyclopaedic | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{schmidt2025fleursslumassivelymultilingualbenchmark,
      archiveprefix = {arXiv},
      author = {Fabian David Schmidt and Ivan Vulić and Goran Glavaš and David Ifeoluwa Adelani},
      eprint = {2501.06117},
      primaryclass = {cs.CL},
      title = {Fleurs-SLU: A Massively Multilingual Benchmark for Spoken Language Understanding},
      url = {https://arxiv.org/abs/2501.06117},
      year = {2025},
    }

    ```
