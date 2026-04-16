
# AudioPairClassification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 5

#### CREMADPairClassification

Classifying pairs as having same or different emotions in actor's voice recordings of text spoken in 6 different emotions

**Dataset:** [`mteb/CREMADPairClassification`](https://huggingface.co/datasets/mteb/CREMADPairClassification) • **License:** http://opendatacommons.org/licenses/odbl/1.0/ • [Learn more →](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | max_ap | eng | Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @article{Cao2014-ih,
      author = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur,
    Ruben C and Nenkova, Ani and Verma, Ragini},
      copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html},
      journal = {IEEE Transactions on Affective Computing},
      keywords = {Emotional corpora; facial expression; multi-modal recognition;
    voice expression},
      language = {en},
      month = oct,
      number = {4},
      pages = {377--390},
      publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
      title = {{CREMA-D}: Crowd-sourced emotional multimodal actors dataset},
      volume = {5},
      year = {2014},
    }

    ```




#### ESC50PairClassification

Environmental Sound Classification Dataset.

**Dataset:** [`mteb/ESC50PairClassification`](https://huggingface.co/datasets/mteb/ESC50PairClassification) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ashraq/esc50)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | max_ap | zxx | Encyclopaedic | human-annotated | found |



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




#### NMSQAPairClassification

A textless Q&A dataset. Given a pair of audio question and audio answer, is the answer relevant to the question?

**Dataset:** [`mteb/NMSQAPairClassification`](https://huggingface.co/datasets/mteb/NMSQAPairClassification) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.researchgate.net/publication/311458869_FMA_A_Dataset_For_Music_Analysis)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | max_ap | eng | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{lin2022dualdiscretespokenunit,
      archiveprefix = {arXiv},
      author = {Guan-Ting Lin and Yung-Sung Chuang and Ho-Lam Chung and Shu-wen Yang and Hsuan-Jui Chen and Shuyan Dong and Shang-Wen Li and Abdelrahman Mohamed and Hung-yi Lee and Lin-shan Lee},
      eprint = {2203.04911},
      primaryclass = {cs.CL},
      title = {DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering},
      url = {https://arxiv.org/abs/2203.04911},
      year = {2022},
    }

    ```




#### VocalSoundPairClassification

Recognizing whether two audio clips are the same human vocal expression (laughing, sighing, etc.)

**Dataset:** [`mteb/VocalSoundPairClassification`](https://huggingface.co/datasets/mteb/VocalSoundPairClassification) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | max_ap | eng | Spoken | human-annotated | found |



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




#### VoxPopuliAccentPairClassification

Classifying same or different regional accent of English

**Dataset:** [`mteb/VoxPopuliAccentPairClassification`](https://huggingface.co/datasets/mteb/VoxPopuliAccentPairClassification) • **License:** cc0-1.0 • [Learn more →](https://aclanthology.org/2021.acl-long.80/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | max_ap | eng | Spoken | human-annotated | created |



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
      editor = {Zong, Chengqing  and
    Xia, Fei  and
    Li, Wenjie  and
    Navigli, Roberto},
      month = aug,
      pages = {993--1003},
      publisher = {Association for Computational Linguistics},
      title = {{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
      url = {https://aclanthology.org/2021.acl-long.80/},
      year = {2021},
    }

    ```
