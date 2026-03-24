---
icon: lucide/search
title: "Retrieval"
---

<style>
.nowrap-table th {
  white-space: nowrap;
}
</style>

# Retrieval

<!-- The following sections are auto-generated, please edit the construction script -->

<!-- START-TASKS -->


## Any2AnyMultilingualRetrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 3

#### `WITT2IRetrieval` { .model-copy }

Retrieve images based on multilingual descriptions.

**Dataset:** [`mteb/wit`](https://huggingface.co/datasets/mteb/wit) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, bul, dan, ell, eng, ... (11) | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{bugliarello2022iglue,
      author = {Bugliarello, Emanuele and Liu, Fangyu and Pfeiffer, Jonas and Reddy, Siva and Elliott, Desmond and Ponti, Edoardo Maria and Vuli{\'c}, Ivan},
      booktitle = {International Conference on Machine Learning},
      organization = {PMLR},
      pages = {2370--2392},
      title = {IGLUE: A benchmark for transfer learning across modalities, tasks, and languages},
      year = {2022},
    }
    
    ```
    



#### `XFlickr30kCoT2IRetrieval` { .model-copy }

Retrieve images based on multilingual descriptions.

**Dataset:** [`mteb/xflickrco`](https://huggingface.co/datasets/mteb/xflickrco) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, ind, jpn, rus, ... (8) | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{bugliarello2022iglue,
      author = {Bugliarello, Emanuele and Liu, Fangyu and Pfeiffer, Jonas and Reddy, Siva and Elliott, Desmond and Ponti, Edoardo Maria and Vuli{\'c}, Ivan},
      booktitle = {International Conference on Machine Learning},
      organization = {PMLR},
      pages = {2370--2392},
      title = {IGLUE: A benchmark for transfer learning across modalities, tasks, and languages},
      year = {2022},
    }
    
    ```
    



#### `XM3600T2IRetrieval` { .model-copy }

Retrieve images based on multilingual descriptions.

**Dataset:** [`mteb/xm3600`](https://huggingface.co/datasets/mteb/xm3600) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.emnlp-main.45/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, ben, ces, dan, deu, ... (38) | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thapliyal2022crossmodal,
      author = {Thapliyal, Ashish V and Tuset, Jordi Pont and Chen, Xi and Soricut, Radu},
      booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
      pages = {715--729},
      title = {Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset},
      year = {2022},
    }
    
    ```




## Any2AnyRetrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 89

#### `AudioCapsA2TRetrieval` { .model-copy }

Natural language description for any kind of audio in the wild.

**Dataset:** [`mteb/audiocaps_a2t`](https://huggingface.co/datasets/mteb/audiocaps_a2t) • **License:** mit • [Learn more →](https://audiocaps.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng, zxx | Encyclopaedic, Written | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{kim2019audiocaps,
      author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
      booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
      pages = {119--132},
      title = {Audiocaps: Generating captions for audios in the wild},
      year = {2019},
    }
    
    ```
    



#### `AudioCapsT2ARetrieval` { .model-copy }

Natural language description for any kind of audio in the wild.

**Dataset:** [`mteb/audiocaps_t2a`](https://huggingface.co/datasets/mteb/audiocaps_t2a) • **License:** mit • [Learn more →](https://audiocaps.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng, zxx | Encyclopaedic, Written | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{kim2019audiocaps,
      author = {Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
      booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
      pages = {119--132},
      title = {Audiocaps: Generating captions for audios in the wild},
      year = {2019},
    }
    
    ```
    



#### `AudioSetStrongA2TRetrieval` { .model-copy }

Retrieve all temporally-strong labeled events within 10s audio clips from the AudioSet Strongly-Labeled subset.

**Dataset:** [`mteb/audioset_strong_a2t`](https://huggingface.co/datasets/mteb/audioset_strong_a2t) • **License:** cc-by-4.0 • [Learn more →](https://research.google.com/audioset/download_strong.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | AudioScene | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{hershey2021benefittemporallystronglabelsaudio,
      archiveprefix = {arXiv},
      author = {Shawn Hershey and Daniel P W Ellis and Eduardo Fonseca and Aren Jansen and Caroline Liu and R Channing Moore and Manoj Plakal},
      eprint = {2105.07031},
      primaryclass = {cs.SD},
      title = {The Benefit Of Temporally-Strong Labels In Audio Event Classification},
      url = {https://arxiv.org/abs/2105.07031},
      year = {2021},
    }
    
    ```
    



#### `AudioSetStrongT2ARetrieval` { .model-copy }

Retrieve audio segments corresponding to a given sound event label from the AudioSet Strongly-Labeled 10s clips.

**Dataset:** [`mteb/audioset_strong_t2a`](https://huggingface.co/datasets/mteb/audioset_strong_t2a) • **License:** cc-by-4.0 • [Learn more →](https://research.google.com/audioset/download_strong.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | AudioScene | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{hershey2021benefittemporallystronglabelsaudio,
      archiveprefix = {arXiv},
      author = {Shawn Hershey and Daniel P W Ellis and Eduardo Fonseca and Aren Jansen and Caroline Liu and R Channing Moore and Manoj Plakal},
      eprint = {2105.07031},
      primaryclass = {cs.SD},
      title = {The Benefit Of Temporally-Strong Labels In Audio Event Classification},
      url = {https://arxiv.org/abs/2105.07031},
      year = {2021},
    }
    
    ```
    



#### `BLINKIT2IRetrieval` { .model-copy }

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`mteb/blink-it2i`](https://huggingface.co/datasets/mteb/blink-it2i) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.12390)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image (it2i) | eng | Encyclopaedic | derived | found | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{fu2024blink,
      author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
      journal = {arXiv preprint arXiv:2404.12390},
      title = {Blink: Multimodal large language models can see but not perceive},
      year = {2024},
    }
    
    ```
    



#### `BLINKIT2TRetrieval` { .model-copy }

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`mteb/blink-it2t`](https://huggingface.co/datasets/mteb/blink-it2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.12390)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | found | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{fu2024blink,
      author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
      journal = {arXiv preprint arXiv:2404.12390},
      title = {Blink: Multimodal large language models can see but not perceive},
      year = {2024},
    }
    
    ```
    



#### `CIRRIT2IRetrieval` { .model-copy }

Retrieve images based on texts and images.

**Dataset:** [`mteb/mbeir_cirr_task7`](https://huggingface.co/datasets/mteb/mbeir_cirr_task7) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image (it2i) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{liu2021image,
      author = {Liu, Zheyuan and Rodriguez-Opazo, Cristian and Teney, Damien and Gould, Stephen},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages = {2125--2134},
      title = {Image retrieval on real-life images with pre-trained vision-and-language models},
      year = {2021},
    }
    
    ```
    



#### `CMUArcticA2TRetrieval` { .model-copy }

Retrieve the correct transcription for an English speech segment. The dataset is derived from the phonetically balanced CMU Arctic single-speaker TTS corpora. The corpora contains 1150 samples based on read-aloud segments from books, which are out of copyright and derived from the Gutenberg project.

**Dataset:** [`mteb/CMU_Arctic_a2t`](https://huggingface.co/datasets/mteb/CMU_Arctic_a2t) • **License:** cc0-1.0 • [Learn more →](http://festvox.org/cmu_arctic/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @techreport{cmu-lti-03-177,
      author = {Clark, Rob and Richmond, Keith},
      institution = {Carnegie Mellon University, Language Technologies Institute},
      number = {CMU-LTI-03-177},
      title = {A detailed report on the CMU Arctic speech database},
      year = {2003},
    }
    
    ```
    



#### `CMUArcticT2ARetrieval` { .model-copy }

Retrieve the correct audio segment for an English transcription. The dataset is derived from the phonetically balanced CMU Arctic single-speaker TTS corpora. The corpora contains 1150 audio-text pairs based on read-aloud segments from public domain books originally sourced from the Gutenberg project.

**Dataset:** [`mteb/CMU_Arctic_t2a`](https://huggingface.co/datasets/mteb/CMU_Arctic_t2a) • **License:** cc0-1.0 • [Learn more →](http://festvox.org/cmu_arctic/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @techreport{cmu-lti-03-177,
      author = {Clark, Rob and Richmond, Keith},
      institution = {Carnegie Mellon University, Language Technologies Institute},
      number = {CMU-LTI-03-177},
      title = {A detailed report on the CMU Arctic speech database},
      year = {2003},
    }
    
    ```
    



#### `CUB200I2IRetrieval` { .model-copy }

Retrieve bird images from 200 classes.

**Dataset:** [`mteb/cub200_retrieval`](https://huggingface.co/datasets/mteb/cub200_retrieval) • **License:** not specified • [Learn more →](https://www.florian-schroff.de/publications/CUB-200.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{welinder2010caltech,
      author = {Welinder, Peter and Branson, Steve and Mita, Takeshi and Wah, Catherine and Schroff, Florian and Belongie, Serge and Perona, Pietro},
      month = {09},
      pages = {},
      title = {Caltech-UCSD Birds 200},
      year = {2010},
    }
    
    ```
    



#### `ClothoA2TRetrieval` { .model-copy }

An audio captioning datasetst containing audio clips and their corresponding captions.

**Dataset:** [`mteb/Clotho`](https://huggingface.co/datasets/mteb/Clotho) • **License:** mit • [Learn more →](https://github.com/audio-captioning/clotho-dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Encyclopaedic, Written | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{drossos2019clothoaudiocaptioningdataset,
      archiveprefix = {arXiv},
      author = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
      eprint = {1910.09387},
      primaryclass = {cs.SD},
      title = {Clotho: An Audio Captioning Dataset},
      url = {https://arxiv.org/abs/1910.09387},
      year = {2019},
    }
    
    ```
    



#### `ClothoT2ARetrieval` { .model-copy }

An audio captioning datasetst containing audio clips from the Freesound platform and their corresponding captions.

**Dataset:** [`mteb/Clotho`](https://huggingface.co/datasets/mteb/Clotho) • **License:** mit • [Learn more →](https://github.com/audio-captioning/clotho-dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Encyclopaedic, Written | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{drossos2019clothoaudiocaptioningdataset,
      archiveprefix = {arXiv},
      author = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
      eprint = {1910.09387},
      primaryclass = {cs.SD},
      title = {Clotho: An Audio Captioning Dataset},
      url = {https://arxiv.org/abs/1910.09387},
      year = {2019},
    }
    
    ```
    



#### `CommonVoiceMini17A2TRetrieval` { .model-copy }

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_17_0_mini`](https://huggingface.co/datasets/mteb/common_voice_17_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | ara, ast, bel, ben, bre, ... (50) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    @inproceedings{ardila2019common,
      author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      pages = {4218--4222},
      title = {Common voice: A massively-multilingual speech corpus},
      year = {2020},
    }
    
    ```
    



#### `CommonVoiceMini17T2ARetrieval` { .model-copy }

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_17_0_mini`](https://huggingface.co/datasets/mteb/common_voice_17_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | ara, ast, bel, ben, bre, ... (50) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    @inproceedings{ardila2019common,
      author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      pages = {4218--4222},
      title = {Common voice: A massively-multilingual speech corpus},
      year = {2020},
    }
    
    ```
    



#### `CommonVoiceMini21A2TRetrieval` { .model-copy }

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_21_0_mini`](https://huggingface.co/datasets/mteb/common_voice_21_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | abk, afr, amh, ara, asm, ... (114) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    @inproceedings{ardila2019common,
      author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      pages = {4218--4222},
      title = {Common voice: A massively-multilingual speech corpus},
      year = {2020},
    }
    
    ```
    



#### `CommonVoiceMini21T2ARetrieval` { .model-copy }

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_21_0_mini`](https://huggingface.co/datasets/mteb/common_voice_21_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | abk, afr, amh, ara, asm, ... (114) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    @inproceedings{ardila2019common,
      author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      pages = {4218--4222},
      title = {Common voice: A massively-multilingual speech corpus},
      year = {2020},
    }
    
    ```
    



#### `EDIST2ITRetrieval` { .model-copy }

Retrieve news images and titles based on news content.

**Dataset:** [`mteb/mbeir_edis_task2`](https://huggingface.co/datasets/mteb/mbeir_edis_task2) • **License:** apache-2.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.297/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | eng | News | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{liu2023edis,
      author = {Liu, Siqi and Feng, Weixi and Fu, Tsu-Jui and Chen, Wenhu and Wang, William},
      booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages = {4877--4894},
      title = {EDIS: Entity-Driven Image Search over Multimodal Web Content},
      year = {2023},
    }
    
    ```
    



#### `EmoVDBA2TRetrieval` { .model-copy }

Natural language emotional captions for speech segments from the EmoV-DB emotional voices database.

**Dataset:** [`mteb/EmoV_DB_a2t`](https://huggingface.co/datasets/mteb/EmoV_DB_a2t) • **License:** https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md • [Learn more →](https://github.com/numediart/EmoV-DB?tab=readme-ov-file)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{adigwe2018emotional,
      archiveprefix = {arXiv},
      author = {Adaeze Adigwe and Noé Tits and Kevin El Haddad and Sarah Ostadabbas and Thierry Dutoit},
      eprint = {1806.09514},
      primaryclass = {cs.CL},
      title = {The Emotional Voices Database: Towards Controlling the Emotion Dimension in Voice Generation Systems},
      url = {https://arxiv.org/abs/1806.09514},
      year = {2018},
    }
    
    ```
    



#### `EmoVDBT2ARetrieval` { .model-copy }

Natural language emotional captions for speech segments from the EmoV-DB emotional voices database.

**Dataset:** [`mteb/EmoV_DB_t2a`](https://huggingface.co/datasets/mteb/EmoV_DB_t2a) • **License:** https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md • [Learn more →](https://github.com/numediart/EmoV-DB?tab=readme-ov-file)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{adigwe2018emotional,
      archiveprefix = {arXiv},
      author = {Adaeze Adigwe and Noé Tits and Kevin El Haddad and Sarah Ostadabbas and Thierry Dutoit},
      eprint = {1806.09514},
      primaryclass = {cs.CL},
      title = {The Emotional Voices Database: Towards Controlling the Emotion Dimension in Voice Generation Systems},
      url = {https://arxiv.org/abs/1806.09514},
      year = {2018},
    }
    
    ```
    



#### `EncyclopediaVQAIT2ITRetrieval` { .model-copy }

Retrieval Wiki passage and image and passage to answer query about an image.

**Dataset:** [`izhx/UMRB-EncyclopediaVQA`](https://huggingface.co/datasets/izhx/UMRB-EncyclopediaVQA) • **License:** cc-by-4.0 • [Learn more →](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image, text (it2it) | eng | Encyclopaedic | derived | created | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{mensink2023encyclopedic,
      author = {Mensink, Thomas and Uijlings, Jasper and Castrejon, Lluis and Goel, Arushi and Cadar, Felipe and Zhou, Howard and Sha, Fei and Araujo, Andr{\'e} and Ferrari, Vittorio},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages = {3113--3124},
      title = {Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories},
      year = {2023},
    }
    
    ```
    



#### `FORBI2IRetrieval` { .model-copy }

Retrieve flat object images from 8 classes.

**Dataset:** [`mteb/forb_retrieval`](https://huggingface.co/datasets/mteb/forb_retrieval) • **License:** not specified • [Learn more →](https://github.com/pxiangwu/FORB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wu2023forbflatobjectretrieval,
      archiveprefix = {arXiv},
      author = {Pengxiang Wu and Siman Wang and Kevin Dela Rosa and Derek Hao Hu},
      eprint = {2309.16249},
      primaryclass = {cs.CV},
      title = {FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding},
      url = {https://arxiv.org/abs/2309.16249},
      year = {2023},
    }
    
    ```
    



#### `Fashion200kI2TRetrieval` { .model-copy }

Retrieve clothes based on descriptions.

**Dataset:** [`mteb/mbeir_fashion200k_task3`](https://huggingface.co/datasets/mteb/mbeir_fashion200k_task3) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{han2017automatic,
      author = {Han, Xintong and Wu, Zuxuan and Huang, Phoenix X and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang and Davis, Larry S},
      booktitle = {Proceedings of the IEEE international conference on computer vision},
      pages = {1463--1471},
      title = {Automatic spatially-aware fashion concept discovery},
      year = {2017},
    }
    
    ```
    



#### `Fashion200kT2IRetrieval` { .model-copy }

Retrieve clothes based on descriptions.

**Dataset:** [`mteb/mbeir_fashion200k_task0`](https://huggingface.co/datasets/mteb/mbeir_fashion200k_task0) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{han2017automatic,
      author = {Han, Xintong and Wu, Zuxuan and Huang, Phoenix X and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang and Davis, Larry S},
      booktitle = {Proceedings of the IEEE international conference on computer vision},
      pages = {1463--1471},
      title = {Automatic spatially-aware fashion concept discovery},
      year = {2017},
    }
    
    ```
    



#### `FashionIQIT2IRetrieval` { .model-copy }

Retrieve clothes based on descriptions.

**Dataset:** [`mteb/mbeir_fashioniq_task7`](https://huggingface.co/datasets/mteb/mbeir_fashioniq_task7) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image (it2i) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wu2021fashion,
      author = {Wu, Hui and Gao, Yupeng and Guo, Xiaoxiao and Al-Halah, Ziad and Rennie, Steven and Grauman, Kristen and Feris, Rogerio},
      booktitle = {Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
      pages = {11307--11317},
      title = {Fashion iq: A new dataset towards retrieving images by natural language feedback},
      year = {2021},
    }
    
    ```
    



#### `FleursA2TRetrieval` { .model-copy }

Speech recordings with corresponding text transcriptions from the FLEURS dataset.

**Dataset:** [`mteb/fleurs`](https://huggingface.co/datasets/mteb/fleurs) • **License:** apache-2.0 • [Learn more →](https://github.com/google-research-datasets/fleurs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | afr, amh, ara, asm, ast, ... (102) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{conneau2023fleurs,
      author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
      booktitle = {2022 IEEE Spoken Language Technology Workshop (SLT)},
      organization = {IEEE},
      pages = {798--805},
      title = {Fleurs: Few-shot learning evaluation of universal representations of speech},
      year = {2023},
    }
    
    ```
    



#### `FleursT2ARetrieval` { .model-copy }

Speech recordings with corresponding text transcriptions from the FLEURS dataset.

**Dataset:** [`mteb/fleurs`](https://huggingface.co/datasets/mteb/fleurs) • **License:** apache-2.0 • [Learn more →](https://github.com/google-research-datasets/fleurs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | afr, amh, ara, asm, ast, ... (102) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{conneau2023fleurs,
      author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
      booktitle = {2022 IEEE Spoken Language Technology Workshop (SLT)},
      organization = {IEEE},
      pages = {798--805},
      title = {Fleurs: Few-shot learning evaluation of universal representations of speech},
      year = {2023},
    }
    
    ```
    



#### `Flickr30kI2TRetrieval` { .model-copy }

Retrieve captions based on images.

**Dataset:** [`mteb/flickr30ki2t`](https://huggingface.co/datasets/mteb/flickr30ki2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{Young2014FromID,
      author = {Peter Young and Alice Lai and Micah Hodosh and J. Hockenmaier},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {67-78},
      title = {From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
      url = {https://api.semanticscholar.org/CorpusID:3104920},
      volume = {2},
      year = {2014},
    }
    
    ```
    



#### `Flickr30kT2IRetrieval` { .model-copy }

Retrieve images based on captions.

**Dataset:** [`mteb/flickr30kt2i`](https://huggingface.co/datasets/mteb/flickr30kt2i) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{Young2014FromID,
      author = {Peter Young and Alice Lai and Micah Hodosh and J. Hockenmaier},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {67-78},
      title = {From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
      url = {https://api.semanticscholar.org/CorpusID:3104920},
      volume = {2},
      year = {2014},
    }
    
    ```
    



#### `GLDv2I2IRetrieval` { .model-copy }

Retrieve names of landmarks based on their image.

**Dataset:** [`mteb/gld-v2`](https://huggingface.co/datasets/mteb/gld-v2) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Weyand_2020_CVPR,
      author = {Weyand, Tobias and Araujo, Andre and Cao, Bingyi and Sim, Jack},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      title = {Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval},
      year = {2020},
    }
    
    ```
    



#### `GLDv2I2TRetrieval` { .model-copy }

Retrieve names of landmarks based on their image.

**Dataset:** [`mteb/gld-v2-i2t`](https://huggingface.co/datasets/mteb/gld-v2-i2t) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Weyand_2020_CVPR,
      author = {Weyand, Tobias and Araujo, Andre and Cao, Bingyi and Sim, Jack},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      title = {Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval},
      year = {2020},
    }
    
    ```
    



#### `GigaSpeechA2TRetrieval` { .model-copy }

Given an English speech segment, retrieve its correct transcription. Audio comes from the 10 000‑hour training subset of GigaSpeech, which originates from ≈40 000 hours of transcribed audiobooks, podcasts, and YouTube.

**Dataset:** [`mteb/gigaspeech_a2t`](https://huggingface.co/datasets/mteb/gigaspeech_a2t) • **License:** https://github.com/SpeechColab/GigaSpeech/blob/main/LICENSE • [Learn more →](https://github.com/SpeechColab/GigaSpeech)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{GigaSpeech2021,
      author = {Chen, Guoguo and Chai, Shuzhou and Wang, Guanbo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and Wang, Yujun and You, Zhao and Yan, Zhiyong},
      booktitle = {Proc. Interspeech 2021},
      title = {GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio},
      year = {2021},
    }
    
    ```
    



#### `GigaSpeechT2ARetrieval` { .model-copy }

Given an English transcription, retrieve its corresponding audio segment. Audio comes from the 10 000‑hour training subset of GigaSpeech, sourced from ≈40 000 hours of transcribed audiobooks, podcasts, and YouTube.

**Dataset:** [`mteb/gigaspeech_t2a`](https://huggingface.co/datasets/mteb/gigaspeech_t2a) • **License:** https://github.com/SpeechColab/GigaSpeech/blob/main/LICENSE • [Learn more →](https://github.com/SpeechColab/GigaSpeech)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{GigaSpeech2021,
      author = {Chen, Guoguo and Chai, Shuzhou and Wang, Guanbo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and Wang, Yujun and You, Zhao and Yan, Zhiyong},
      booktitle = {Proc. Interspeech 2021},
      title = {GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio},
      year = {2021},
    }
    
    ```
    



#### `GoogleSVQA2TRetrieval` { .model-copy }

Multilingual audio-to-text retrieval using the Simple Voice Questions (SVQ) dataset. Given an audio query, retrieve the corresponding text transcription.

**Dataset:** [`mteb/svq`](https://huggingface.co/datasets/mteb/svq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/google/svq)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | acm, apc, arq, arz, ben, ... (20) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{heigold2025massive,
      author = {Georg Heigold and Ehsan Variani and Tom Bagby and Cyril Allauzen and Ji Ma and Shankar Kumar and Michael Riley},
      booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
      title = {Massive Sound Embedding Benchmark ({MSEB})},
      url = {https://openreview.net/forum?id=X0juYgFVng},
      year = {2025},
    }
    
    ```
    



#### `GoogleSVQT2ARetrieval` { .model-copy }

Multilingual text-to-audio retrieval using the Simple Voice Questions (SVQ) dataset. Given a text query, retrieve the corresponding audio recording.

**Dataset:** [`mteb/svq`](https://huggingface.co/datasets/mteb/svq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/google/svq)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | acm, apc, arq, arz, ben, ... (20) | Spoken | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{heigold2025massive,
      author = {Georg Heigold and Ehsan Variani and Tom Bagby and Cyril Allauzen and Ji Ma and Shankar Kumar and Michael Riley},
      booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
      title = {Massive Sound Embedding Benchmark ({MSEB})},
      url = {https://openreview.net/forum?id=X0juYgFVng},
      year = {2025},
    }
    
    ```
    



#### `HatefulMemesI2TRetrieval` { .model-copy }

Retrieve captions based on memes to assess OCR abilities.

**Dataset:** [`mteb/MMSoc_HatefulMemes`](https://huggingface.co/datasets/mteb/MMSoc_HatefulMemes) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2005.04790)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{kiela2020hateful,
      author = {Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
      journal = {Advances in neural information processing systems},
      pages = {2611--2624},
      title = {The hateful memes challenge: Detecting hate speech in multimodal memes},
      volume = {33},
      year = {2020},
    }
    
    ```
    



#### `HatefulMemesT2IRetrieval` { .model-copy }

Retrieve captions based on memes to assess OCR abilities.

**Dataset:** [`mteb/MMSoc_HatefulMemes`](https://huggingface.co/datasets/mteb/MMSoc_HatefulMemes) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2005.04790)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{kiela2020hateful,
      author = {Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
      journal = {Advances in neural information processing systems},
      pages = {2611--2624},
      title = {The hateful memes challenge: Detecting hate speech in multimodal memes},
      volume = {33},
      year = {2020},
    }
    
    ```
    



#### `HiFiTTSA2TRetrieval` { .model-copy }

Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.

**Dataset:** [`mteb/hifi-tts_a2t`](https://huggingface.co/datasets/mteb/hifi-tts_a2t) • **License:** cc-by-4.0 • [Learn more →](https://openslr.org/109/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{bakhturina2021hi,
      author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
      journal = {arXiv preprint arXiv:2104.01497},
      title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
      year = {2021},
    }
    
    ```
    



#### `HiFiTTST2ARetrieval` { .model-copy }

Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.

**Dataset:** [`mteb/hifi-tts_t2a`](https://huggingface.co/datasets/mteb/hifi-tts_t2a) • **License:** cc-by-4.0 • [Learn more →](https://openslr.org/109/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{bakhturina2021hi,
      author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
      journal = {arXiv preprint arXiv:2104.01497},
      title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
      year = {2021},
    }
    
    ```
    



#### `ImageCoDeT2IRetrieval` { .model-copy }

Retrieve a specific video frame based on a precise caption.

**Dataset:** [`mteb/imagecode`](https://huggingface.co/datasets/mteb/imagecode) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.acl-long.241.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web, Written | derived | found | hit_rate_at_3 |



??? quote "Citation"

    
    ```bibtex
    
    @article{krojer2022image,
      author = {Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
      journal = {arXiv preprint arXiv:2203.15867},
      title = {Image retrieval from contextual descriptions},
      year = {2022},
    }
    
    ```
    



#### `InfoSeekIT2ITRetrieval` { .model-copy }

Retrieve source text and image information to answer questions about images.

**Dataset:** [`mteb/InfoSeekIT2ITRetrieval`](https://huggingface.co/datasets/mteb/InfoSeekIT2ITRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.925)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image, text (it2it) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{chen2023can,
      author = {Chen, Yang and Hu, Hexiang and Luan, Yi and Sun, Haitian and Changpinyo, Soravit and Ritter, Alan and Chang, Ming-Wei},
      booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages = {14948--14968},
      title = {Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?},
      year = {2023},
    }
    
    ```
    



#### `InfoSeekIT2TRetrieval` { .model-copy }

Retrieve source information to answer questions about images.

**Dataset:** [`mteb/mbeir_infoseek_task6`](https://huggingface.co/datasets/mteb/mbeir_infoseek_task6) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.925)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{chen2023can,
      author = {Chen, Yang and Hu, Hexiang and Luan, Yi and Sun, Haitian and Changpinyo, Soravit and Ritter, Alan and Chang, Ming-Wei},
      booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages = {14948--14968},
      title = {Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?},
      year = {2023},
    }
    
    ```
    



#### `JLCorpusA2TRetrieval` { .model-copy }

Emotional speech segments from the JL-Corpus, balanced over long vowels and annotated for primary and secondary emotions.

**Dataset:** [`mteb/jl_corpus_a2t`](https://huggingface.co/datasets/mteb/jl_corpus_a2t) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/tli725/jl-corpus)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{james2018open,
      author = {James, Jesin and Li, Tian and Watson, Catherine},
      booktitle = {Proc. Interspeech 2018},
      title = {An Open Source Emotional Speech Corpus for Human Robot Interaction Applications},
      year = {2018},
    }
    
    ```
    



#### `JLCorpusT2ARetrieval` { .model-copy }

Emotional speech segments from the JL-Corpus, balanced over long vowels and annotated for primary and secondary emotions.

**Dataset:** [`mteb/jl_corpus_t2a`](https://huggingface.co/datasets/mteb/jl_corpus_t2a) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/tli725/jl-corpus)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{james2018open,
      author = {James, Jesin and Li, Tian and Watson, Catherine},
      booktitle = {Proc. Interspeech 2018},
      title = {An Open Source Emotional Speech Corpus for Human Robot Interaction Applications},
      year = {2018},
    }
    
    ```
    



#### `JamAltArtistA2ARetrieval` { .model-copy }

Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset

**Dataset:** [`mteb/jam-alt-lines`](https://huggingface.co/datasets/mteb/jam-alt-lines) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to audio (a2a) | deu, eng, fra, spa | Music | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{cifka-2024-jam-alt,
      author = {Ond{\v{r}}ej C{\'{\i}}fka and
    Hendrik Schreiber and
    Luke Miner and
    Fabian{-}Robert St{\"{o}}ter},
      booktitle = {Proceedings of the 25th International Society for
    Music Information Retrieval Conference},
      doi = {10.5281/ZENODO.14877443},
      pages = {737--744},
      publisher = {ISMIR},
      title = {Lyrics Transcription for Humans: {A} Readability-Aware Benchmark},
      url = {https://doi.org/10.5281/zenodo.14877443},
      year = {2024},
    }
    
    ```
    



#### `JamAltLyricA2TRetrieval` { .model-copy }

From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset

**Dataset:** [`mteb/jam-alt-lines`](https://huggingface.co/datasets/mteb/jam-alt-lines) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | deu, eng, fra, spa | Music | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{cifka-2024-jam-alt,
      author = {Ond{\v{r}}ej C{\'{\i}}fka and
    Hendrik Schreiber and
    Luke Miner and
    Fabian{-}Robert St{\"{o}}ter},
      booktitle = {Proceedings of the 25th International Society for
    Music Information Retrieval Conference},
      doi = {10.5281/ZENODO.14877443},
      pages = {737--744},
      publisher = {ISMIR},
      title = {Lyrics Transcription for Humans: {A} Readability-Aware Benchmark},
      url = {https://doi.org/10.5281/zenodo.14877443},
      year = {2024},
    }
    
    ```
    



#### `JamAltLyricT2ARetrieval` { .model-copy }

From textual lyrics (query), retrieve corresponding audio clips of songs from the Jam-Alt-Lines dataset

**Dataset:** [`mteb/jam-alt-lines`](https://huggingface.co/datasets/mteb/jam-alt-lines) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | deu, eng, fra, spa | Music | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{cifka-2024-jam-alt,
      author = {Ond{\v{r}}ej C{\'{\i}}fka and
    Hendrik Schreiber and
    Luke Miner and
    Fabian{-}Robert St{\"{o}}ter},
      booktitle = {Proceedings of the 25th International Society for
    Music Information Retrieval Conference},
      doi = {10.5281/ZENODO.14877443},
      pages = {737--744},
      publisher = {ISMIR},
      title = {Lyrics Transcription for Humans: {A} Readability-Aware Benchmark},
      url = {https://doi.org/10.5281/zenodo.14877443},
      year = {2024},
    }
    
    ```
    



#### `LASSA2TRetrieval` { .model-copy }

Language-Queried Audio Source Separation (LASS) dataset for audio-to-text retrieval. Retrieve text descriptions/captions for audio clips using natural language queries.The original dataset is based on the AudioCaps dataset.The source audio has been synthesized by mixing two audio with their labelled snr ratio as indicated in the dataset.

**Dataset:** [`mteb/lass-synth-a2t`](https://huggingface.co/datasets/mteb/lass-synth-a2t) • **License:** mit • [Learn more →](https://dcase.community/challenge2024/task-language-queried-audio-source-separation)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | AudioScene | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{liu2022separate,
      author = {Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
      booktitle = {INTERSPEEH},
      title = {Separate What You Describe: Language-Queried Audio Source Separation},
      year = {2022},
    }
    
    ```
    



#### `LASST2ARetrieval` { .model-copy }

Language-Queried Audio Source Separation (LASS) dataset for text-to-audio retrieval. Retrieve audio clips corresponding to natural language text descriptions/captions.The original dataset is based on the AudioCaps dataset.The source audio has been synthesized by mixing two audio with their labelled snr ratio as indicated in the dataset.

**Dataset:** [`mteb/lass-synth-t2a`](https://huggingface.co/datasets/mteb/lass-synth-t2a) • **License:** mit • [Learn more →](https://dcase.community/challenge2024/task-language-queried-audio-source-separation)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | AudioScene | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{liu2022separate,
      author = {Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
      booktitle = {INTERSPEEH},
      title = {Separate What You Describe: Language-Queried Audio Source Separation},
      year = {2022},
    }
    
    ```
    



#### `LLaVAIT2TRetrieval` { .model-copy }

Retrieve responses to answer questions about images.

**Dataset:** [`izhx/UMRB-LLaVA`](https://huggingface.co/datasets/izhx/UMRB-LLaVA) • **License:** cc-by-4.0 • [Learn more →](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{lin-etal-2024-preflmr,
      address = {Bangkok, Thailand},
      author = {Lin, Weizhe  and
    Mei, Jingbiao  and
    Chen, Jinghong  and
    Byrne, Bill},
      booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/2024.acl-long.289},
      editor = {Ku, Lun-Wei  and
    Martins, Andre  and
    Srikumar, Vivek},
      month = aug,
      pages = {5294--5316},
      publisher = {Association for Computational Linguistics},
      title = {{P}re{FLMR}: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers},
      url = {https://aclanthology.org/2024.acl-long.289},
      year = {2024},
    }
    
    ```
    



#### `LibriTTSA2TRetrieval` { .model-copy }

Given audiobook speech segments from the multi‑speaker LibriTTS corpus, retrieve the correct text transcription. LibriTTS is a 585‑hour, 24 kHz, multi‑speaker English TTS corpus derived from LibriVox (audio) and Project Gutenberg (text).

**Dataset:** [`mteb/LibriTTS_a2t`](https://huggingface.co/datasets/mteb/LibriTTS_a2t) • **License:** cc-by-4.0 • [Learn more →](https://www.openslr.org/60/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{zen2019librittscorpusderivedlibrispeech,
      archiveprefix = {arXiv},
      author = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
      eprint = {1904.02882},
      primaryclass = {cs.SD},
      title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
      url = {https://arxiv.org/abs/1904.02882},
      year = {2019},
    }
    
    ```
    



#### `LibriTTST2ARetrieval` { .model-copy }

Given an English text transcription, retrieve its corresponding audiobook speech segment from the multi‑speaker LibriTTS corpus. LibriTTS is a 585‑hour, 24 kHz, multi‑speaker English TTS corpus derived from LibriVox and Project Gutenberg.

**Dataset:** [`mteb/LibriTTS_t2a`](https://huggingface.co/datasets/mteb/LibriTTS_t2a) • **License:** cc-by-4.0 • [Learn more →](https://www.openslr.org/60/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Spoken | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{zen2019librittscorpusderivedlibrispeech,
      archiveprefix = {arXiv},
      author = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
      eprint = {1904.02882},
      primaryclass = {cs.SD},
      title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
      url = {https://arxiv.org/abs/1904.02882},
      year = {2019},
    }
    
    ```
    



#### `MACSA2TRetrieval` { .model-copy }

Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset.

**Dataset:** [`mteb/MACS_a2t`](https://huggingface.co/datasets/mteb/MACS_a2t) • **License:** https://zenodo.org/records/5114771 • [Learn more →](https://zenodo.org/records/5114771)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | eng | AudioScene | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{martinmorato2021groundtruthreliabilitymultiannotator,
      archiveprefix = {arXiv},
      author = {Irene Martin-Morato and Annamaria Mesaros},
      eprint = {2104.04214},
      primaryclass = {eess.AS},
      title = {What is the ground truth? Reliability of multi-annotator data for audio tagging},
      url = {https://arxiv.org/abs/2104.04214},
      year = {2021},
    }
    
    ```
    



#### `MACST2ARetrieval` { .model-copy }

Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset.

**Dataset:** [`mteb/MACS_t2a`](https://huggingface.co/datasets/mteb/MACS_t2a) • **License:** https://zenodo.org/records/5114771 • [Learn more →](https://zenodo.org/records/5114771)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | AudioScene | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{martinmorato2021groundtruthreliabilitymultiannotator,
      archiveprefix = {arXiv},
      author = {Irene Martin-Morato and Annamaria Mesaros},
      eprint = {2104.04214},
      primaryclass = {eess.AS},
      title = {What is the ground truth? Reliability of multi-annotator data for audio tagging},
      url = {https://arxiv.org/abs/2104.04214},
      year = {2021},
    }
    
    ```
    



#### `METI2IRetrieval` { .model-copy }

Retrieve photos of more than 224k artworks.

**Dataset:** [`mteb/met`](https://huggingface.co/datasets/mteb/met) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2202.01747)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{ypsilantis2021met,
      author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {The met dataset: Instance-level recognition for artworks},
      year = {2021},
    }
    
    ```
    



#### `MSCOCOI2TRetrieval` { .model-copy }

Retrieve captions based on images.

**Dataset:** [`mteb/mbeir_mscoco_task3`](https://huggingface.co/datasets/mteb/mbeir_mscoco_task3) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{lin2014microsoft,
      author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
      booktitle = {Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
      organization = {Springer},
      pages = {740--755},
      title = {Microsoft coco: Common objects in context},
      year = {2014},
    }
    
    ```
    



#### `MSCOCOT2IRetrieval` { .model-copy }

Retrieve images based on captions.

**Dataset:** [`mteb/mbeir_mscoco_task0`](https://huggingface.co/datasets/mteb/mbeir_mscoco_task0) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{lin2014microsoft,
      author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
      booktitle = {Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
      organization = {Springer},
      pages = {740--755},
      title = {Microsoft coco: Common objects in context},
      year = {2014},
    }
    
    ```
    



#### `MemotionI2TRetrieval` { .model-copy }

Retrieve captions based on memes.

**Dataset:** [`mteb/MMSoc_Memotion`](https://huggingface.co/datasets/mteb/MMSoc_Memotion) • **License:** mit • [Learn more →](https://aclanthology.org/2020.semeval-1.99/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sharma2020semeval,
      author = {Sharma, Chhavi and Bhageria, Deepesh and Scott, William and Pykl, Srinivas and Das, Amitava and Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
      booktitle = {Proceedings of the Fourteenth Workshop on Semantic Evaluation},
      pages = {759--773},
      title = {SemEval-2020 Task 8: Memotion Analysis-the Visuo-Lingual Metaphor!},
      year = {2020},
    }
    
    ```
    



#### `MemotionT2IRetrieval` { .model-copy }

Retrieve memes based on captions.

**Dataset:** [`mteb/MMSoc_Memotion`](https://huggingface.co/datasets/mteb/MMSoc_Memotion) • **License:** mit • [Learn more →](https://aclanthology.org/2020.semeval-1.99/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sharma2020semeval,
      author = {Sharma, Chhavi and Bhageria, Deepesh and Scott, William and Pykl, Srinivas and Das, Amitava and Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
      booktitle = {Proceedings of the Fourteenth Workshop on Semantic Evaluation},
      pages = {759--773},
      title = {SemEval-2020 Task 8: Memotion Analysis-the Visuo-Lingual Metaphor!},
      year = {2020},
    }
    
    ```
    



#### `MusicCapsA2TRetrieval` { .model-copy }

Natural language description for music audio.

**Dataset:** [`mteb/MusicCaps_a2t`](https://huggingface.co/datasets/mteb/MusicCaps_a2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/nateraw/download-musiccaps-dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | zxx | Music | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{agostinelli2023musiclmgeneratingmusictext,
      archiveprefix = {arXiv},
      author = {Andrea Agostinelli and Timo I. Denk and Zalán Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matt Sharifi and Neil Zeghidour and Christian Frank},
      eprint = {2301.11325},
      primaryclass = {cs.SD},
      title = {MusicLM: Generating Music From Text},
      url = {https://arxiv.org/abs/2301.11325},
      year = {2023},
    }
    
    ```
    



#### `MusicCapsT2ARetrieval` { .model-copy }

Natural language description for music audio.

**Dataset:** [`mteb/MusicCaps_t2a`](https://huggingface.co/datasets/mteb/MusicCaps_t2a) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/nateraw/download-musiccaps-dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | zxx | Music | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{agostinelli2023musiclmgeneratingmusictext,
      archiveprefix = {arXiv},
      author = {Andrea Agostinelli and Timo I. Denk and Zalán Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matt Sharifi and Neil Zeghidour and Christian Frank},
      eprint = {2301.11325},
      primaryclass = {cs.SD},
      title = {MusicLM: Generating Music From Text},
      url = {https://arxiv.org/abs/2301.11325},
      year = {2023},
    }
    
    ```
    



#### `NIGHTSI2IRetrieval` { .model-copy }

Retrieval identical image to the given image.

**Dataset:** [`mteb/mbeir_nights_task4`](https://huggingface.co/datasets/mteb/mbeir_nights_task4) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{fu2024dreamsim,
      author = {Fu, Stephanie and Tamir, Netanel and Sundaram, Shobhita and Chai, Lucy and Zhang, Richard and Dekel, Tali and Isola, Phillip},
      journal = {Advances in Neural Information Processing Systems},
      title = {DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data},
      volume = {36},
      year = {2024},
    }
    
    ```
    



#### `OKVQAIT2TRetrieval` { .model-copy }

Retrieval a Wiki passage to answer query about an image.

**Dataset:** [`izhx/UMRB-OKVQA`](https://huggingface.co/datasets/izhx/UMRB-OKVQA) • **License:** cc-by-4.0 • [Learn more →](https://okvqa.allenai.org/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | created | hit_rate_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{marino2019ok,
      author = {Marino, Kenneth and Rastegari, Mohammad and Farhadi, Ali and Mottaghi, Roozbeh},
      booktitle = {Proceedings of the IEEE/cvf conference on computer vision and pattern recognition},
      pages = {3195--3204},
      title = {Ok-vqa: A visual question answering benchmark requiring external knowledge},
      year = {2019},
    }
    
    ```
    



#### `OVENIT2ITRetrieval` { .model-copy }

Retrieval a Wiki image and passage to answer query about an image.

**Dataset:** [`mteb/mbeir_oven_task8`](https://huggingface.co/datasets/mteb/mbeir_oven_task8) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image, text (it2it) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hu2023open,
      author = {Hu, Hexiang and Luan, Yi and Chen, Yang and Khandelwal, Urvashi and Joshi, Mandar and Lee, Kenton and Toutanova, Kristina and Chang, Ming-Wei},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages = {12065--12075},
      title = {Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
      year = {2023},
    }
    
    ```
    



#### `OVENIT2TRetrieval` { .model-copy }

Retrieval a Wiki passage to answer query about an image.

**Dataset:** [`mteb/mbeir_oven_task6`](https://huggingface.co/datasets/mteb/mbeir_oven_task6) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hu2023open,
      author = {Hu, Hexiang and Luan, Yi and Chen, Yang and Khandelwal, Urvashi and Joshi, Mandar and Lee, Kenton and Toutanova, Kristina and Chang, Ming-Wei},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages = {12065--12075},
      title = {Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
      year = {2023},
    }
    
    ```
    



#### `ROxfordEasyI2IRetrieval` { .model-copy }

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`mteb/r-oxford-easy-multi`](https://huggingface.co/datasets/mteb/r-oxford-easy-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | map_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{radenovic2018revisiting,
      author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {5706--5715},
      title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
      year = {2018},
    }
    
    ```
    



#### `ROxfordHardI2IRetrieval` { .model-copy }

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`mteb/r-oxford-hard-multi`](https://huggingface.co/datasets/mteb/r-oxford-hard-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | map_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{radenovic2018revisiting,
      author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {5706--5715},
      title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
      year = {2018},
    }
    
    ```
    



#### `ROxfordMediumI2IRetrieval` { .model-copy }

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`mteb/r-oxford-medium-multi`](https://huggingface.co/datasets/mteb/r-oxford-medium-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | map_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{radenovic2018revisiting,
      author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {5706--5715},
      title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
      year = {2018},
    }
    
    ```
    



#### `RP2kI2IRetrieval` { .model-copy }

Retrieve photos of 39457 products.

**Dataset:** [`mteb/rp2k`](https://huggingface.co/datasets/mteb/rp2k) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2006.12634)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{peng2020rp2k,
      author = {Peng, Jingtian and Xiao, Chang and Li, Yifan},
      journal = {arXiv preprint arXiv:2006.12634},
      title = {RP2K: A large-scale retail product dataset for fine-grained image classification},
      year = {2020},
    }
    
    ```
    



#### `RParisEasyI2IRetrieval` { .model-copy }

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`mteb/r-paris-easy-multi`](https://huggingface.co/datasets/mteb/r-paris-easy-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | map_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{radenovic2018revisiting,
      author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {5706--5715},
      title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
      year = {2018},
    }
    
    ```
    



#### `RParisHardI2IRetrieval` { .model-copy }

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`mteb/r-paris-hard-multi`](https://huggingface.co/datasets/mteb/r-paris-hard-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | map_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{radenovic2018revisiting,
      author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {5706--5715},
      title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
      year = {2018},
    }
    
    ```
    



#### `RParisMediumI2IRetrieval` { .model-copy }

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`mteb/r-paris-medium-multi`](https://huggingface.co/datasets/mteb/r-paris-medium-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Web | derived | created | map_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{radenovic2018revisiting,
      author = {Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {5706--5715},
      title = {Revisiting oxford and paris: Large-scale image retrieval benchmarking},
      year = {2018},
    }
    
    ```
    



#### `ReMuQIT2TRetrieval` { .model-copy }

Retrieval of a Wiki passage to answer a query about an image.

**Dataset:** [`izhx/UMRB-ReMuQ`](https://huggingface.co/datasets/izhx/UMRB-ReMuQ) • **License:** cc0-1.0 • [Learn more →](https://github.com/luomancs/ReMuQ)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | created | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{luo-etal-2023-end,
      address = {Toronto, Canada},
      author = {Luo, Man  and
    Fang, Zhiyuan  and
    Gokhale, Tejas  and
    Yang, Yezhou  and
    Baral, Chitta},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/2023.acl-long.478},
      editor = {Rogers, Anna  and
    Boyd-Graber, Jordan  and
    Okazaki, Naoaki},
      month = jul,
      pages = {8573--8589},
      publisher = {Association for Computational Linguistics},
      title = {End-to-end Knowledge Retrieval with Multi-modal Queries},
      url = {https://aclanthology.org/2023.acl-long.478},
      year = {2023},
    }
    
    ```
    



#### `SOPI2IRetrieval` { .model-copy }

Retrieve product photos of 22634 online products.

**Dataset:** [`mteb/stanford-online-products`](https://huggingface.co/datasets/mteb/stanford-online-products) • **License:** not specified • [Learn more →](https://paperswithcode.com/dataset/stanford-online-products)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{oh2016deep,
      author = {Oh Song, Hyun and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {4004--4012},
      title = {Deep metric learning via lifted structured feature embedding},
      year = {2016},
    }
    
    ```
    



#### `SciMMIRI2TRetrieval` { .model-copy }

Retrieve captions based on figures and tables.

**Dataset:** [`mteb/SciMMIR`](https://huggingface.co/datasets/mteb/SciMMIR) • **License:** mit • [Learn more →](https://aclanthology.org/2024.findings-acl.746/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Academic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wu2024scimmirbenchmarkingscientificmultimodal,
      archiveprefix = {arXiv},
      author = {Siwei Wu and Yizhi Li and Kang Zhu and Ge Zhang and Yiming Liang and Kaijing Ma and Chenghao Xiao and Haoran Zhang and Bohao Yang and Wenhu Chen and Wenhao Huang and Noura Al Moubayed and Jie Fu and Chenghua Lin},
      eprint = {2401.13478},
      primaryclass = {cs.IR},
      title = {SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval},
      url = {https://arxiv.org/abs/2401.13478},
      year = {2024},
    }
    
    ```
    



#### `SciMMIRT2IRetrieval` { .model-copy }

Retrieve figures and tables based on captions.

**Dataset:** [`mteb/SciMMIR`](https://huggingface.co/datasets/mteb/SciMMIR) • **License:** mit • [Learn more →](https://aclanthology.org/2024.findings-acl.746/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wu2024scimmirbenchmarkingscientificmultimodal,
      archiveprefix = {arXiv},
      author = {Siwei Wu and Yizhi Li and Kang Zhu and Ge Zhang and Yiming Liang and Kaijing Ma and Chenghao Xiao and Haoran Zhang and Bohao Yang and Wenhu Chen and Wenhao Huang and Noura Al Moubayed and Jie Fu and Chenghua Lin},
      eprint = {2401.13478},
      primaryclass = {cs.IR},
      title = {SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval},
      url = {https://arxiv.org/abs/2401.13478},
      year = {2024},
    }
    
    ```
    



#### `SketchyI2IRetrieval` { .model-copy }

Retrieve photos from sketches.

**Dataset:** [`mteb/sketchy`](https://huggingface.co/datasets/mteb/sketchy) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2202.01747)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{ypsilantis2021met,
      author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {The met dataset: Instance-level recognition for artworks},
      year = {2021},
    }
    
    ```
    



#### `SoundDescsA2TRetrieval` { .model-copy }

Natural language description for different audio sources from the BBC Sound Effects webpage.

**Dataset:** [`mteb/sounddescs_a2t`](https://huggingface.co/datasets/mteb/sounddescs_a2t) • **License:** apache-2.0 • [Learn more →](https://github.com/akoepke/audio-retrieval-benchmark)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | zxx | Encyclopaedic, Written | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Koepke2022,
      author = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
      booktitle = {IEEE Transactions on Multimedia},
      title = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
      year = {2022},
    }
    
    ```
    



#### `SoundDescsT2ARetrieval` { .model-copy }

Natural language description for different audio sources from the BBC Sound Effects webpage.

**Dataset:** [`mteb/sounddescs_t2a`](https://huggingface.co/datasets/mteb/sounddescs_t2a) • **License:** apache-2.0 • [Learn more →](https://github.com/akoepke/audio-retrieval-benchmark)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | zxx | Encyclopaedic, Written | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Koepke2022,
      author = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
      booktitle = {IEEE Transactions on Multimedia},
      title = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
      year = {2022},
    }
    
    ```
    



#### `SpokenSQuADT2ARetrieval` { .model-copy }

Text-to-audio retrieval task based on SpokenSQuAD dataset. Given a text question, retrieve relevant audio segments that contain the answer. Questions are derived from SQuAD reading comprehension dataset with corresponding spoken passages.

**Dataset:** [`mteb/spoken-squad-t2a`](https://huggingface.co/datasets/mteb/spoken-squad-t2a) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/chiuwy/Spoken-SQuAD)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | eng | Academic, Encyclopaedic, Non-fiction | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{li2018spokensquad,
      author = {Li, Chia-Hsuan and Ma, Szu-Lin and Zhang, Hsin-Wei and Lee, Hung-yi and Lee, Lin-shan},
      booktitle = {Interspeech},
      pages = {3459--3463},
      title = {Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension},
      year = {2018},
    }
    
    ```
    



#### `StanfordCarsI2IRetrieval` { .model-copy }

Retrieve car images from 196 makes.

**Dataset:** [`mteb/stanford_cars_retrieval`](https://huggingface.co/datasets/mteb/stanford_cars_retrieval) • **License:** not specified • [Learn more →](https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic | derived | created | hit_rate_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Krause2013CollectingAL,
      author = {Jonathan Krause and Jia Deng and Michael Stark and Li Fei-Fei},
      title = {Collecting a Large-scale Dataset of Fine-grained Cars},
      url = {https://api.semanticscholar.org/CorpusID:16632981},
      year = {2013},
    }
    
    ```
    



#### `TUBerlinT2IRetrieval` { .model-copy }

Retrieve sketch images based on text descriptions.

**Dataset:** [`mteb/tu-berlin`](https://huggingface.co/datasets/mteb/tu-berlin) • **License:** cc-by-sa-4.0 • [Learn more →](https://dl.acm.org/doi/pdf/10.1145/2185520.2185540)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{eitz2012humans,
      author = {Eitz, Mathias and Hays, James and Alexa, Marc},
      journal = {ACM Transactions on graphics (TOG)},
      number = {4},
      pages = {1--10},
      publisher = {Acm New York, NY, USA},
      title = {How do humans sketch objects?},
      volume = {31},
      year = {2012},
    }
    
    ```
    



#### `UrbanSound8KA2TRetrieval` { .model-copy }

UrbanSound8K: Audio-to-text retrieval of urban sound events.

**Dataset:** [`mteb/Urbansound8K_a2t`](https://huggingface.co/datasets/mteb/Urbansound8K_a2t) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/CLAPv2/Urbansound8K)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to text (a2t) | zxx | AudioScene | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Salamon:UrbanSound:ACMMM:14,
      author = {Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
      booktitle = {Proceedings of the 22nd ACM international conference on Multimedia},
      organization = {ACM},
      pages = {1041--1044},
      title = {A Dataset and Taxonomy for Urban Sound Research},
      year = {2014},
    }
    
    ```
    



#### `UrbanSound8KT2ARetrieval` { .model-copy }

UrbanSound8K: Text-to-audio retrieval of urban sound events.

**Dataset:** [`mteb/Urbansound8K_t2a`](https://huggingface.co/datasets/mteb/Urbansound8K_t2a) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/CLAPv2/Urbansound8K)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to audio (t2a) | zxx | AudioScene | human-annotated | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Salamon:UrbanSound:ACMMM:14,
      author = {Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
      booktitle = {Proceedings of the 22nd ACM international conference on Multimedia},
      organization = {ACM},
      pages = {1041--1044},
      title = {A Dataset and Taxonomy for Urban Sound Research},
      year = {2014},
    }
    
    ```
    



#### `VQA2IT2TRetrieval` { .model-copy }

Retrieve the correct answer for a question about an image.

**Dataset:** [`mteb/vqa-2`](https://huggingface.co/datasets/mteb/vqa-2) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Goyal_2017_CVPR,
      author = {Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {July},
      title = {Making the v in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering},
      year = {2017},
    }
    
    ```
    



#### `VisualNewsI2TRetrieval` { .model-copy }

Retrieval entity-rich captions for news images.

**Dataset:** [`mteb/mbeir_visualnews_task3`](https://huggingface.co/datasets/mteb/mbeir_visualnews_task3) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.emnlp-main.542/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to text (i2t) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{liu2021visual,
      author = {Liu, Fuxiao and Wang, Yinghan and Wang, Tianlu and Ordonez, Vicente},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      pages = {6761--6771},
      title = {Visual News: Benchmark and Challenges in News Image Captioning},
      year = {2021},
    }
    
    ```
    



#### `VisualNewsT2IRetrieval` { .model-copy }

Retrieve news images with captions.

**Dataset:** [`mteb/mbeir_visualnews_task0`](https://huggingface.co/datasets/mteb/mbeir_visualnews_task0) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.emnlp-main.542/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{liu2021visual,
      author = {Liu, Fuxiao and Wang, Yinghan and Wang, Tianlu and Ordonez, Vicente},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      pages = {6761--6771},
      title = {Visual News: Benchmark and Challenges in News Image Captioning},
      year = {2021},
    }
    
    ```
    



#### `VizWizIT2TRetrieval` { .model-copy }

Retrieve the correct answer for a question about an image.

**Dataset:** [`mteb/vizwiz`](https://huggingface.co/datasets/mteb/vizwiz) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{gurari2018vizwiz,
      author = {Gurari, Danna and Li, Qing and Stangl, Abigale J and Guo, Anhong and Lin, Chi and Grauman, Kristen and Luo, Jiebo and Bigham, Jeffrey P},
      booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages = {3608--3617},
      title = {Vizwiz grand challenge: Answering visual questions from blind people},
      year = {2018},
    }
    
    ```
    



#### `WebQAT2ITRetrieval` { .model-copy }

Retrieve sources of information based on questions.

**Dataset:** [`mteb/mbeir_webqa_task2`](https://huggingface.co/datasets/mteb/mbeir_webqa_task2) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{chang2022webqa,
      author = {Chang, Yingshan and Narang, Mridu and Suzuki, Hisami and Cao, Guihong and Gao, Jianfeng and Bisk, Yonatan},
      booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
      pages = {16495--16504},
      title = {Webqa: Multihop and multimodal qa},
      year = {2022},
    }
    
    ```
    



#### `WebQAT2TRetrieval` { .model-copy }

Retrieve sources of information based on questions.

**Dataset:** [`mteb/mbeir_webqa_task1`](https://huggingface.co/datasets/mteb/mbeir_webqa_task1) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{chang2022webqa,
      author = {Chang, Yingshan and Narang, Mridu and Suzuki, Hisami and Cao, Guihong and Gao, Jianfeng and Bisk, Yonatan},
      booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
      pages = {16495--16504},
      title = {Webqa: Multihop and multimodal qa},
      year = {2022},
    }
    
    ```




## AudioReranking

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 5

#### `ESC50AudioReranking` { .model-copy }

ESC-50 environmental sound dataset adapted for audio reranking. Given a query audio of environmental sounds, rank 5 relevant audio samples higher than 16 irrelevant ones from different sound classes. Contains 200 queries across 50 environmental sound categories for robust evaluation.

**Dataset:** [`mteb/ESC50AudioReranking`](https://huggingface.co/datasets/mteb/ESC50AudioReranking) • **License:** cc-by-3.0 • [Learn more →](https://github.com/karolpiczak/ESC-50)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to audio (a2a) | zxx | AudioScene | expert-annotated | found | map_at_1000 |



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
    



#### `FSDnoisy18kAudioReranking` { .model-copy }

FSDnoisy18k sound event dataset adapted for audio reranking. Given a query audio with potential label noise, rank 4 relevant audio samples higher than 16 irrelevant ones from different sound classes. Contains 200 queries across 20 sound event categories.

**Dataset:** [`mteb/FSDnoisy18kAudioReranking`](https://huggingface.co/datasets/mteb/FSDnoisy18kAudioReranking) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/record/2529934)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to audio (a2a) | eng | AudioScene | human-annotated | found | map_at_1000 |



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
    



#### `GTZANAudioReranking` { .model-copy }

GTZAN music genre dataset adapted for audio reranking. Given a query audio from one of 10 music genres, rank 3 relevant audio samples higher than 10 irrelevant ones from different genres. Contains 100 queries across 10 music genres for comprehensive evaluation.

**Dataset:** [`mteb/GTZANAudioReranking`](https://huggingface.co/datasets/mteb/GTZANAudioReranking) • **License:** not specified • [Learn more →](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to audio (a2a) | zxx | Music | human-annotated | found | map_at_1000 |



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
    



#### `UrbanSound8KAudioReranking` { .model-copy }

UrbanSound8K urban sound dataset adapted for audio reranking. Given a query audio of urban sounds, rank 4 relevant audio samples higher than 16 irrelevant ones from different urban sound classes. Contains 200 queries across 10 urban sound categories for comprehensive evaluation.

**Dataset:** [`mteb/UrbanSound8KAudioReranking`](https://huggingface.co/datasets/mteb/UrbanSound8KAudioReranking) • **License:** cc-by-4.0 • [Learn more →](https://urbansounddataset.weebly.com/urbansound8k.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to audio (a2a) | zxx | Spoken | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Salamon:UrbanSound:ACMMM:14,
      author = {Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
      booktitle = {Proceedings of the 22nd ACM international conference on Multimedia},
      organization = {ACM},
      pages = {1041--1044},
      title = {A Dataset and Taxonomy for Urban Sound Research},
      year = {2014},
    }
    
    ```
    



#### `VocalSoundAudioReranking` { .model-copy }

VocalSound dataset adapted for audio reranking. Given a query vocal sound from one of 6 categories, rank 4 relevant vocal samples higher than 16 irrelevant ones from different vocal sound types. Contains 198 queries across 6 vocal sound categories for robust evaluation.

**Dataset:** [`mteb/VocalSoundAudioReranking`](https://huggingface.co/datasets/mteb/VocalSoundAudioReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| audio to audio (a2a) | eng | Spoken | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Gong_2022,
      author = {Gong, Yuan and Yu, Jin and Glass, James},
      booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      doi = {10.1109/icassp43922.2022.9746828},
      month = may,
      publisher = {IEEE},
      title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
      url = {http://dx.doi.org/10.1109/ICASSP43922.2022.9746828},
      year = {2022},
    }
    
    ```




## DocumentUnderstanding

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 86

#### `IRPapersT2IRetrieval` { .model-copy }

IRPAPERS is a collection of 166 Information Retrieval papers spanning 3,230 pages. Each page in the dataset is jointly represented as a base64 encoded string of the page image as well as an OCR-derived text transcription. IRPAPERS also contains 180 needle-in-the-haystack queries.

**Dataset:** [`mteb/IRPapersRetrieval`](https://huggingface.co/datasets/mteb/IRPapersRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2602.17687)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | recall_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{shorten2026,
      archiveprefix = {arXiv},
      author = {Connor Shorten and Augustas Skaburskas and Daniel M. Jones and Charles Pierse and Roberto Esposito and John Trengrove and Etienne Dilocker and Bob van Luijt},
      eprint = {2602.17687},
      primaryclass = {cs.IR},
      title = {IRPAPERS: A Visual Document Benchmark for Scientific Retrieval and Question Answering},
      url = {https://arxiv.org/pdf/2602.17687},
      year = {2026},
    }
    
    ```
    



#### `IRPapersT2ITRetrieval` { .model-copy }

IRPAPERS is a collection of 166 Information Retrieval papers spanning 3,230 pages. Each page in the dataset is jointly represented as a base64 encoded string of the page image as well as an OCR-derived text transcription. IRPAPERS also contains 180 needle-in-the-haystack queries.

**Dataset:** [`mteb/IRPapersRetrieval`](https://huggingface.co/datasets/mteb/IRPapersRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2602.17687)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | eng | Academic | derived | found | recall_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{shorten2026,
      archiveprefix = {arXiv},
      author = {Connor Shorten and Augustas Skaburskas and Daniel M. Jones and Charles Pierse and Roberto Esposito and John Trengrove and Etienne Dilocker and Bob van Luijt},
      eprint = {2602.17687},
      primaryclass = {cs.IR},
      title = {IRPAPERS: A Visual Document Benchmark for Scientific Retrieval and Question Answering},
      url = {https://arxiv.org/pdf/2602.17687},
      year = {2026},
    }
    
    ```
    



#### `JinaVDRAirbnbSyntheticRetrieval` { .model-copy }

Retrieve rendered tables from Airbnb listings based on templated queries. This dataset is created from the original Kaggle [New York City Airbnb Open Data dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

**Dataset:** [`jinaai/airbnb-synthetic-retrieval_beir`](https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, deu, eng, fra, hin, ... (10) | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRArabicChartQARetrieval` { .model-copy }

Retrieve Arabic charts based on queries. This dataset is derived from the [Arabic ChartQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_chartqa), reformatting the train split as a test split with modified field names such that it is compatible with the ViDoRe evaluation benchmark.

**Dataset:** [`jinaai/arabic_chartqa_ar_beir`](https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRArabicInfographicsVQARetrieval` { .model-copy }

Retrieve Arabic infographics based on queries. This dataset is derived from the [Arabic Infographics VQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_infographicsvqa), reformatting the train split as a test split with modified field names so it can be used in the ViDoRe evaluation benchmark.

**Dataset:** [`jinaai/arabic_infographicsvqa_ar_beir`](https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRArxivQARetrieval` { .model-copy }

Retrieve figures from scientific papers from arXiv based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/arxivqa_beir`](https://huggingface.co/datasets/jinaai/arxivqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/arxivqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRAutomobileCatelogRetrieval` { .model-copy }

Retrieve automobile marketing documents based on LLM generated queries. Marketing document from Toyota Japanese website featuring [RAV4](https://toyota.jp/pages/contents/request/webcatalog/rav4/rav4_special1_202310.pdf) and [Corolla](https://toyota.jp/pages/contents/request/webcatalog/corolla/corolla_special1_202407.pdf). The `text_description` column contains OCR text extracted from the images using EasyOCR.

**Dataset:** [`jinaai/automobile_catalogue_jp_beir`](https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | jpn | Engineering, Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRBeveragesCatalogueRetrieval` { .model-copy }

Retrieve beverages marketing documents based on LLM generated queries. This dataset was self-curated by searching beverage catalogs on Google search and downloading PDFs.

**Dataset:** [`jinaai/beverages_catalogue_ru_beir`](https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | rus | Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRCharXivOCRRetrieval` { .model-copy }

Retrieve charts from scientific papers based on human annotated queries. This dataset is derived from the [CharXiv dataset](https://huggingface.co/datasets/princeton-nlp/CharXiv), reformatting the test split with modified field names, so that it can be used in the ViDoRe benchmark.

**Dataset:** [`jinaai/CharXiv-en_beir`](https://huggingface.co/datasets/jinaai/CharXiv-en_beir) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/CharXiv-en_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRChartQARetrieval` { .model-copy }

Retrieve charts based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA

**Dataset:** [`jinaai/ChartQA_beir`](https://huggingface.co/datasets/jinaai/ChartQA_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/ChartQA_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRDocQAAI` { .model-copy }

Retrieve AI documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docqa_artificial_intelligence_beir`](https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRDocQAEnergyRetrieval` { .model-copy }

Retrieve energy industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docqa_energy_beir`](https://huggingface.co/datasets/jinaai/docqa_energy_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_energy_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRDocQAGovReportRetrieval` { .model-copy }

Retrieve government reports based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docqa_gov_report_beir`](https://huggingface.co/datasets/jinaai/docqa_gov_report_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_gov_report_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Government | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRDocQAHealthcareIndustryRetrieval` { .model-copy }

Retrieve healthcare industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d). For more information regarding the filtering please read [our paper](https://arxiv.org/abs/2506.18902) or [this discussion on github](https://github.com/embeddings-benchmark/mteb/pull/2942#discussion_r2240711654).

**Dataset:** [`jinaai/docqa_healthcare_industry_beir`](https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Medical | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRDocVQARetrieval` { .model-copy }

Retrieve industry documents based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docvqa_beir`](https://huggingface.co/datasets/jinaai/docvqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/docvqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRDonutVQAISynHMPRetrieval` { .model-copy }

Retrieve medical records based on templated queries. Source dataset https://huggingface.co/datasets/warshakhan/donut_vqa_ISynHMP

**Dataset:** [`jinaai/donut_vqa_beir`](https://huggingface.co/datasets/jinaai/donut_vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/donut_vqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Medical | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDREuropeanaDeNewsRetrieval` { .model-copy }

Retrieve German news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of German news articles

**Dataset:** [`jinaai/europeana-de-news_beir`](https://huggingface.co/datasets/jinaai/europeana-de-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-de-news_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu | News | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDREuropeanaEsNewsRetrieval` { .model-copy }

Retrieve Spanish news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Spanish news articles

**Dataset:** [`jinaai/europeana-es-news_beir`](https://huggingface.co/datasets/jinaai/europeana-es-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-es-news_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | spa | News | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDREuropeanaFrNewsRetrieval` { .model-copy }

Retrieve French news articles from Europeana based on LLM generated queries. This dataset was created from records of the [Europeana online collection](https://europeana.eu) by selecting scans of French news articles.

**Dataset:** [`jinaai/europeana-fr-news_beir`](https://huggingface.co/datasets/jinaai/europeana-fr-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-fr-news_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | fra | News | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDREuropeanaItScansRetrieval` { .model-copy }

Retrieve Italian historical articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Italian news articles

**Dataset:** [`jinaai/europeana-it-scans_beir`](https://huggingface.co/datasets/jinaai/europeana-it-scans_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-it-scans_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ita | News | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDREuropeanaNlLegalRetrieval` { .model-copy }

Retrieve Dutch historical legal documents based on LLM generated queries.  This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Dutch news articles

**Dataset:** [`jinaai/europeana-nl-legal_beir`](https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | nld | Legal | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRGitHubReadmeRetrieval` { .model-copy }

Retrieve GitHub readme files based their description. This dataset consists of rendered GitHub readmes in a variety of different languages, together with their accompanying descriptions as queries and their license in the `license_type` and `license_text` columns. This particular dataset is a subsample of 1000 random rows per language from the full dataset which can be found [here](https://huggingface.co/datasets/jinaai/github-readme-retrieval-ml-filtered).

**Dataset:** [`jinaai/github-readme-retrieval-multilingual_beir`](https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, ben, deu, eng, fra, ... (17) | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRHindiGovVQARetrieval` { .model-copy }

Retrieve Hindi government documents based on LLM generated queries.

**Dataset:** [`jinaai/hindi-gov-vqa_beir`](https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | hin | Government | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRHungarianDocQARetrieval` { .model-copy }

Retrieve Hungarian documents in various formats based on human annotated queries. Document Question answering from [Hungurian doc qa dataset](https://huggingface.co/datasets/jlli/HungarianDocQA-OCR), test split.

**Dataset:** [`jinaai/hungarian_doc_qa_beir`](https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | hun | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRInfovqaRetrieval` { .model-copy }

Retrieve infographics based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/infovqa_beir`](https://huggingface.co/datasets/jinaai/infovqa_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/infovqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRJDocQARetrieval` { .model-copy }

Retrieve Japanese documents in various formats based on human annotated queries. Document Question answering from [JDocQAJP dataset](https://huggingface.co/datasets/jlli/JDocQA-nonbinary), test split.

**Dataset:** [`jinaai/jdocqa_beir`](https://huggingface.co/datasets/jinaai/jdocqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/jdocqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | jpn | Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRJina2024YearlyBookRetrieval` { .model-copy }

Retrieve pages from the 2024 Jina yearbook based on human annotated questions. 75 human annotated questions created from digital version of Jina AI yearly book 2024, 166 pages in total. 

**Dataset:** [`jinaai/jina_2024_yearly_book_beir`](https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRMMTabRetrieval` { .model-copy }

Retrieve tables from the MMTab dataset based on queries. This dataset is a copy of the original test split from MMTab, taking only items where an 'original_query' is present, and removing the 'input' and 'output' columns, as they are unnecessary for retrieval tasks.

**Dataset:** [`jinaai/MMTab_beir`](https://huggingface.co/datasets/jinaai/MMTab_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/MMTab_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRMPMQARetrieval` { .model-copy }

Retrieve product manuals based on human annotated queries. 155 questions and 782 document images cleaned from [jinaai/MPMQA](https://huggingface.co/datasets/jinaai/MPMQA), test set.

**Dataset:** [`jinaai/mpmqa_small_beir`](https://huggingface.co/datasets/jinaai/mpmqa_small_beir) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/jinaai/mpmqa_small_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | human-annotated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRMedicalPrescriptionsRetrieval` { .model-copy }

Retrieve medical prescriptions based on templated queries. Source dataset https://huggingface.co/datasets/Technoculture/medical-prescriptions

**Dataset:** [`jinaai/medical-prescriptions_beir`](https://huggingface.co/datasets/jinaai/medical-prescriptions_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/medical-prescriptions_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Medical | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDROWIDChartsRetrieval` { .model-copy }

Retrieve charts from the OWID dataset based on accompanied text snippets. We sampled a set of ~5k charts and articles from [Our World In Data](https://ourworldindata.org) to produce this evaluation set. This particular dataset is a subsample of 1000 random charts from the full dataset which can be found [here](https://huggingface.co/datasets/jjinaai/owid_charts).

**Dataset:** [`jinaai/owid_charts_en_beir`](https://huggingface.co/datasets/jinaai/owid_charts_en_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/owid_charts_en_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDROpenAINewsRetrieval` { .model-copy }

Retrieve news articles from the OpenAI news website based on human annotated queries. News taken from https://openai.com/news/

**Dataset:** [`jinaai/openai-news_beir`](https://huggingface.co/datasets/jinaai/openai-news_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/openai-news_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | News, Web | human-annotated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRPlotQARetrieval` { .model-copy }

Retrieve plots from the PlotQA dataset based on LLM generated queries. Questions subsampled from [PlotQA](https://github.com/NiteshMethani/PlotQA) test set. It is following a subsample + LLM-based classification process, using LLM to verify the question quality, e.g. queries like `How many different coloured dotlines are there` will be filtered out.

**Dataset:** [`jinaai/plotqa_beir`](https://huggingface.co/datasets/jinaai/plotqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/plotqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRRamensBenchmarkRetrieval` { .model-copy }

Retrieve ramen restaurant marketing documents based on LLM generated queries. Marketing document from Ramen [restaurants](https://www.city.niigata.lg.jp/kanko/kanko/oshirase/ramen.files/guidebook.pdf).

**Dataset:** [`jinaai/ramen_benchmark_jp_beir`](https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | jpn | Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRShanghaiMasterPlanRetrieval` { .model-copy }

Retrieve pages from the Shanghai Master Plan based on human annotated queries. The master plan document is taken from [here](https://www.shanghai.gov.cn/newshanghai/xxgkfj/2035004.pdf).

**Dataset:** [`jinaai/shanghai_master_plan_beir`](https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | zho | Web | human-annotated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRShiftProjectRetrieval` { .model-copy }

Retrieve documents with graphs from the Shift Project based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/shiftproject_beir`](https://huggingface.co/datasets/jinaai/shiftproject_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/shiftproject_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRStanfordSlideRetrieval` { .model-copy }

Retrieve scientific and engineering slides based on human annotated queries. Source dataset https://exhibits.stanford.edu/data/catalog/mv327tb8364

**Dataset:** [`jinaai/stanford_slide_beir`](https://huggingface.co/datasets/jinaai/stanford_slide_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/stanford_slide_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | human-annotated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRStudentEnrollmentSyntheticRetrieval` { .model-copy }

Retrieve student enrollment data based on templated queries. This dataset is created from the original Kaggle [Delaware Student Enrollment](https://www.kaggle.com/datasets/noeyislearning/delaware-student-enrollment) dataset. The charts are rendered and queries created using templates.

**Dataset:** [`jinaai/student-enrollment_beir`](https://huggingface.co/datasets/jinaai/student-enrollment_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/student-enrollment_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRTQARetrieval` { .model-copy }

Retrieve textbook pages (images and text) based on LLM generated queries from the text. Source datasets https://prior.allenai.org/projects/tqa

**Dataset:** [`jinaai/tqa_beir`](https://huggingface.co/datasets/jinaai/tqa_beir) • **License:** cc-by-nc-3.0 • [Learn more →](https://huggingface.co/datasets/jinaai/tqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRTabFQuadRetrieval` { .model-copy }

Retrieve tables from industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/tabfquad_beir`](https://huggingface.co/datasets/jinaai/tabfquad_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/tabfquad_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRTableVQARetrieval` { .model-copy }

Retrieve scientific tables based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA or https://huggingface.co/datasets/cmarkea/aftdb

**Dataset:** [`jinaai/table-vqa_beir`](https://huggingface.co/datasets/jinaai/table-vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/table-vqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRTatQARetrieval` { .model-copy }

Retrieve financial reports based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/tatqa_beir`](https://huggingface.co/datasets/jinaai/tatqa_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/tatqa_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRTweetStockSyntheticsRetrieval` { .model-copy }

Retrieve rendered tables of stock prices based on templated queries. This dataset is created from the original Kaggle [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns) dataset.

**Dataset:** [`jinaai/tweet-stock-synthetic-retrieval_beir`](https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, deu, eng, fra, hin, ... (10) | Social | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRWikimediaCommonsDocumentsRetrieval` { .model-copy }

Retrieve historical documents from Wikimedia Commons based on their description. Wikimedia Commons Documents. It contains images of (mostly historic) documents which should be identified based on their description. We extracted those descriptions from Wikimedia Commons. We have included the license type and a link (`license_text`) to the original Wikimedia Commons page for each extracted image.

**Dataset:** [`jinaai/wikimedia-commons-documents-ml_beir`](https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, ben, deu, eng, fra, ... (20) | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `JinaVDRWikimediaCommonsMapsRetrieval` { .model-copy }

Retrieve maps from Wikimedia Commons based on their description. It contains images of (mostly historic) maps which should be identified based on their description. We extracted those descriptions from [Wikimedia Commons](https://commons.wikimedia.org/). We have included the license type and a link (license_text) to the original Wikimedia Commons page for each extracted image.

**Dataset:** [`jinaai/wikimedia-commons-maps_beir`](https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```
    



#### `KoVidore2CybersecurityRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Cybersecurity, is a corpus of technical reports on cyber threat trends and security incident responses in Korea, intended for complex-document understanding tasks.

**Dataset:** [`whybe-choi/kovidore-v2-cybersecurity-mteb`](https://huggingface.co/datasets/whybe-choi/kovidore-v2-cybersecurity-mteb) • **License:** cc-by-4.0 • [Learn more →](https://github.com/whybe-choi/kovidore-data-generator)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | kor | Social | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{choi2026kovidorev2,
      author = {Yongbin Choi},
      note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains},
      title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
      url = {https://github.com/whybe-choi/kovidore-data-generator},
      year = {2026},
    }
    
    ```
    



#### `KoVidore2EconomicRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Economic trends, is a corpus of periodic reports on major economic indicators in Korea, intended for complex-document understanding tasks.

**Dataset:** [`whybe-choi/kovidore-v2-economic-mteb`](https://huggingface.co/datasets/whybe-choi/kovidore-v2-economic-mteb) • **License:** cc-by-4.0 • [Learn more →](https://github.com/whybe-choi/kovidore-data-generator)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | kor | Social | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{choi2026kovidorev2,
      author = {Yongbin Choi},
      note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains},
      title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
      url = {https://github.com/whybe-choi/kovidore-data-generator},
      year = {2026},
    }
    
    ```
    



#### `KoVidore2EnergyRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Energy, is a corpus of reports on energy market trends, policy planning, and industry statistics, intended for complex-document understanding tasks.

**Dataset:** [`whybe-choi/kovidore-v2-energy-mteb`](https://huggingface.co/datasets/whybe-choi/kovidore-v2-energy-mteb) • **License:** cc-by-4.0 • [Learn more →](https://github.com/whybe-choi/kovidore-data-generator)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | kor | Social | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{choi2026kovidorev2,
      author = {Yongbin Choi},
      note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains},
      title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
      url = {https://github.com/whybe-choi/kovidore-data-generator},
      year = {2026},
    }
    
    ```
    



#### `KoVidore2HrRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports on workforce outlook and employment policy in korea, intended for complex-document understanding tasks.

**Dataset:** [`whybe-choi/kovidore-v2-hr-mteb`](https://huggingface.co/datasets/whybe-choi/kovidore-v2-hr-mteb) • **License:** cc-by-4.0 • [Learn more →](https://github.com/whybe-choi/kovidore-data-generator)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | kor | Social | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{choi2026kovidorev2,
      author = {Yongbin Choi},
      note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains},
      title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
      url = {https://github.com/whybe-choi/kovidore-data-generator},
      year = {2026},
    }
    
    ```
    



#### `MIRACLVisionRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`nvidia/miracl-vision`](https://huggingface.co/datasets/nvidia/miracl-vision) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2505.11651)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic | derived | created | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{osmulski2025miraclvisionlargemultilingualvisual,
      author = {Radek Osmulski and Gabriel de Souza P. Moreira and Ronay Ak and Mengyao Xu and Benedikt Schifferer and Even Oldridge},
      eprint = {2505.11651},
      journal = {arxiv},
      title = {{MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark}},
      url = {https://arxiv.org/abs/2505.11651},
      year = {2025},
    }
    
    ```
    



#### `MMDocIRT2IRetrieval` { .model-copy }

MMDocIR evaluation set includes 313 long documents averaging 65.1 pages, categorized into ten main domains: research reports, administration&industry, tutorials&workshops, academic papers, brochures, financial reports, guidebooks, government documents, laws, and news articles. Different domains feature distinct distributions of multi-modal information. Overall, the modality distribution is: Text (60.4%), Image (18.8%), Table (16.7%), and other modalities (4.1%).

**Dataset:** [`mteb/MMDocIRT2ITRetrieval`](https://huggingface.co/datasets/mteb/MMDocIRT2ITRetrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2501.08828)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic, Government, Legal, News, Non-fiction | derived | found | recall_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{dong2025mmdocirbenchmarkingmultimodalretrieval,
      archiveprefix = {arXiv},
      author = {Kuicai Dong and Yujing Chang and Xin Deik Goh and Dexun Li and Ruiming Tang and Yong Liu},
      eprint = {2501.08828},
      primaryclass = {cs.IR},
      title = {MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents},
      url = {https://arxiv.org/abs/2501.08828},
      year = {2025},
    }
    
    ```
    



#### `MMDocIRT2ITRetrieval` { .model-copy }

MMDocIR evaluation set includes 313 long documents averaging 65.1 pages, categorized into ten main domains: research reports, administration&industry, tutorials&workshops, academic papers, brochures, financial reports, guidebooks, government documents, laws, and news articles. Different domains feature distinct distributions of multi-modal information. Overall, the modality distribution is: Text (60.4%), Image (18.8%), Table (16.7%), and other modalities (4.1%).

**Dataset:** [`mteb/MMDocIRT2ITRetrieval`](https://huggingface.co/datasets/mteb/MMDocIRT2ITRetrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2501.08828)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | eng | Academic, Government, Legal, News, Non-fiction | derived | found | recall_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{dong2025mmdocirbenchmarkingmultimodalretrieval,
      archiveprefix = {arXiv},
      author = {Kuicai Dong and Yujing Chang and Xin Deik Goh and Dexun Li and Ruiming Tang and Yong Liu},
      eprint = {2501.08828},
      primaryclass = {cs.IR},
      title = {MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents},
      url = {https://arxiv.org/abs/2501.08828},
      year = {2025},
    }
    
    ```
    



#### `Vidore2BioMedicalLecturesRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`vidore/biomedical_lectures_v2`](https://huggingface.co/datasets/vidore/biomedical_lectures_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, spa | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }
    
    ```
    



#### `Vidore2ESGReportsHLRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`vidore/esg_reports_human_labeled_v2`](https://huggingface.co/datasets/vidore/esg_reports_human_labeled_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }
    
    ```
    



#### `Vidore2ESGReportsRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`vidore/esg_reports_v2`](https://huggingface.co/datasets/vidore/esg_reports_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, spa | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }
    
    ```
    



#### `Vidore2EconomicsReportsRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`vidore/economics_reports_v2`](https://huggingface.co/datasets/vidore/economics_reports_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, spa | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }
    
    ```
    



#### `Vidore3ComputerScienceRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_computer_science_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_computer_science_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Engineering, Programming | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3ComputerScienceRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3ComputerScienceOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3ComputerScienceOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Engineering, Programming | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3EnergyRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_energy_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_energy_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Academic, Chemistry, Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3EnergyRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3EnergyOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3EnergyOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Academic, Chemistry, Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3FinanceEnRetrieval` { .model-copy }

Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_finance_en_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_finance_en_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Financial | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3FinanceEnRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3FinanceEnOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3FinanceEnOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Financial | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3FinanceFrRetrieval` { .model-copy }

Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_finance_fr_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_finance_fr_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Financial | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3FinanceFrRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3FinanceFrOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3FinanceFrOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Financial | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3HrRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_hr_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_hr_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Social | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3HrRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3HrOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3HrOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Social | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3IndustrialRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_industrial_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_industrial_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3IndustrialRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3IndustrialOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3IndustrialOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3NuclearRetrieval` { .model-copy }

Retrieve associated pages according to questions. Contributed by Illuin Technology.

**Dataset:** [`mteb-private/Vidore3NuclearRetrieval`](https://huggingface.co/datasets/mteb-private/Vidore3NuclearRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Chemistry, Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3NuclearRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb-private/Vidore3NuclearOCRRetrieval`](https://huggingface.co/datasets/mteb-private/Vidore3NuclearOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Chemistry, Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3PharmaceuticalsRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_pharmaceuticals_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_pharmaceuticals_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Medical | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3PharmaceuticalsRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3PharmaceuticalsOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3PharmaceuticalsOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Medical | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3PhysicsRetrieval` { .model-copy }

Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish. Contributed by Illuin Technology.

**Dataset:** [`vidore/vidore_v3_physics_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_physics_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Academic, Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3PhysicsRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb/Vidore3PhysicsOCRRetrieval`](https://huggingface.co/datasets/mteb/Vidore3PhysicsOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Academic, Engineering | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3TelecomRetrieval` { .model-copy }

Retrieve associated pages according to questions. Contributed by Illuin Technology.

**Dataset:** [`mteb-private/Vidore3TelecomRetrieval`](https://huggingface.co/datasets/mteb-private/Vidore3TelecomRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, por, ... (6) | Engineering, Programming | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `Vidore3TelecomRetrieval.v2` { .model-copy }

Retrieve associated pages according to questions.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Contributed by Illuin Technology.

**Dataset:** [`mteb-private/Vidore3TelecomOCRRetrieval`](https://huggingface.co/datasets/mteb-private/Vidore3TelecomOCRRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2601.08620)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image, text (t2it) | deu, eng, fra, ita, por, ... (6) | Engineering, Programming | derived | created and machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{loison2026vidorev3comprehensiveevaluation,
      archiveprefix = {arXiv},
      author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
      eprint = {2601.08620},
      primaryclass = {cs.AI},
      title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
      url = {https://arxiv.org/abs/2601.08620},
      year = {2026},
    }
    
    ```
    



#### `VidoreArxivQARetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/arxivqa_test_subsampled_beir`](https://huggingface.co/datasets/mteb/arxivqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreDocVQARetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/docvqa_test_subsampled_beir`](https://huggingface.co/datasets/mteb/docvqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreInfoVQARetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/infovqa_test_subsampled_beir`](https://huggingface.co/datasets/mteb/infovqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreShiftProjectRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/shiftproject_test_beir`](https://huggingface.co/datasets/mteb/shiftproject_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreSyntheticDocQAAIRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/syntheticDocQA_artificial_intelligence_test_beir`](https://huggingface.co/datasets/mteb/syntheticDocQA_artificial_intelligence_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreSyntheticDocQAEnergyRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/syntheticDocQA_energy_test_beir`](https://huggingface.co/datasets/mteb/syntheticDocQA_energy_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreSyntheticDocQAGovernmentReportsRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/syntheticDocQA_government_reports_test_beir`](https://huggingface.co/datasets/mteb/syntheticDocQA_government_reports_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreSyntheticDocQAHealthcareIndustryRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/syntheticDocQA_healthcare_industry_test_beir`](https://huggingface.co/datasets/mteb/syntheticDocQA_healthcare_industry_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreTabfquadRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/tabfquad_test_subsampled_beir`](https://huggingface.co/datasets/mteb/tabfquad_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```
    



#### `VidoreTatdqaRetrieval` { .model-copy }

Retrieve associated pages according to questions.

**Dataset:** [`mteb/tatdqa_test_beir`](https://huggingface.co/datasets/mteb/tatdqa_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }
    
    ```




## InstructionReranking

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 5

#### `Core17InstructionRetrieval` { .model-copy }

Measuring retrieval instruction following ability on Core17 narratives for the FollowIR benchmark.

**Dataset:** [`mteb/Core17InstructionRetrieval`](https://huggingface.co/datasets/mteb/Core17InstructionRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/abs/2403.15246)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | derived | found | p-MRR |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2024followir,
      archiveprefix = {arXiv},
      author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      eprint = {2403.15246},
      primaryclass = {cs.IR},
      title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
      year = {2024},
    }
    
    ```
    



#### `News21InstructionRetrieval` { .model-copy }

Measuring retrieval instruction following ability on News21 narratives for the FollowIR benchmark.

**Dataset:** [`mteb/News21InstructionRetrieval`](https://huggingface.co/datasets/mteb/News21InstructionRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/abs/2403.15246)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | derived | found | p-MRR |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2024followir,
      archiveprefix = {arXiv},
      author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      eprint = {2403.15246},
      primaryclass = {cs.IR},
      title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
      year = {2024},
    }
    
    ```
    



#### `Robust04InstructionRetrieval` { .model-copy }

Measuring retrieval instruction following ability on Robust04 narratives for the FollowIR benchmark.

**Dataset:** [`mteb/Robust04InstructionRetrieval`](https://huggingface.co/datasets/mteb/Robust04InstructionRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/abs/2403.15246)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | derived | found | p-MRR |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2024followir,
      archiveprefix = {arXiv},
      author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      eprint = {2403.15246},
      primaryclass = {cs.IR},
      title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
      year = {2024},
    }
    
    ```
    



#### `mFollowIR` { .model-copy }

This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages.

**Dataset:** [`jhu-clsp/mFollowIR-parquet-mteb`](https://huggingface.co/datasets/jhu-clsp/mFollowIR-parquet-mteb) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas, rus, zho | News, Written | expert-annotated | found | p-MRR |



??? quote "Citation"

    
    ```bibtex
    
    @article{weller2024mfollowir,
      author = {Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Sam and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
      journal = {arXiv preprint TODO},
      title = {{mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieval}},
      year = {2024},
    }
    
    ```
    



#### `mFollowIRCrossLingual` { .model-copy }

This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages with English queries/instructions.

**Dataset:** [`jhu-clsp/mFollowIR-cross-lingual-parquet-mteb`](https://huggingface.co/datasets/jhu-clsp/mFollowIR-cross-lingual-parquet-mteb) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, fas, rus, zho | News, Written | expert-annotated | found | p-MRR |



??? quote "Citation"

    
    ```bibtex
    
    @article{weller2024mfollowir,
      author = {Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Sam and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
      journal = {arXiv preprint TODO},
      title = {{mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieval}},
      year = {2024},
    }
    
    ```




## InstructionRetrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 8

#### `IFIRAila` { .model-copy }

Benchmark aila subset in aila within instruction following abilities. The instructions simulate lawyers' or legal assistants' nuanced queries to retrieve relevant legal documents. 

**Dataset:** [`if-ir/aila`](https://huggingface.co/datasets/if-ir/aila) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | human-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `IFIRCds` { .model-copy }

Benchmark IFIR cds subset within instruction following abilities. The instructions simulate a doctor's nuanced queries to retrieve suitable clinical trails, treatment and diagnosis information. 

**Dataset:** [`if-ir/cds`](https://huggingface.co/datasets/if-ir/cds) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical, Written | human-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `IFIRFiQA` { .model-copy }

Benchmark IFIR fiqa subset within instruction following abilities. The instructions simulate people's daily life queries to retrieve suitable financial suggestions. 

**Dataset:** [`if-ir/fiqa`](https://huggingface.co/datasets/if-ir/fiqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Financial, Written | human-annotated | created | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `IFIRFire` { .model-copy }

Benchmark IFIR fire subset within instruction following abilities. The instructions simulate lawyers' or legal assistants' nuanced queries to retrieve relevant legal documents. 

**Dataset:** [`if-ir/fire`](https://huggingface.co/datasets/if-ir/fire) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | human-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `IFIRNFCorpus` { .model-copy }

Benchmark IFIR nfcorpus subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature in the medical and biological domains. 

**Dataset:** [`if-ir/nfcorpus`](https://huggingface.co/datasets/if-ir/nfcorpus) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | human-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `IFIRPm` { .model-copy }

Benchmark IFIR pm subset within instruction following abilities. The instructions simulate a doctor's nuanced queries to retrieve suitable clinical trails, treatment and diagnosis information. 

**Dataset:** [`if-ir/pm`](https://huggingface.co/datasets/if-ir/pm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical, Written | human-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `IFIRScifact` { .model-copy }

Benchmark IFIR scifact_open subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature. 

**Dataset:** [`if-ir/scifact_open`](https://huggingface.co/datasets/if-ir/scifact_open) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Written | human-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{song2025ifir,
      author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
      booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
      pages = {10186--10204},
      title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
      year = {2025},
    }
    
    ```
    



#### `InstructIR` { .model-copy }

A benchmark specifically designed to evaluate the instruction following ability in information retrieval models. Our approach focuses on user-aligned instructions tailored to each query instance, reflecting the diverse characteristics inherent in real-world search scenarios. **NOTE**: scores on this may differ unless you include instruction first, then "[SEP]" and then the query via redefining `combine_query_and_instruction` in your model.

**Dataset:** [`mteb/InstructIR-mteb`](https://huggingface.co/datasets/mteb/InstructIR-mteb) • **License:** mit • [Learn more →](https://github.com/kaistAI/InstructIR/tree/main)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web | human-annotated | created | robustness_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{oh2024instructir,
      archiveprefix = {{arXiv}},
      author = {{Hanseok Oh and Hyunji Lee and Seonghyeon Ye and Haebin Shin and Hansol Jang and Changwook Jun and Minjoon Seo}},
      eprint = {{2402.14334}},
      primaryclass = {{cs.CL}},
      title = {{INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models}},
      year = {{2024}},
    }
    
    ```




## Reranking

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 43

#### `AlloprofReranking` { .model-copy }

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`mteb/AlloprofReranking`](https://huggingface.co/datasets/mteb/AlloprofReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Academic, Web, Written | expert-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lef23,
      author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
      copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
      doi = {10.48550/ARXIV.2302.07738},
      keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      publisher = {arXiv},
      title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
      url = {https://arxiv.org/abs/2302.07738},
      year = {2023},
    }
    
    ```
    



#### `AskUbuntuDupQuestions` { .model-copy }

AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar

**Dataset:** [`mteb/AskUbuntuDupQuestions`](https://huggingface.co/datasets/mteb/AskUbuntuDupQuestions) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/taolei87/askubuntu)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Web | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{wang-2021-TSDAE,
      author = {Wang, Kexin and Reimers, Nils and  Gurevych, Iryna},
      journal = {arXiv preprint arXiv:2104.06979},
      month = {4},
      title = {TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning},
      url = {https://arxiv.org/abs/2104.06979},
      year = {2021},
    }
    
    ```
    



#### `AskUbuntuDupQuestions-VN` { .model-copy }

A translated dataset from AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`mteb/AskUbuntuDupQuestions-VN`](https://huggingface.co/datasets/mteb/AskUbuntuDupQuestions-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/taolei87/askubuntu)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Programming, Web | derived | machine-translated and LM verified | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `BuiltBenchReranking` { .model-copy }

Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchReranking`](https://huggingface.co/datasets/mteb/BuiltBenchReranking) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Engineering, Written | derived | created | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    
    ```
    



#### `CMedQAv1-reranking` { .model-copy }

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv1-reranking`](https://huggingface.co/datasets/mteb/CMedQAv1-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Medical, Written | expert-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhang2017chinese,
      author = {Zhang, Sheng and Zhang, Xin and Wang, Hui and Cheng, Jiajun and Li, Pei and Ding, Zhaoyun},
      journal = {Applied Sciences},
      number = {8},
      pages = {767},
      publisher = {Multidisciplinary Digital Publishing Institute},
      title = {Chinese Medical Question Answer Matching Using End-to-End Character-Level Multi-Scale CNNs},
      volume = {7},
      year = {2017},
    }
    
    ```
    



#### `CMedQAv2-reranking` { .model-copy }

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv2-reranking`](https://huggingface.co/datasets/mteb/CMedQAv2-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Medical, Written | expert-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{8548603,
      author = {S. Zhang and X. Zhang and H. Wang and L. Guo and S. Liu},
      doi = {10.1109/ACCESS.2018.2883637},
      issn = {2169-3536},
      journal = {IEEE Access},
      keywords = {Biomedical imaging;Data mining;Semantics;Medical services;Feature extraction;Knowledge discovery;Medical question answering;interactive attention;deep learning;deep neural networks},
      month = {},
      number = {},
      pages = {74061-74071},
      title = {Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection},
      volume = {6},
      year = {2018},
    }
    
    ```
    



#### `CodeRAGLibraryDocumentationSolutions` { .model-copy }

Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.

**Dataset:** [`code-rag-bench/library-documentation`](https://huggingface.co/datasets/code-rag-bench/library-documentation) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | python | Programming | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }
        
    ```
    



#### `CodeRAGOnlineTutorials` { .model-copy }

Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.

**Dataset:** [`code-rag-bench/online-tutorials`](https://huggingface.co/datasets/code-rag-bench/online-tutorials) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | python | Programming | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }
        
    ```
    



#### `CodeRAGProgrammingSolutions` { .model-copy }

Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.

**Dataset:** [`code-rag-bench/programming-solutions`](https://huggingface.co/datasets/code-rag-bench/programming-solutions) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | python | Programming | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }
        
    ```
    



#### `CodeRAGStackoverflowPosts` { .model-copy }

Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.

**Dataset:** [`code-rag-bench/stackoverflow-posts`](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | python | Programming | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }
        
    ```
    



#### `ERESSReranking` { .model-copy }

ERESS is a comprehensive e-commerce reranking dataset designed for holistic
    evaluation of reranking models. It includes diverse query intents including
    attribute-rich queries, navigational queries, gift/audience-specific queries,
    utility queries, and more.

**Dataset:** [`thebajajra/ERESSReranking`](https://huggingface.co/datasets/thebajajra/ERESSReranking) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/thebajajra/ERESSReranking)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | E-commerce, Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @article{Bajaj2026RexRerankers,
      author = {Bajaj, Rahul and Garg, Anuj and Nupur, Jaya},
      journal = {Hugging Face Blog (Community Article)},
      month = jan,
      title = {{RexRerankers}: {SOTA} Rankers for Product Discovery and {AI} Assistants},
      url = {https://huggingface.co/blog/thebajajra/rexrerankers},
      urldate = {2026-01-24},
      year = {2026},
    }
    
    ```
    



#### `ESCIReranking` { .model-copy }



**Dataset:** [`mteb/ESCIReranking`](https://huggingface.co/datasets/mteb/ESCIReranking) • **License:** apache-2.0 • [Learn more →](https://github.com/amazon-science/esci-data/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, jpn, spa | Written | derived | created | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    @article{reddy2022shopping,
      archiveprefix = {arXiv},
      author = {Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
      eprint = {2206.06588},
      title = {Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
      year = {2022},
    }
    ```
    



#### `HUMECore17InstructionReranking` { .model-copy }

Human evaluation subset of Core17 instruction retrieval dataset for reranking evaluation.

**Dataset:** [`mteb/HUMECore17InstructionReranking`](https://huggingface.co/datasets/mteb/HUMECore17InstructionReranking) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2403.15246)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2024followir,
      archiveprefix = {arXiv},
      author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      eprint = {2403.15246},
      primaryclass = {cs.IR},
      title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
      year = {2024},
    }
    
    ```
    



#### `HUMENews21InstructionReranking` { .model-copy }

Human evaluation subset of News21 instruction retrieval dataset for reranking evaluation.

**Dataset:** [`mteb/HUMENews21InstructionReranking`](https://huggingface.co/datasets/mteb/HUMENews21InstructionReranking) • **License:** not specified • [Learn more →](https://trec.nist.gov/data/news2021.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{soboroff2021trec,
      author = {Soboroff, Ian and Macdonald, Craig and McCreadie, Richard},
      booktitle = {TREC},
      title = {TREC 2021 News Track Overview},
      year = {2021},
    }
    
    ```
    



#### `HUMERobust04InstructionReranking` { .model-copy }

Human evaluation subset of Robust04 instruction retrieval dataset for reranking evaluation.

**Dataset:** [`mteb/HUMERobust04InstructionReranking`](https://huggingface.co/datasets/mteb/HUMERobust04InstructionReranking) • **License:** not specified • [Learn more →](https://trec.nist.gov/data/robust/04.guidelines.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{voorhees2005trec,
      author = {Voorhees, Ellen M},
      booktitle = {TREC},
      title = {TREC 2004 Robust Retrieval Track Overview},
      year = {2005},
    }
    
    ```
    



#### `HUMEWikipediaRerankingMultilingual` { .model-copy }

Human evaluation subset of Wikipedia reranking dataset across multiple languages.

**Dataset:** [`mteb/HUMEWikipediaRerankingMultilingual`](https://huggingface.co/datasets/mteb/HUMEWikipediaRerankingMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan, eng, nob | Encyclopaedic, Written | derived | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wikipedia_reranking_2023,
      author = {Ellamind},
      title = {Wikipedia 2023-11 Reranking Multilingual Dataset},
      url = {https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual},
      year = {2023},
    }
    
    ```
    



#### `JQaRAReranking` { .model-copy }

JQaRA: Japanese Question Answering with Retrieval Augmentation  - 検索拡張(RAG)評価のための日本語 Q&A データセット. JQaRA is an information retrieval task for questions against 100 candidate data (including one or more correct answers).

**Dataset:** [`mteb/JQaRAReranking`](https://huggingface.co/datasets/mteb/JQaRAReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/hotchpotch/JQaRA)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Non-fiction, Written | derived | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yuichi-tateno-2024-jqara,
      author = {Yuichi Tateno},
      title = {JQaRA: Japanese Question Answering with Retrieval Augmentation - 検索拡張(RAG)評価のための日本語Q&Aデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JQaRA},
    }
    
    ```
    



#### `JQaRARerankingLite` { .model-copy }

JQaRA (Japanese Question Answering with Retrieval Augmentation) is a reranking dataset consisting of questions from JAQKET and corpus from Japanese Wikipedia. This is the lightweight version with a reduced corpus (172,897 documents) constructed using hard negatives from 5 high-performance models.

**Dataset:** [`mteb/JQaRARerankingLite`](https://huggingface.co/datasets/mteb/JQaRARerankingLite) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/hotchpotch/JQaRA)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb_lite,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
    and Kawahara, Daisuke},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
      title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
      year = {2025},
    }
    
    @misc{yuichi-tateno-2024-jqara,
      author = {Yuichi Tateno},
      title = {JQaRA: Japanese Question Answering with Retrieval Augmentation
    - 検索拡張(RAG)評価のための日本語Q&Aデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JQaRA},
    }
    
    ```
    



#### `JaCWIRReranking` { .model-copy }

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of 5000 question texts and approximately 500k web page titles and web page introductions or summaries (meta descriptions, etc.). The question texts are created based on one of the 500k web pages, and that data is used as a positive example for the question text.

**Dataset:** [`mteb/JaCWIRReranking`](https://huggingface.co/datasets/mteb/JaCWIRReranking) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | derived | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yuichi-tateno-2024-jacwir,
      author = {Yuichi Tateno},
      title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
    }
    
    ```
    



#### `JaCWIRRerankingLite` { .model-copy }

JaCWIR (Japanese Casual Web IR) is a dataset consisting of questions and webpage meta descriptions collected from Hatena Bookmark. This is the lightweight reranking version with a reduced corpus (188,033 documents) constructed using hard negatives from 5 high-performance models.

**Dataset:** [`mteb/JaCWIRRerankingLite`](https://huggingface.co/datasets/mteb/JaCWIRRerankingLite) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb_lite,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
    and Kawahara, Daisuke},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
      title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
      year = {2025},
    }
    
    @misc{yuichi-tateno-2024-jacwir,
      author = {Yuichi Tateno},
      title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
    }
    
    ```
    



#### `LocBenchRR` { .model-copy }

Software Issue Localization.

**Dataset:** [`mteb/LocBenchRR`](https://huggingface.co/datasets/mteb/LocBenchRR) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.09089)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{chen2025locagentgraphguidedllmagents,
      archiveprefix = {arXiv},
      author = {Zhaoling Chen and Xiangru Tang and Gangda Deng and Fang Wu and Jialong Wu and Zhiwei Jiang and Viktor Prasanna and Arman Cohan and Xingyao Wang},
      eprint = {2503.09089},
      primaryclass = {cs.SE},
      title = {LocAgent: Graph-Guided LLM Agents for Code Localization},
      url = {https://arxiv.org/abs/2503.09089},
      year = {2025},
    }
    
    ```
    



#### `MIRACLReranking` { .model-copy }

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`mteb/MIRACLReranking`](https://huggingface.co/datasets/mteb/MIRACLReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://project-miracl.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{10.1162/tacl_a_00595,
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      volume = {11},
      year = {2023},
    }
    ```
    



#### `MMarcoReranking` { .model-copy }

mMARCO is a multilingual version of the MS MARCO passage ranking dataset

**Dataset:** [`mteb/MMarcoReranking`](https://huggingface.co/datasets/mteb/MMarcoReranking) • **License:** not specified • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | machine-translated | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/abs-2108-13897,
      author = {Luiz Bonifacio and
    Israel Campiotti and
    Roberto de Alencar Lotufo and
    Rodrigo Frassetto Nogueira},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-2108-13897.bib},
      eprint = {2108.13897},
      eprinttype = {arXiv},
      journal = {CoRR},
      timestamp = {Mon, 20 Mar 2023 15:35:34 +0100},
      title = {mMARCO: {A} Multilingual Version of {MS} {MARCO} Passage Ranking Dataset},
      url = {https://arxiv.org/abs/2108.13897},
      volume = {abs/2108.13897},
      year = {2021},
    }
    
    ```
    



#### `MindSmallReranking` { .model-copy }

Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research

**Dataset:** [`mteb/MindSmallReranking`](https://huggingface.co/datasets/mteb/MindSmallReranking) • **License:** https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf • [Learn more →](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | expert-annotated | found | max_over_subqueries_map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wu-etal-2020-mind,
      address = {Online},
      author = {Wu, Fangzhao  and Qiao, Ying  and Chen, Jiun-Hung  and Wu, Chuhan  and Qi,
    Tao  and Lian, Jianxun  and Liu, Danyang  and Xie, Xing  and Gao, Jianfeng  and Wu, Winnie  and Zhou, Ming},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2020.acl-main.331},
      editor = {Jurafsky, Dan  and Chai, Joyce  and Schluter, Natalie  and Tetreault, Joel},
      month = jul,
      pages = {3597--3606},
      publisher = {Association for Computational Linguistics},
      title = {{MIND}: A Large-scale Dataset for News
    Recommendation},
      url = {https://aclanthology.org/2020.acl-main.331},
      year = {2020},
    }
    
    ```
    



#### `MultiLongDocReranking` { .model-copy }

Reranking version of MultiLongDocRetrieval (MLDR). MLDR is a Multilingual Long-Document Retrieval dataset built on Wikipedia, Wudao and mC4, covering 13 typologically diverse languages. Specifically, we sample lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.

**Dataset:** [`mteb/MultiLongDocReranking`](https://huggingface.co/datasets/mteb/MultiLongDocReranking) • **License:** mit • [Learn more →](https://huggingface.co/datasets/Shitao/MLDR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, deu, eng, fra, hin, ... (13) | Encyclopaedic, Fiction, Non-fiction, Web, Written | LM-generated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{bge-m3,
      archiveprefix = {arXiv},
      author = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      eprint = {2402.03216},
      primaryclass = {cs.CL},
      title = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
      year = {2024},
    }
    
    ```
    



#### `MultiSWEbenchRR` { .model-copy }

Multilingual Software Issue Localization.

**Dataset:** [`mteb/MultiSWEbenchRR`](https://huggingface.co/datasets/mteb/MultiSWEbenchRR) • **License:** mit • [Learn more →](https://multi-swe-bench.github.io/#/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{zan2025multiswebench,
      archiveprefix = {arXiv},
      author = {Daoguang Zan and Zhirong Huang and Wei Liu and Hanwu Chen and Linhao Zhang and Shulin Xin and Lu Chen and Qi Liu and Xiaojian Zhong and Aoyan Li and Siyao Liu and Yongsheng Xiao and Liangqiang Chen and Yuyu Zhang and Jing Su and Tianyu Liu and Rui Long and Kai Shen and Liang Xiang},
      eprint = {2504.02605},
      primaryclass = {cs.SE},
      title = {Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving},
      url = {https://arxiv.org/abs/2504.02605},
      year = {2025},
    }
    
    ```
    



#### `NamaaMrTydiReranking` { .model-copy }

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations. This dataset adapts the arabic test split for Reranking evaluation purposes by the addition of multiple (Hard) Negatives to each query and positive

**Dataset:** [`mteb/NamaaMrTydiReranking`](https://huggingface.co/datasets/mteb/NamaaMrTydiReranking) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/NAMAA-Space)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara | Encyclopaedic, Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{muennighoff2022mteb,
      author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\\i}c and Reimers, Nils},
      doi = {10.48550/ARXIV.2210.07316},
      journal = {arXiv preprint arXiv:2210.07316},
      publisher = {arXiv},
      title = {MTEB: Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2210.07316},
      year = {2022},
    }
    
    ```
    



#### `NevIR` { .model-copy }

Paired evaluation of real world negation in retrieval, with questions and passages. Since models generally prefer one passage over the other always, there are two questions that the model must get right to understand the negation (hence the `paired_accuracy` metric).

**Dataset:** [`orionweller/NevIR-mteb`](https://huggingface.co/datasets/orionweller/NevIR-mteb) • **License:** mit • [Learn more →](https://github.com/orionw/NevIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web | human-annotated | created | paired_accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Weller2023NevIRNI,
      author = {{Orion Weller and Dawn J Lawrie and Benjamin Van Durme}},
      booktitle = {{Conference of the European Chapter of the Association for Computational Linguistics}},
      title = {{NevIR: Negation in Neural Information Retrieval}},
      url = {{https://api.semanticscholar.org/CorpusID:258676146}},
      year = {{2023}},
    }
    
    ```
    



#### `RuBQReranking` { .model-copy }

Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.

**Dataset:** [`mteb/RuBQReranking`](https://huggingface.co/datasets/mteb/RuBQReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | Encyclopaedic, Written | human-annotated | created | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{RuBQ2021,
      author = {Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
      booktitle = {ESWC},
      pages = {532--547},
      title = {RuBQ 2.0: An Innovated Russian Question Answering Dataset},
      year = {2021},
    }
    
    ```
    



#### `SWEPolyBenchRR` { .model-copy }

Multilingual Software Issue Localization.

**Dataset:** [`mteb/SWEPolyBenchRR`](https://huggingface.co/datasets/mteb/SWEPolyBenchRR) • **License:** mit • [Learn more →](https://amazon-science.github.io/SWE-PolyBench/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{rashid2025swepolybenchmultilanguagebenchmarkrepository,
      archiveprefix = {arXiv},
      author = {Muhammad Shihab Rashid and Christian Bock and Yuan Zhuang and Alexander Buchholz and Tim Esler and Simon Valentin and Luca Franceschi and Martin Wistuba and Prabhu Teja Sivaprasad and Woo Jung Kim and Anoop Deoras and Giovanni Zappella and Laurent Callot},
      eprint = {2504.08703},
      primaryclass = {cs.SE},
      title = {SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents},
      url = {https://arxiv.org/abs/2504.08703},
      year = {2025},
    }
    
    ```
    



#### `SWEbenchLiteRR` { .model-copy }

Software Issue Localization.

**Dataset:** [`mteb/SWEbenchLiteRR`](https://huggingface.co/datasets/mteb/SWEbenchLiteRR) • **License:** mit • [Learn more →](https://www.swebench.com/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jimenez2024swebenchlanguagemodelsresolve,
      archiveprefix = {arXiv},
      author = {Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
      eprint = {2310.06770},
      primaryclass = {cs.CL},
      title = {SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
      url = {https://arxiv.org/abs/2310.06770},
      year = {2024},
    }
    
    ```
    



#### `SWEbenchMultilingualRR` { .model-copy }

Multilingual Software Issue Localization.

**Dataset:** [`mteb/SWEbenchMultilingualRR`](https://huggingface.co/datasets/mteb/SWEbenchMultilingualRR) • **License:** mit • [Learn more →](https://www.swebench.com/multilingual.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yang2025swesmith,
      archiveprefix = {arXiv},
      author = {John Yang and Kilian Lieret and Carlos E. Jimenez and Alexander Wettig and Kabir Khandpur and Yanzhe Zhang and Binyuan Hui and Ofir Press and Ludwig Schmidt and Diyi Yang},
      eprint = {2504.21798},
      primaryclass = {cs.SE},
      title = {SWE-smith: Scaling Data for Software Engineering Agents},
      url = {https://arxiv.org/abs/2504.21798},
      year = {2025},
    }
    
    ```
    



#### `SWEbenchVerifiedRR` { .model-copy }

Software Issue Localization for SWE-bench Verified

**Dataset:** [`mteb/SWEbenchVerifiedRR`](https://huggingface.co/datasets/mteb/SWEbenchVerifiedRR) • **License:** mit • [Learn more →](https://openai.com/index/introducing-swe-bench-verified/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{openai2024swebenchverified,
      author = {OpenAI},
      title = {Introducing swe-bench verified},
      url = {https://openai.com/index/introducing-swe-bench-verified/},
      year = {2024},
    }
    
    ```
    



#### `SciDocsRR` { .model-copy }

Ranking of related scientific papers based on their title.

**Dataset:** [`mteb/SciDocsRR`](https://huggingface.co/datasets/mteb/SciDocsRR) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | derived | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### `SciDocsRR-VN` { .model-copy }

A translated dataset from Ranking of related scientific papers based on their title. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`mteb/SciDocsRR-VN`](https://huggingface.co/datasets/mteb/SciDocsRR-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `StackOverflowDupQuestions` { .model-copy }

Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python

**Dataset:** [`mteb/StackOverflowDupQuestions`](https://huggingface.co/datasets/mteb/StackOverflowDupQuestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, Programming, Written | derived | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @article{Liu2018LinkSOAD,
      author = {Xueqing Liu and Chi Wang and Yue Leng and ChengXiang Zhai},
      journal = {Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering},
      title = {LinkSO: a dataset for learning to retrieve similar question answer pairs on software development forums},
      url = {https://api.semanticscholar.org/CorpusID:53111679},
      year = {2018},
    }
    
    ```
    



#### `StackOverflowDupQuestions-VN` { .model-copy }

A translated dataset from Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`mteb/StackOverflowDupQuestions-VN`](https://huggingface.co/datasets/mteb/StackOverflowDupQuestions-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `SyntecReranking` { .model-copy }

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`mteb/SyntecReranking`](https://huggingface.co/datasets/mteb/SyntecReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Legal, Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{ciancone2024extending,
      archiveprefix = {arXiv},
      author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      eprint = {2405.20468},
      primaryclass = {cs.CL},
      title = {Extending the Massive Text Embedding Benchmark to French},
      year = {2024},
    }
    
    ```
    



#### `T2Reranking` { .model-copy }

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Reranking`](https://huggingface.co/datasets/mteb/T2Reranking) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2304.03679)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{xie2023t2ranking,
      archiveprefix = {arXiv},
      author = {Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
      eprint = {2304.03679},
      primaryclass = {cs.IR},
      title = {T2Ranking: A large-scale Chinese Benchmark for Passage Ranking},
      year = {2023},
    }
    
    ```
    



#### `VoyageMMarcoReranking` { .model-copy }

a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite

**Dataset:** [`mteb/VoyageMMarcoReranking`](https://huggingface.co/datasets/mteb/VoyageMMarcoReranking) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2312.16144)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Non-fiction, Written | derived | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{clavié2023jacolbert,
      archiveprefix = {arXiv},
      author = {Benjamin Clavié},
      eprint = {2312.16144},
      title = {JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report},
      year = {2023},
    }
    
    ```
    



#### `WebLINXCandidatesReranking` { .model-copy }

WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.

**Dataset:** [`mteb/WebLINXCandidatesReranking`](https://huggingface.co/datasets/mteb/WebLINXCandidatesReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/weblinx)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Web, Written | expert-annotated | created | mrr_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lù2024weblinx,
      archiveprefix = {arXiv},
      author = {Xing Han Lù and Zdeněk Kasner and Siva Reddy},
      eprint = {2402.05930},
      primaryclass = {cs.CL},
      title = {WebLINX: Real-World Website Navigation with Multi-Turn Dialogue},
      year = {2024},
    }
    
    ```
    



#### `WikipediaRerankingMultilingual` { .model-copy }

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRerankingMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRerankingMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-multilingual)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ben, bul, ces, dan, deu, ... (18) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @online{wikidump2024,
      author = {Wikimedia Foundation},
      title = {Wikimedia Downloads},
      url = {https://dumps.wikimedia.org},
    }
    
    ```
    



#### `XGlueWPRReranking` { .model-copy }

XGLUE is a new benchmark dataset to evaluate the performance of cross-lingual pre-trained models with respect to cross-lingual natural language understanding and generation. XGLUE is composed of 11 tasks spans 19 languages.

**Dataset:** [`mteb/XGlueWPRReranking`](https://huggingface.co/datasets/mteb/XGlueWPRReranking) • **License:** http://hdl.handle.net/11234/1-3105 • [Learn more →](https://github.com/microsoft/XGLUE)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu, eng, fra, ita, por, ... (7) | Written | human-annotated | found | map_at_1000 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{11234/1-3105,
      author = {Zeman, Daniel and Nivre, Joakim and Abrams, Mitchell and Aepli, No{\"e}mi and Agi{\'c}, {\v Z}eljko and Ahrenberg, Lars and Aleksandravi{\v c}i{\=u}t{\.e}, Gabriel{\.e} and Antonsen, Lene and Aplonova, Katya and Aranzabe, Maria Jesus and Arutie, Gashaw and Asahara, Masayuki and Ateyah, Luma and Attia, Mohammed and Atutxa, Aitziber and Augustinus, Liesbeth and Badmaeva, Elena and Ballesteros, Miguel and Banerjee, Esha and Bank, Sebastian and Barbu Mititelu, Verginica and Basmov, Victoria and Batchelor, Colin and Bauer, John and Bellato, Sandra and Bengoetxea, Kepa and Berzak, Yevgeni and Bhat, Irshad Ahmad and Bhat, Riyaz Ahmad and Biagetti, Erica and Bick, Eckhard and Bielinskien{\.e}, Agn{\.e} and Blokland, Rogier and Bobicev, Victoria and Boizou, Lo{\"{\i}}c and Borges V{\"o}lker, Emanuel and B{\"o}rstell, Carl and Bosco, Cristina and Bouma, Gosse and Bowman, Sam and Boyd, Adriane and Brokait{\.e}, Kristina and Burchardt, Aljoscha and Candito, Marie and Caron, Bernard and Caron, Gauthier and Cavalcanti, Tatiana and Cebiro{\u g}lu Eryi{\u g}it, G{\"u}l{\c s}en and Cecchini, Flavio Massimiliano and Celano, Giuseppe G. A. and {\v C}{\'e}pl{\"o}, Slavom{\'{\i}}r and Cetin, Savas and Chalub, Fabricio and Choi, Jinho and Cho, Yongseok and Chun, Jayeol and Cignarella, Alessandra T. and Cinkov{\'a}, Silvie and Collomb, Aur{\'e}lie and {\c C}{\"o}ltekin, {\c C}a{\u g}r{\i} and Connor, Miriam and Courtin, Marine and Davidson, Elizabeth and de Marneffe, Marie-Catherine and de Paiva, Valeria and de Souza, Elvis and Diaz de Ilarraza, Arantza and Dickerson, Carly and Dione, Bamba and Dirix, Peter and Dobrovoljc, Kaja and Dozat, Timothy and Droganova, Kira and Dwivedi, Puneet and Eckhoff, Hanne and Eli, Marhaba and Elkahky, Ali and Ephrem, Binyam and Erina, Olga and Erjavec, Toma{\v z} and Etienne, Aline and Evelyn, Wograine and Farkas, Rich{\'a}rd and Fernandez Alcalde, Hector and Foster, Jennifer and Freitas, Cl{\'a}udia and Fujita, Kazunori and Gajdo{\v s}ov{\'a}, Katar{\'{\i}}na and Galbraith, Daniel and Garcia, Marcos and G{\"a}rdenfors, Moa and Garza, Sebastian and Gerdes, Kim and Ginter, Filip and Goenaga, Iakes and Gojenola, Koldo and G{\"o}k{\i}rmak, Memduh and Goldberg, Yoav and G{\'o}mez Guinovart, Xavier and Gonz{\'a}lez Saavedra, Berta and Grici{\=u}t{\.e}, Bernadeta and Grioni, Matias and Gr{\=u}z{\={\i}}tis, Normunds and Guillaume, Bruno and Guillot-Barbance, C{\'e}line and Habash, Nizar and Haji{\v c}, Jan and Haji{\v c} jr., Jan and H{\"a}m{\"a}l{\"a}inen, Mika and H{\`a} M{\~y}, Linh and Han, Na-Rae and Harris, Kim and Haug, Dag and Heinecke, Johannes and Hennig, Felix and Hladk{\'a}, Barbora and Hlav{\'a}{\v c}ov{\'a}, Jaroslava and Hociung, Florinel and Hohle, Petter and Hwang, Jena and Ikeda, Takumi and Ion, Radu and Irimia, Elena and Ishola, {\d O}l{\'a}j{\'{\i}}d{\'e} and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Johannsen, Anders and J{\o}rgensen, Fredrik and Juutinen, Markus and Ka{\c s}{\i}kara, H{\"u}ner and Kaasen, Andre and Kabaeva, Nadezhda and Kahane, Sylvain and Kanayama, Hiroshi and Kanerva, Jenna and Katz, Boris and Kayadelen, Tolga and Kenney, Jessica and Kettnerov{\'a}, V{\'a}clava and Kirchner, Jesse and Klementieva, Elena and K{\"o}hn, Arne and Kopacewicz, Kamil and Kotsyba, Natalia and Kovalevskait{\.e}, Jolanta and Krek, Simon and Kwak, Sookyoung and Laippala, Veronika and Lambertino, Lorenzo and Lam, Lucia and Lando, Tatiana and Larasati, Septina Dian and Lavrentiev, Alexei and Lee, John and L{\^e} H{\`{\^o}}ng, Phương and Lenci, Alessandro and Lertpradit, Saran and Leung, Herman and Li, Cheuk Ying and Li, Josie and Li, Keying and Lim, {KyungTae} and Liovina, Maria and Li, Yuan and Ljube{\v s}i{\'c}, Nikola and Loginova, Olga and Lyashevskaya, Olga and Lynn, Teresa and Macketanz, Vivien and Makazhanov, Aibek and Mandl, Michael and Manning, Christopher and Manurung, Ruli and M{\u a}r{\u a}nduc, C{\u a}t{\u a}lina and Mare{\v c}ek, David and Marheinecke, Katrin and Mart{\'{\i}}nez Alonso, H{\'e}ctor and Martins, Andr{\'e} and Ma{\v s}ek, Jan and Matsumoto, Yuji and {McDonald}, Ryan and {McGuinness}, Sarah and Mendon{\c c}a, Gustavo and Miekka, Niko and Misirpashayeva, Margarita and Missil{\"a}, Anna and Mititelu, C{\u a}t{\u a}lin and Mitrofan, Maria and Miyao, Yusuke and Montemagni, Simonetta and More, Amir and Moreno Romero, Laura and Mori, Keiko Sophie and Morioka, Tomohiko and Mori, Shinsuke and Moro, Shigeki and Mortensen, Bjartur and Moskalevskyi, Bohdan and Muischnek, Kadri and Munro, Robert and Murawaki, Yugo and M{\"u}{\"u}risep, Kaili and Nainwani, Pinkey and Navarro Hor{\~n}iacek, Juan Ignacio and Nedoluzhko, Anna and Ne{\v s}pore-B{\=e}rzkalne, Gunta and Nguy{\~{\^e}}n Th{\d i}, Lương and Nguy{\~{\^e}}n Th{\d i} Minh, Huy{\`{\^e}}n and Nikaido, Yoshihiro and Nikolaev, Vitaly and Nitisaroj, Rattima and Nurmi, Hanna and Ojala, Stina and Ojha, Atul Kr. and Ol{\'u}{\`o}kun, Ad{\'e}day{\d o}̀ and Omura, Mai and Osenova, Petya and {\"O}stling, Robert and {\O}vrelid, Lilja and Partanen, Niko and Pascual, Elena and Passarotti, Marco and Patejuk, Agnieszka and Paulino-Passos, Guilherme and Peljak-{\L}api{\'n}ska, Angelika and Peng, Siyao and Perez, Cenel-Augusto and Perrier, Guy and Petrova, Daria and Petrov, Slav and Phelan, Jason and Piitulainen, Jussi and Pirinen, Tommi A and Pitler, Emily and Plank, Barbara and Poibeau, Thierry and Ponomareva, Larisa and Popel, Martin and Pretkalni{\c n}a, Lauma and Pr{\'e}vost, Sophie and Prokopidis, Prokopis and Przepi{\'o}rkowski, Adam and Puolakainen, Tiina and Pyysalo, Sampo and Qi, Peng and R{\"a}{\"a}bis, Andriela and Rademaker, Alexandre and Ramasamy, Loganathan and Rama, Taraka and Ramisch, Carlos and Ravishankar, Vinit and Real, Livy and Reddy, Siva and Rehm, Georg and Riabov, Ivan and Rie{\ss}ler, Michael and Rimkut{\.e}, Erika and Rinaldi, Larissa and Rituma, Laura and Rocha, Luisa and Romanenko, Mykhailo and Rosa, Rudolf and Rovati, Davide and Roșca, Valentin and Rudina, Olga and Rueter, Jack and Sadde, Shoval and Sagot, Beno{\^{\i}}t and Saleh, Shadi and Salomoni, Alessio and Samard{\v z}i{\'c}, Tanja and Samson, Stephanie and Sanguinetti, Manuela and S{\"a}rg, Dage and Saul{\={\i}}te, Baiba and Sawanakunanon, Yanin and Schneider, Nathan and Schuster, Sebastian and Seddah, Djam{\'e} and Seeker, Wolfgang and Seraji, Mojgan and Shen, Mo and Shimada, Atsuko and Shirasu, Hiroyuki and Shohibussirri, Muh and Sichinava, Dmitry and Silveira, Aline and Silveira, Natalia and Simi, Maria and Simionescu, Radu and Simk{\'o}, Katalin and {\v S}imkov{\'a}, M{\'a}ria and Simov, Kiril and Smith, Aaron and Soares-Bastos, Isabela and Spadine, Carolyn and Stella, Antonio and Straka, Milan and Strnadov{\'a}, Jana and Suhr, Alane and Sulubacak, Umut and Suzuki, Shingo and Sz{\'a}nt{\'o}, Zsolt and Taji, Dima and Takahashi, Yuta and Tamburini, Fabio and Tanaka, Takaaki and Tellier, Isabelle and Thomas, Guillaume and Torga, Liisi and Trosterud, Trond and Trukhina, Anna and Tsarfaty, Reut and Tyers, Francis and Uematsu, Sumire and Ure{\v s}ov{\'a}, Zde{\v n}ka and Uria, Larraitz and Uszkoreit, Hans and Utka, Andrius and Vajjala, Sowmya and van Niekerk, Daniel and van Noord, Gertjan and Varga, Viktor and Villemonte de la Clergerie, Eric and Vincze, Veronika and Wallin, Lars and Walsh, Abigail and Wang, Jing Xian and Washington, Jonathan North and Wendt, Maximilan and Williams, Seyi and Wir{\'e}n, Mats and Wittern, Christian and Woldemariam, Tsegay and Wong, Tak-sum and Wr{\'o}blewska, Alina and Yako, Mary and Yamazaki, Naoki and Yan, Chunxiao and Yasuoka, Koichi and Yavrumyan, Marat M. and Yu, Zhuoran and {\v Z}abokrtsk{\'y}, Zden{\v e}k and Zeldes, Amir and Zhang, Manying and Zhu, Hanzhi},
      copyright = {Licence Universal Dependencies v2.5},
      note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
      title = {Universal Dependencies 2.5},
      url = {http://hdl.handle.net/11234/1-3105},
      year = {2019},
    }
    
    @inproceedings{conneau2018xnli,
      author = {Conneau, Alexis
    and Rinott, Ruty
    and Lample, Guillaume
    and Williams, Adina
    and Bowman, Samuel R.
    and Schwenk, Holger
    and Stoyanov, Veselin},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods
    in Natural Language Processing},
      location = {Brussels, Belgium},
      publisher = {Association for Computational Linguistics},
      title = {XNLI: Evaluating Cross-lingual Sentence Representations},
      year = {2018},
    }
    
    @article{lewis2019mlqa,
      author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
      eid = {arXiv: 1910.07475},
      journal = {arXiv preprint arXiv:1910.07475},
      title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
      year = {2019},
    }
    
    @article{Liang2020XGLUEAN,
      author = {Yaobo Liang and Nan Duan and Yeyun Gong and Ning Wu and Fenfei Guo and Weizhen Qi and Ming Gong and Linjun Shou and Daxin Jiang and Guihong Cao and Xiaodong Fan and Ruofei Zhang and Rahul Agrawal and Edward Cui and Sining Wei and Taroon Bharti and Ying Qiao and Jiun-Hung Chen and Winnie Wu and Shuguang Liu and Fan Yang and Daniel Campos and Rangan Majumder and Ming Zhou},
      journal = {arXiv},
      title = {XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation},
      volume = {abs/2004.01401},
      year = {2020},
    }
    
    @article{Sang2002IntroductionTT,
      author = {Erik F. Tjong Kim Sang},
      journal = {ArXiv},
      title = {Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition},
      volume = {cs.CL/0209010},
      year = {2002},
    }
    
    @article{Sang2003IntroductionTT,
      author = {Erik F. Tjong Kim Sang and Fien De Meulder},
      journal = {ArXiv},
      title = {Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition},
      volume = {cs.CL/0306050},
      year = {2003},
    }
    
    @misc{yang2019pawsx,
      archiveprefix = {arXiv},
      author = {Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
      eprint = {1908.11828},
      primaryclass = {cs.CL},
      title = {PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
      year = {2019},
    }
    
    ```




## Retrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 431

#### `AILACasedocs` { .model-copy }

The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.

**Dataset:** [`mteb/AILA_casedocs`](https://huggingface.co/datasets/mteb/AILA_casedocs) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{paheli_bhattacharya_2020_4063986,
      author = {Paheli Bhattacharya and
    Kripabandhu Ghosh and
    Saptarshi Ghosh and
    Arindam Pal and
    Parth Mehta and
    Arnab Bhattacharya and
    Prasenjit Majumder},
      doi = {10.5281/zenodo.4063986},
      month = oct,
      publisher = {Zenodo},
      title = {AILA 2019 Precedent \& Statute Retrieval Task},
      url = {https://doi.org/10.5281/zenodo.4063986},
      year = {2020},
    }
    
    ```
    



#### `AILAStatutes` { .model-copy }

This dataset is structured for the task of identifying the most relevant statutes for a given situation.

**Dataset:** [`mteb/AILA_statutes`](https://huggingface.co/datasets/mteb/AILA_statutes) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{paheli_bhattacharya_2020_4063986,
      author = {Paheli Bhattacharya and
    Kripabandhu Ghosh and
    Saptarshi Ghosh and
    Arindam Pal and
    Parth Mehta and
    Arnab Bhattacharya and
    Prasenjit Majumder},
      doi = {10.5281/zenodo.4063986},
      month = oct,
      publisher = {Zenodo},
      title = {AILA 2019 Precedent \& Statute Retrieval Task},
      url = {https://doi.org/10.5281/zenodo.4063986},
      year = {2020},
    }
    
    ```
    



#### `ARCChallenge` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.

**Dataset:** [`mteb/ARCChallenge`](https://huggingface.co/datasets/mteb/ARCChallenge) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/arc)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{clark2018think,
      author = {Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
      journal = {arXiv preprint arXiv:1803.05457},
      title = {Think you have solved question answering? try arc, the ai2 reasoning challenge},
      year = {2018},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `AlloprofRetrieval` { .model-copy }

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`mteb/AlloprofRetrieval`](https://huggingface.co/datasets/mteb/AlloprofRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lef23,
      author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
      copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
      doi = {10.48550/ARXIV.2302.07738},
      keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      publisher = {arXiv},
      title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
      url = {https://arxiv.org/abs/2302.07738},
      year = {2023},
    }
    
    ```
    



#### `AlphaNLI` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on AlphaNLI.

**Dataset:** [`mteb/AlphaNLI`](https://huggingface.co/datasets/mteb/AlphaNLI) • **License:** cc-by-nc-4.0 • [Learn more →](https://leaderboard.allenai.org/anli/submissions/get-started)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{bhagavatula2019abductive,
      author = {Bhagavatula, Chandra and Bras, Ronan Le and Malaviya, Chaitanya and Sakaguchi, Keisuke and Holtzman, Ari and Rashkin, Hannah and Downey, Doug and Yih, Scott Wen-tau and Choi, Yejin},
      journal = {arXiv preprint arXiv:1908.05739},
      title = {Abductive commonsense reasoning},
      year = {2019},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `AppsRetrieval` { .model-copy }

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.09938)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{hendrycksapps2021,
      author = {Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
      journal = {NeurIPS},
      title = {Measuring Coding Challenge Competence With APPS},
      year = {2021},
    }
    
    ```
    



#### `ArguAna` { .model-copy }

ArguAna: Retrieval of the Best Counterargument without Prior Topic Knowledge

**Dataset:** [`mteb/arguana`](https://huggingface.co/datasets/mteb/arguana) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Social, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wachsmuth2018retrieval,
      author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
      booktitle = {ACL},
      title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
      year = {2018},
    }
    
    ```
    



#### `ArguAna-Fa` { .model-copy }

ArguAna-Fa

**Dataset:** [`MCINext/arguana-fa`](https://huggingface.co/datasets/MCINext/arguana-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/arguana-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Blog | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `ArguAna-Fa.v2` { .model-copy }

ArguAna-Fa.v2

**Dataset:** [`MCINext/arguana-fa-v2`](https://huggingface.co/datasets/MCINext/arguana-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/arguana-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Blog | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `ArguAna-NL` { .model-copy }

ArguAna involves the task of retrieval of the best counterargument to an argument. ArguAna-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-arguana`](https://huggingface.co/datasets/clips/beir-nl-arguana) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-arguana)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `ArguAna-NL.v2` { .model-copy }

ArguAna involves the task of retrieval of the best counterargument to an argument. ArguAna-NL is a Dutch translation. This version adds a Dutch prompt to the dataset.

**Dataset:** [`clips/beir-nl-arguana`](https://huggingface.co/datasets/clips/beir-nl-arguana) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-arguana)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `ArguAna-PL` { .model-copy }

ArguAna-PL

**Dataset:** [`mteb/ArguAna-PL`](https://huggingface.co/datasets/mteb/ArguAna-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clarin-knext/arguana-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Medical, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `ArguAna-VN` { .model-copy }

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/arguana-vn`](https://huggingface.co/datasets/GreenNode/arguana-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Medical, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `AutoRAGRetrieval` { .model-copy }

This dataset enables the evaluation of Korean RAG performance across various domains—finance, public sector, healthcare, legal, and commerce—by providing publicly accessible documents, questions, and answers.

**Dataset:** [`yjoonjang/markers_bm`](https://huggingface.co/datasets/yjoonjang/markers_bm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2410.20878)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kor | Financial, Government, Legal, Medical, Social | human-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{kim2024autoragautomatedframeworkoptimization,
      archiveprefix = {arXiv},
      author = {Dongkyu Kim and Byoungwook Kim and Donggeon Han and Matouš Eibich},
      eprint = {2410.20878},
      primaryclass = {cs.CL},
      title = {AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline},
      url = {https://arxiv.org/abs/2410.20878},
      year = {2024},
    }
    
    ```
    



#### `BIRCO-ArguAna` { .model-copy }

Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument that directly refutes the query’s stance.

**Dataset:** [`mteb/BIRCO-ArguAna-Test`](https://huggingface.co/datasets/mteb/BIRCO-ArguAna-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### `BIRCO-ClinicalTrial` { .model-copy }

Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant.

**Dataset:** [`mteb/BIRCO-ClinicalTrial-Test`](https://huggingface.co/datasets/mteb/BIRCO-ClinicalTrial-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### `BIRCO-DorisMae` { .model-copy }

Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries that are complex research questions from computer scientists. Each query has a candidate pool of approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant).

**Dataset:** [`mteb/BIRCO-DorisMae-Test`](https://huggingface.co/datasets/mteb/BIRCO-DorisMae-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### `BIRCO-Relic` { .model-copy }

Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries which are excerpts from literary analyses with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. The objective is to retrieve the passage that best completes the literary analysis.

**Dataset:** [`mteb/BIRCO-Relic-Test`](https://huggingface.co/datasets/mteb/BIRCO-Relic-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### `BIRCO-WTB` { .model-copy }

Retrieval task using the WhatsThatBook dataset from BIRCO. This dataset contains 100 queries where each query is an ambiguous description of a book. Each query has a candidate pool of 50 book descriptions. The objective is to retrieve the correct book description.

**Dataset:** [`mteb/BIRCO-WTB-Test`](https://huggingface.co/datasets/mteb/BIRCO-WTB-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### `BSARDRetrieval` { .model-copy }

The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.

**Dataset:** [`mteb/BSARDRetrieval`](https://huggingface.co/datasets/mteb/BSARDRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Legal, Spoken | expert-annotated | found | recall_at_100 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{louis2022statutory,
      address = {Dublin, Ireland},
      author = {Louis, Antoine and Spanakis, Gerasimos},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2022.acl-long.468},
      month = may,
      pages = {6789–6803},
      publisher = {Association for Computational Linguistics},
      title = {A Statutory Article Retrieval Dataset in French},
      url = {https://aclanthology.org/2022.acl-long.468/},
      year = {2022},
    }
    
    ```
    



#### `BSARDRetrieval.v2` { .model-copy }

BSARD is a French native dataset for legal information retrieval. BSARDRetrieval.v2 covers multi-article queries, fixing issues (#2906) with the previous data loading. 

**Dataset:** [`mteb/BSARDRetrieval.v2`](https://huggingface.co/datasets/mteb/BSARDRetrieval.v2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Legal, Spoken | expert-annotated | found | recall_at_100 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{louis2022statutory,
      address = {Dublin, Ireland},
      author = {Louis, Antoine and Spanakis, Gerasimos},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2022.acl-long.468},
      month = may,
      pages = {6789–6803},
      publisher = {Association for Computational Linguistics},
      title = {A Statutory Article Retrieval Dataset in French},
      url = {https://aclanthology.org/2022.acl-long.468/},
      year = {2022},
    }
    
    ```
    



#### `BarExamQA` { .model-copy }

A benchmark for retrieving legal provisions that answer US bar exam questions.

**Dataset:** [`isaacus/mteb-barexam-qa`](https://huggingface.co/datasets/isaacus/mteb-barexam-qa) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/reglab/barexam_qa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Legal | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Zheng_2025,
      author = {Zheng, Lucia and Guha, Neel and Arifov, Javokhir and Zhang, Sarah and Skreta, Michal and Manning, Christopher D. and Henderson, Peter and Ho, Daniel E.},
      booktitle = {Proceedings of the Symposium on Computer Science and Law on ZZZ},
      collection = {CSLAW ’25},
      doi = {10.1145/3709025.3712219},
      eprint = {2505.03970},
      month = mar,
      pages = {169–193},
      publisher = {ACM},
      series = {CSLAW ’25},
      title = {A Reasoning-Focused Legal Retrieval Benchmark},
      url = {http://dx.doi.org/10.1145/3709025.3712219},
      year = {2025},
    }
    
    ```
    



#### `BelebeleRetrieval` { .model-copy }

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants (including 115 distinct languages and their scripts)

**Dataset:** [`mteb/belebele`](https://huggingface.co/datasets/mteb/belebele) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2308.16884)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | acm, afr, als, amh, apc, ... (115) | News, Web, Written | expert-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{bandarkar2023belebele,
      author = {Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      journal = {arXiv preprint arXiv:2308.16884},
      title = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
      year = {2023},
    }
    
    ```
    



#### `BillSumCA` { .model-copy }

A benchmark for retrieving Californian bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumCA`](https://huggingface.co/datasets/isaacus/mteb-BillSumCA) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Government, Legal | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Eidelman_2019,
      author = {Eidelman, Vladimir},
      booktitle = {Proceedings of the 2nd Workshop on New Frontiers in Summarization},
      doi = {10.18653/v1/d19-5406},
      pages = {48–56},
      publisher = {Association for Computational Linguistics},
      title = {BillSum: A Corpus for Automatic Summarization of US Legislation},
      url = {http://dx.doi.org/10.18653/v1/D19-5406},
      year = {2019},
    }
    
    ```
    



#### `BillSumUS` { .model-copy }

A benchmark for retrieving US federal bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumUS`](https://huggingface.co/datasets/isaacus/mteb-BillSumUS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Government, Legal | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Eidelman_2019,
      author = {Eidelman, Vladimir},
      booktitle = {Proceedings of the 2nd Workshop on New Frontiers in Summarization},
      doi = {10.18653/v1/d19-5406},
      pages = {48–56},
      publisher = {Association for Computational Linguistics},
      title = {BillSum: A Corpus for Automatic Summarization of US Legislation},
      url = {http://dx.doi.org/10.18653/v1/D19-5406},
      year = {2019},
    }
    
    ```
    



#### `BrightAopsRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar Math Olympiad problems from Art of Problem Solving.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightBiologyLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightBiologyRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightEarthScienceLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightEarthScienceRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightEconomicsLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightEconomicsRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightLeetcodeRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar algorithmic problems based on shared solution techniques.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightLongRetrieval` { .model-copy }

Bright retrieval dataset with long documents.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightPonyLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightPonyRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightPsychologyLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightPsychologyRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightRetrieval` { .model-copy }

BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightRoboticsLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightRoboticsRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightStackoverflowLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightStackoverflowRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightSustainableLivingLongRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers with long documents.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | recall_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightSustainableLivingRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightTheoremQAQuestionsRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions from ProofWiki given questions rephrased as real-world scenarios.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BrightTheoremQATheoremsRetrieval` { .model-copy }

Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions and proofs from ProofWiki.

**Dataset:** [`mteb/BRIGHT`](https://huggingface.co/datasets/mteb/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }
    
    ```
    



#### `BuiltBenchRetrieval` { .model-copy }

Retrieval of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchRetrieval`](https://huggingface.co/datasets/mteb/BuiltBenchRetrieval) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Engineering, Written | derived | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    
    ```
    



#### `COIRCodeSearchNetRetrieval` { .model-copy }

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{husain2019codesearchnet,
      author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
      journal = {arXiv preprint arXiv:1909.09436},
      title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
      year = {2019},
    }
    
    ```
    



#### `CQADupstack-Android-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Android-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Android-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-android-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Non-fiction, Programming, Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-English-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-English-PL`](https://huggingface.co/datasets/mteb/CQADupstack-English-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-english-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Gaming-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gaming-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gaming-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gaming-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Gis-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gis-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gis-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gis-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Non-fiction, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Mathematica-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Mathematica-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Mathematica-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-mathematica-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Non-fiction, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    


??? info "Tasks"

    | name                                                                  | type      | modalities   | languages   |
    |:----------------------------------------------------------------------|:----------|:-------------|:------------|
    | [CQADupstackAndroid-NL](./retrieval.md#cqadupstackandroid-nl)         | Retrieval | text         | nld         |
    | [CQADupstackEnglish-NL](./retrieval.md#cqadupstackenglish-nl)         | Retrieval | text         | nld         |
    | [CQADupstackGaming-NL](./retrieval.md#cqadupstackgaming-nl)           | Retrieval | text         | nld         |
    | [CQADupstackGis-NL](./retrieval.md#cqadupstackgis-nl)                 | Retrieval | text         | nld         |
    | [CQADupstackMathematica-NL](./retrieval.md#cqadupstackmathematica-nl) | Retrieval | text         | nld         |
    | [CQADupstackPhysics-NL](./retrieval.md#cqadupstackphysics-nl)         | Retrieval | text         | nld         |
    | [CQADupstackProgrammers-NL](./retrieval.md#cqadupstackprogrammers-nl) | Retrieval | text         | nld         |
    | [CQADupstackStats-NL](./retrieval.md#cqadupstackstats-nl)             | Retrieval | text         | nld         |
    | [CQADupstackTex-NL](./retrieval.md#cqadupstacktex-nl)                 | Retrieval | text         | nld         |
    | [CQADupstackUnix-NL](./retrieval.md#cqadupstackunix-nl)               | Retrieval | text         | nld         |
    | [CQADupstackWebmasters-NL](./retrieval.md#cqadupstackwebmasters-nl)   | Retrieval | text         | nld         |
    | [CQADupstackWordpress-NL](./retrieval.md#cqadupstackwordpress-nl)     | Retrieval | text         | nld         |


#### `CQADupstack-Physics-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Physics-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Physics-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-physics-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Non-fiction, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Programmers-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Programmers-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Programmers-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-programmers-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Non-fiction, Programming, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Stats-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Stats-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Stats-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-stats-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Non-fiction, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Tex-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Tex-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Tex-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-tex-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Non-fiction, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Unix-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Unix-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Unix-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-unix-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Programming, Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Webmasters-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Webmasters-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Webmasters-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-webmasters-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstack-Wordpress-PL` { .model-copy }

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Wordpress-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Wordpress-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-wordpress-pl)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Programming, Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackAndroid-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackAndroid-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-android-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-android-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Non-fiction, Programming, Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackAndroidRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/CQADupstackAndroidRetrieval`](https://huggingface.co/datasets/mteb/CQADupstackAndroidRetrieval) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Programming, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackAndroidRetrieval-Fa` { .model-copy }

CQADupstackAndroidRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-android-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackEnglish-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackEnglishRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-english`](https://huggingface.co/datasets/mteb/cqadupstack-english) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackEnglishRetrieval-Fa` { .model-copy }

CQADupstackEnglishRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-english-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackGaming-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackGamingRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gaming`](https://huggingface.co/datasets/mteb/cqadupstack-gaming) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackGamingRetrieval-Fa` { .model-copy }

CQADupstackGamingRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gaming-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackGis-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackGis-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-gis-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-gis-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Non-fiction, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackGisRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gis`](https://huggingface.co/datasets/mteb/cqadupstack-gis) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackGisRetrieval-Fa` { .model-copy }

CQADupstackGisRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gis-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackMathematica-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackMathematica-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-mathematica-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-mathematica-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackMathematicaRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-mathematica`](https://huggingface.co/datasets/mteb/cqadupstack-mathematica) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackMathematicaRetrieval-Fa` { .model-copy }

CQADupstackMathematicaRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-mathematica-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackPhysics-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackPhysics-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-physics-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-physics-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackPhysicsRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-physics`](https://huggingface.co/datasets/mteb/cqadupstack-physics) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackPhysicsRetrieval-Fa` { .model-copy }

CQADupstackPhysicsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-physics-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackProgrammers-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackProgrammers-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-programmers-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-programmers-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Non-fiction, Programming, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackProgrammersRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-programmers`](https://huggingface.co/datasets/mteb/cqadupstack-programmers) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackProgrammersRetrieval-Fa` { .model-copy }

CQADupstackProgrammersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-programmers-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Programming, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    


??? info "Tasks"

    | name                                                                              | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [CQADupstackAndroidRetrieval](./retrieval.md#cqadupstackandroidretrieval)         | Retrieval | text         | eng         |
    | [CQADupstackEnglishRetrieval](./retrieval.md#cqadupstackenglishretrieval)         | Retrieval | text         | eng         |
    | [CQADupstackGamingRetrieval](./retrieval.md#cqadupstackgamingretrieval)           | Retrieval | text         | eng         |
    | [CQADupstackGisRetrieval](./retrieval.md#cqadupstackgisretrieval)                 | Retrieval | text         | eng         |
    | [CQADupstackMathematicaRetrieval](./retrieval.md#cqadupstackmathematicaretrieval) | Retrieval | text         | eng         |
    | [CQADupstackPhysicsRetrieval](./retrieval.md#cqadupstackphysicsretrieval)         | Retrieval | text         | eng         |
    | [CQADupstackProgrammersRetrieval](./retrieval.md#cqadupstackprogrammersretrieval) | Retrieval | text         | eng         |
    | [CQADupstackStatsRetrieval](./retrieval.md#cqadupstackstatsretrieval)             | Retrieval | text         | eng         |
    | [CQADupstackTexRetrieval](./retrieval.md#cqadupstacktexretrieval)                 | Retrieval | text         | eng         |
    | [CQADupstackUnixRetrieval](./retrieval.md#cqadupstackunixretrieval)               | Retrieval | text         | eng         |
    | [CQADupstackWebmastersRetrieval](./retrieval.md#cqadupstackwebmastersretrieval)   | Retrieval | text         | eng         |
    | [CQADupstackWordpressRetrieval](./retrieval.md#cqadupstackwordpressretrieval)     | Retrieval | text         | eng         |


#### `CQADupstackRetrieval-Fa` { .model-copy }

CQADupstackRetrieval-Fa

**License:** not specified

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
     
    ```
    


??? info "Tasks"

    | name                                                                                    | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [CQADupstackAndroidRetrieval-Fa](./retrieval.md#cqadupstackandroidretrieval-fa)         | Retrieval | text         | fas         |
    | [CQADupstackEnglishRetrieval-Fa](./retrieval.md#cqadupstackenglishretrieval-fa)         | Retrieval | text         | fas         |
    | [CQADupstackGamingRetrieval-Fa](./retrieval.md#cqadupstackgamingretrieval-fa)           | Retrieval | text         | fas         |
    | [CQADupstackGisRetrieval-Fa](./retrieval.md#cqadupstackgisretrieval-fa)                 | Retrieval | text         | fas         |
    | [CQADupstackMathematicaRetrieval-Fa](./retrieval.md#cqadupstackmathematicaretrieval-fa) | Retrieval | text         | fas         |
    | [CQADupstackPhysicsRetrieval-Fa](./retrieval.md#cqadupstackphysicsretrieval-fa)         | Retrieval | text         | fas         |
    | [CQADupstackProgrammersRetrieval-Fa](./retrieval.md#cqadupstackprogrammersretrieval-fa) | Retrieval | text         | fas         |
    | [CQADupstackStatsRetrieval-Fa](./retrieval.md#cqadupstackstatsretrieval-fa)             | Retrieval | text         | fas         |
    | [CQADupstackTexRetrieval-Fa](./retrieval.md#cqadupstacktexretrieval-fa)                 | Retrieval | text         | fas         |
    | [CQADupstackUnixRetrieval-Fa](./retrieval.md#cqadupstackunixretrieval-fa)               | Retrieval | text         | fas         |
    | [CQADupstackWebmastersRetrieval-Fa](./retrieval.md#cqadupstackwebmastersretrieval-fa)   | Retrieval | text         | fas         |
    | [CQADupstackWordpressRetrieval-Fa](./retrieval.md#cqadupstackwordpressretrieval-fa)     | Retrieval | text         | fas         |


#### `CQADupstackRetrieval-PL` { .model-copy }

CQADupstackRetrieval-PL

**License:** not specified

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Non-fiction, Programming, Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    


??? info "Tasks"

    | name                                                                    | type      | modalities   | languages   |
    |:------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [CQADupstack-Android-PL](./retrieval.md#cqadupstack-android-pl)         | Retrieval | text         | pol         |
    | [CQADupstack-English-PL](./retrieval.md#cqadupstack-english-pl)         | Retrieval | text         | pol         |
    | [CQADupstack-Gaming-PL](./retrieval.md#cqadupstack-gaming-pl)           | Retrieval | text         | pol         |
    | [CQADupstack-Gis-PL](./retrieval.md#cqadupstack-gis-pl)                 | Retrieval | text         | pol         |
    | [CQADupstack-Mathematica-PL](./retrieval.md#cqadupstack-mathematica-pl) | Retrieval | text         | pol         |
    | [CQADupstack-Physics-PL](./retrieval.md#cqadupstack-physics-pl)         | Retrieval | text         | pol         |
    | [CQADupstack-Programmers-PL](./retrieval.md#cqadupstack-programmers-pl) | Retrieval | text         | pol         |
    | [CQADupstack-Stats-PL](./retrieval.md#cqadupstack-stats-pl)             | Retrieval | text         | pol         |
    | [CQADupstack-Tex-PL](./retrieval.md#cqadupstack-tex-pl)                 | Retrieval | text         | pol         |
    | [CQADupstack-Unix-PL](./retrieval.md#cqadupstack-unix-pl)               | Retrieval | text         | pol         |
    | [CQADupstack-Webmasters-PL](./retrieval.md#cqadupstack-webmasters-pl)   | Retrieval | text         | pol         |
    | [CQADupstack-Wordpress-PL](./retrieval.md#cqadupstack-wordpress-pl)     | Retrieval | text         | pol         |


#### `CQADupstackStats-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackStats-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-stats-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-stats-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackStatsRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-stats`](https://huggingface.co/datasets/mteb/cqadupstack-stats) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackStatsRetrieval-Fa` { .model-copy }

CQADupstackStatsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-stats-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackTex-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackTex-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-tex-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-tex-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Non-fiction, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackTexRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-tex`](https://huggingface.co/datasets/mteb/cqadupstack-tex) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackTexRetrieval-Fa` { .model-copy }

CQADupstackTexRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-tex-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackUnix-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackUnix-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-unix-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-unix-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Programming, Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackUnixRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-unix`](https://huggingface.co/datasets/mteb/cqadupstack-unix) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackUnixRetrieval-Fa` { .model-copy }

CQADupstackUnixRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-unix-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackWebmasters-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackWebmasters-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-webmasters-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-webmasters-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackWebmastersRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-webmasters`](https://huggingface.co/datasets/mteb/cqadupstack-webmasters) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackWebmastersRetrieval-Fa` { .model-copy }

CQADupstackWebmastersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-webmasters-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CQADupstackWordpress-NL` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `CQADupstackWordpress-VN` { .model-copy }

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-wordpress-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-wordpress-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Programming, Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `CQADupstackWordpressRetrieval` { .model-copy }

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-wordpress`](https://huggingface.co/datasets/mteb/cqadupstack-wordpress) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### `CQADupstackWordpressRetrieval-Fa` { .model-copy }

CQADupstackWordpressRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-wordpress-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `CUREv1` { .model-copy }

Collection of query-passage pairs curated by medical professionals, across 10 disciplines and 3 cross-lingual settings.

**Dataset:** [`clinia/CUREv1`](https://huggingface.co/datasets/clinia/CUREv1) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/clinia/CUREv1)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, fra, spa | Academic, Medical, Written | expert-annotated | created | ndcg_at_10 |



#### `ChatDoctorRetrieval` { .model-copy }

A medical retrieval task based on ChatDoctor_HealthCareMagic dataset containing 112,000 real-world medical question-and-answer pairs. Each query is a medical question from patients (e.g., 'What are the symptoms of diabetes?'), and the corpus contains medical responses and healthcare information. The task is to retrieve the correct medical information that answers the patient's question. The dataset includes grammatical inconsistencies which help separate strong healthcare retrieval models from weak ones. Queries are patient medical questions while the corpus contains relevant medical responses, diagnoses, and treatment information from healthcare professionals.

**Dataset:** [`embedding-benchmark/ChatDoctor_HealthCareMagic`](https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{chatdoctor_healthcaremagic,
      title = {ChatDoctor HealthCareMagic: Medical Question-Answer Retrieval Dataset},
      url = {https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k},
      year = {2023},
    }
    
    ```
    



#### `ChemHotpotQARetrieval` { .model-copy }

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemHotpotQARetrieval`](https://huggingface.co/datasets/BASF-AI/ChemHotpotQARetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    
    @inproceedings{yang-etal-2018-hotpotqa,
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### `ChemNQRetrieval` { .model-copy }

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemNQRetrieval`](https://huggingface.co/datasets/BASF-AI/ChemNQRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    
    ```
    



#### `ChemRxivRetrieval` { .model-copy }

A retrieval task based on ChemRxiv papers where queries are LLM-synthesized to match specific paragraphs.

**Dataset:** [`BASF-AI/ChemRxivRetrieval`](https://huggingface.co/datasets/BASF-AI/ChemRxivRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2508.01643)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{kasmaee2025chembed,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Astaraki, Mahdi and Saloot, Mohammad Arshi and Sherck, Nicholas and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2508.01643},
      title = {Chembed: Enhancing chemical literature search through domain-specific text embeddings},
      year = {2025},
    }
    ```
    



#### `ClimateFEVER` { .model-copy }

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims (queries) regarding climate-change. The underlying corpus is the same as FEVER.

**Dataset:** [`mteb/climate-fever`](https://huggingface.co/datasets/mteb/climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### `ClimateFEVER-Fa` { .model-copy }

ClimateFEVER-Fa

**Dataset:** [`MCINext/climate-fever-fa`](https://huggingface.co/datasets/MCINext/climate-fever-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/climate-fever-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `ClimateFEVER-NL` { .model-copy }

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. ClimateFEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-climate-fever`](https://huggingface.co/datasets/clips/beir-nl-climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-climate-fever)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Encyclopaedic, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `ClimateFEVER-VN` { .model-copy }

A translated dataset from CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/climate-fever-vn`](https://huggingface.co/datasets/GreenNode/climate-fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `ClimateFEVER.v2` { .model-copy }

CLIMATE-FEVER is a dataset following the FEVER methodology, containing 1,535 real-world climate change claims. This updated version addresses corpus mismatches and qrel inconsistencies in MTEB, restoring labels while refining corpus-query alignment for better accuracy.

**Dataset:** [`mteb/climate-fever-v2`](https://huggingface.co/datasets/mteb/climate-fever-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### `ClimateFEVERHardNegatives` { .model-copy }

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/ClimateFEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/ClimateFEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### `ClimateFEVERHardNegatives.v2` { .model-copy }

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/ClimateFEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/ClimateFEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### `CmedqaRetrieval` { .model-copy }

Online medical consultation text. Used the CMedQAv2 as its underlying dataset.

**Dataset:** [`mteb/CmedqaRetrieval`](https://huggingface.co/datasets/mteb/CmedqaRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Medical, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{qiu2022dureaderretrievallargescalechinesebenchmark,
      archiveprefix = {arXiv},
      author = {Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
      eprint = {2203.10232},
      primaryclass = {cs.CL},
      title = {DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
      url = {https://arxiv.org/abs/2203.10232},
      year = {2022},
    }
    
    ```
    



#### `Code1Retrieval` { .model-copy }

Code retrieval dataset with programming questions paired with C/Python/Go/Ruby code snippets for multi-language code retrieval evaluation. Contributed by Voyage AI.

**Dataset:** [`mteb-private/Code1Retrieval`](https://huggingface.co/datasets/mteb-private/Code1Retrieval) • **License:** bsd-3-clause • [Learn more →](https://huggingface.co/datasets/mteb-private/Code1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Written | derived | found | ndcg_at_10 |



#### `CodeEditSearchRetrieval` { .model-copy }

The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.

**Dataset:** [`cassanof/CodeEditSearch`](https://huggingface.co/datasets/cassanof/CodeEditSearch) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | c, c++, go, java, javascript, ... (13) | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{muennighoff2023octopack,
      author = {Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal = {arXiv preprint arXiv:2308.07124},
      title = {OctoPack: Instruction Tuning Code Large Language Models},
      year = {2023},
    }
    
    ```
    



#### `CodeFeedbackMT` { .model-copy }

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-mt`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-mt) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.14658)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{zheng2024opencodeinterpreterintegratingcodegeneration,
      archiveprefix = {arXiv},
      author = {Tianyu Zheng and Ge Zhang and Tianhao Shen and Xueling Liu and Bill Yuchen Lin and Jie Fu and Wenhu Chen and Xiang Yue},
      eprint = {2402.14658},
      primaryclass = {cs.SE},
      title = {OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement},
      url = {https://arxiv.org/abs/2402.14658},
      year = {2024},
    }
    
    ```
    



#### `CodeFeedbackST` { .model-copy }

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-st`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-st) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2024coircomprehensivebenchmarkcode,
      archiveprefix = {arXiv},
      author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
      eprint = {2407.02883},
      primaryclass = {cs.IR},
      title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      url = {https://arxiv.org/abs/2407.02883},
      year = {2024},
    }
    
    ```
    



#### `CodeSearchNetCCRetrieval` { .model-copy }

The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet-ccr`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet-ccr) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2024coircomprehensivebenchmarkcode,
      archiveprefix = {arXiv},
      author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
      eprint = {2407.02883},
      primaryclass = {cs.IR},
      title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      url = {https://arxiv.org/abs/2407.02883},
      year = {2024},
    }
    
    ```
    



#### `CodeSearchNetRetrieval` { .model-copy }

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`mteb/CodeSearchNetRetrieval`](https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{husain2019codesearchnet,
      author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
      journal = {arXiv preprint arXiv:1909.09436},
      title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
      year = {2019},
    }
    
    ```
    



#### `CodeTransOceanContest` { .model-copy }

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet

**Dataset:** [`CoIR-Retrieval/codetrans-contest`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-contest) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | c++, python | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
      archiveprefix = {arXiv},
      author = {Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
      eprint = {2310.04951},
      primaryclass = {cs.AI},
      title = {CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation},
      url = {https://arxiv.org/abs/2310.04951},
      year = {2023},
    }
    
    ```
    



#### `CodeTransOceanDL` { .model-copy }

The dataset is a collection of equivalent Python Deep Learning code snippets written in different machine learning framework. The task is to retrieve the equivalent code snippet in another framework, given a query code snippet from one framework.

**Dataset:** [`CoIR-Retrieval/codetrans-dl`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-dl) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | python | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
      archiveprefix = {arXiv},
      author = {Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
      eprint = {2310.04951},
      primaryclass = {cs.AI},
      title = {CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation},
      url = {https://arxiv.org/abs/2310.04951},
      year = {2023},
    }
    
    ```
    



#### `CosQA` { .model-copy }

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/cosqa`](https://huggingface.co/datasets/CoIR-Retrieval/cosqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.13239)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{huang2021cosqa20000webqueries,
      archiveprefix = {arXiv},
      author = {Junjie Huang and Duyu Tang and Linjun Shou and Ming Gong and Ke Xu and Daxin Jiang and Ming Zhou and Nan Duan},
      eprint = {2105.13239},
      primaryclass = {cs.CL},
      title = {CoSQA: 20,000+ Web Queries for Code Search and Question Answering},
      url = {https://arxiv.org/abs/2105.13239},
      year = {2021},
    }
    
    ```
    



#### `CovidRetrieval` { .model-copy }

COVID-19 news articles

**Dataset:** [`mteb/CovidRetrieval`](https://huggingface.co/datasets/mteb/CovidRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Entertainment, Medical | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### `CrossLingualSemanticDiscriminationWMT19` { .model-copy }

Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT19 DE-FR test set

**Dataset:** [`Andrianos/clsd_wmt19_21`](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu, fra | News, Written | derived | LM-generated and verified | recall_at_1 |



#### `CrossLingualSemanticDiscriminationWMT21` { .model-copy }

Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT21 DE-FR test set

**Dataset:** [`Andrianos/clsd_wmt19_21`](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu, fra | News, Written | derived | LM-generated and verified | recall_at_1 |



#### `DAPFAMAllTitlAbsClmToFullTextRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMAllTitlAbsClmToTitlAbsRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMAllTitlAbsToFullTextRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMAllTitlAbsToTitlAbsClmRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMAllTitlAbsToTitlAbsRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMInTitlAbsClmToFullTextRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMInTitlAbsClmToTitlAbsClmRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMInTitlAbsClmToTitlAbsRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMInTitlAbsToFullTextRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMInTitlAbsToTitlAbsClmRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMInTitlAbsToTitlAbsRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMOutTitlAbsClmToFullTextRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMOutTitlAbsClmToTitlAbsRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMOutTitlAbsToFullTextRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMOutTitlAbsToTitlAbsClmRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DAPFAMOutTitlAbsToTitlAbsRetrieval` { .model-copy }

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry, Engineering, Legal | derived | created | ndcg_at_100 |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### `DBPedia` { .model-copy }

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/dbpedia`](https://huggingface.co/datasets/mteb/dbpedia) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### `DBPedia-Fa` { .model-copy }

DBPedia-Fa

**Dataset:** [`MCINext/dbpedia-fa`](https://huggingface.co/datasets/MCINext/dbpedia-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/dbpedia-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `DBPedia-NL` { .model-copy }

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. DBPedia-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-dbpedia-entity`](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Encyclopaedic, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `DBPedia-PL` { .model-copy }

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/DBPedia-PL`](https://huggingface.co/datasets/mteb/DBPedia-PL) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Encyclopaedic, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### `DBPedia-PLHardNegatives` { .model-copy }

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia-PLHardNegatives`](https://huggingface.co/datasets/mteb/DBPedia-PLHardNegatives) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Encyclopaedic, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### `DBPedia-VN` { .model-copy }

A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/dbpedia-vn`](https://huggingface.co/datasets/GreenNode/dbpedia-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `DBPediaHardNegatives` { .model-copy }

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### `DBPediaHardNegatives.v2` { .model-copy }

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/DBPedia_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### `DS1000Retrieval` { .model-copy }

A code retrieval task based on 1,000 data science programming problems from DS-1000. Each query is a natural language description of a data science task (e.g., 'Create a scatter plot of column A vs column B with matplotlib'), and the corpus contains Python code implementations using libraries like pandas, numpy, matplotlib, scikit-learn, and scipy. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations focused on data science workflows.

**Dataset:** [`embedding-benchmark/DS1000`](https://huggingface.co/datasets/embedding-benchmark/DS1000) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/DS1000)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{lai2022ds,
      author = {Lai, Yuhang and Li, Chengxi and Wang, Yiming and Zhang, Tianyi and Zhong, Ruiqi and Zettlemoyer, Luke and Yih, Wen-tau and Fried, Daniel and Wang, Sida and Yu, Tao},
      journal = {arXiv preprint arXiv:2211.11501},
      title = {DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
      year = {2022},
    }
    
    ```
    



#### `DanFEVER` { .model-copy }

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.

**Dataset:** [`mteb/DanFEVER`](https://huggingface.co/datasets/mteb/DanFEVER) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{norregaard-derczynski-2021-danfever,
      address = {Reykjavik, Iceland (Online)},
      author = {N{\o}rregaard, Jeppe  and
    Derczynski, Leon},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {422--428},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {{D}an{FEVER}: claim verification dataset for {D}anish},
      url = {https://aclanthology.org/2021.nodalida-main.47},
      year = {2021},
    }
    
    ```
    



#### `DanFeverRetrieval` { .model-copy }

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset. DanFeverRetrieval fixed an issue in DanFever where some corpus entries were incorrectly removed.

**Dataset:** [`strombergnlp/danfever`](https://huggingface.co/datasets/strombergnlp/danfever) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{norregaard-derczynski-2021-danfever,
      address = {Reykjavik, Iceland (Online)},
      author = {N{\o}rregaard, Jeppe  and
    Derczynski, Leon},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {422--428},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {{D}an{FEVER}: claim verification dataset for {D}anish},
      url = {https://aclanthology.org/2021.nodalida-main.47},
      year = {2021},
    }
    
    ```
    



#### `DuRetrieval` { .model-copy }

A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine

**Dataset:** [`mteb/DuRetrieval`](https://huggingface.co/datasets/mteb/DuRetrieval) • **License:** apache-2.0 • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{qiu2022dureaderretrievallargescalechinesebenchmark,
      archiveprefix = {arXiv},
      author = {Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
      eprint = {2203.10232},
      primaryclass = {cs.CL},
      title = {DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
      url = {https://arxiv.org/abs/2203.10232},
      year = {2022},
    }
    
    ```
    



#### `DutchNewsArticlesRetrieval` { .model-copy }

This dataset contains all the articles published by the NOS as of the 1st of January 2010. The data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news organizations in the Netherlands.

**Dataset:** [`clips/mteb-nl-news-articles-ret`](https://huggingface.co/datasets/clips/mteb-nl-news-articles-ret) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | News, Written | derived | found | ndcg_at_10 |



#### `EcomRetrieval` { .model-copy }

EcomRetrieval

**Dataset:** [`mteb/EcomRetrieval`](https://huggingface.co/datasets/mteb/EcomRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Reviews, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### `EnglishFinance1Retrieval` { .model-copy }

Financial document retrieval dataset with queries about stock compensation, corporate governance, and SEC filing content. Contributed by Voyage AI.

**Dataset:** [`mteb-private/EnglishFinance1Retrieval`](https://huggingface.co/datasets/mteb-private/EnglishFinance1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/EnglishFinance1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



#### `EnglishFinance2Retrieval` { .model-copy }

Financial performance retrieval dataset with queries about stock performance, S&P 500 comparisons, and railroad industry metrics. Contributed by Voyage AI.

**Dataset:** [`mteb-private/EnglishFinance2Retrieval`](https://huggingface.co/datasets/mteb-private/EnglishFinance2Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/EnglishFinance2Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



#### `EnglishFinance3Retrieval` { .model-copy }

Personal finance Q&A retrieval dataset with questions about tax codes, business expenses, and financial advice. Contributed by Voyage AI.

**Dataset:** [`mteb-private/EnglishFinance3Retrieval`](https://huggingface.co/datasets/mteb-private/EnglishFinance3Retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/mteb-private/EnglishFinance3Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



#### `EnglishFinance4Retrieval` { .model-copy }

Personal finance advice retrieval dataset with questions about car financing, investment strategies, and financial planning. Contributed by Voyage AI.

**Dataset:** [`mteb-private/EnglishFinance4Retrieval`](https://huggingface.co/datasets/mteb-private/EnglishFinance4Retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/mteb-private/EnglishFinance4Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Non-fiction, Written | derived | found | ndcg_at_10 |



#### `EnglishHealthcare1Retrieval` { .model-copy }

Medical research retrieval dataset with queries about HIV transmission, genetic variants, and biomedical research findings. Contributed by Voyage AI.

**Dataset:** [`mteb-private/EnglishHealthcare1Retrieval`](https://huggingface.co/datasets/mteb-private/EnglishHealthcare1Retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/mteb-private/EnglishHealthcare1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | derived | found | ndcg_at_10 |



#### `EstQA` { .model-copy }

EstQA is an Estonian question answering dataset based on Wikipedia.

**Dataset:** [`kardosdrur/estonian-qa`](https://huggingface.co/datasets/kardosdrur/estonian-qa) • **License:** not specified • [Learn more →](https://www.semanticscholar.org/paper/Extractive-Question-Answering-for-Estonian-Language-182912IAPM-Alum%C3%A4e/ea4f60ab36cadca059c880678bc4c51e293a85d6?utm_source=direct_link)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | est | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{mastersthesis,
      author = {Anu Käver},
      school = {Tallinn University of Technology (TalTech)},
      title = {Extractive Question Answering for Estonian Language},
      year = {2021},
    }
    
    ```
    



#### `EuroPIRQRetrieval` { .model-copy }

The EuroPIRQ retrieval dataset is a multilingual collection designed for evaluating retrieval and cross-lingual retrieval tasks. Dataset contains 10,000 parallel passages & 100 parallel queries (synthetic) in three languages: English, Portuguese, and Finnish, constructed from the European Union's DGT-Acquis corpus.

**Dataset:** [`eherra/EuroPIRQ-retrieval`](https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, fin, por | Legal | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{eherra_2025_europirq,
      author = { {Elias Herranen} },
      publisher = { Hugging Face },
      title = { EuroPIRQ: European Parallel Information Retrieval Queries },
      url = { https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval },
      year = {2025},
    }
    
    ```
    



#### `FEVER` { .model-copy }

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`mteb/fever`](https://huggingface.co/datasets/mteb/fever) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### `FEVER-FaHardNegatives` { .model-copy }

FEVER-FaHardNegatives

**Dataset:** [`MCINext/FEVER_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/FEVER_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/FEVER_FA_test_top_250_only_w_correct-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `FEVER-NL` { .model-copy }

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. FEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-fever`](https://huggingface.co/datasets/clips/beir-nl-fever) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fever)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Encyclopaedic, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `FEVER-VN` { .model-copy }

A translated dataset from FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fever-vn`](https://huggingface.co/datasets/GreenNode/fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://fever.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `FEVERHardNegatives` { .model-copy }

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/FEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### `FEVERHardNegatives.v2` { .model-copy }

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/FEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### `FQuADRetrieval` { .model-copy }

This dataset has been built from the French SQuad dataset.

**Dataset:** [`manu/fquad2_test`](https://huggingface.co/datasets/manu/fquad2_test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/manu/fquad2_test)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Encyclopaedic, Written | human-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{dhoffschmidt-etal-2020-fquad,
      address = {Online},
      author = {d{'}Hoffschmidt, Martin  and
    Belblidia, Wacim  and
    Heinrich, Quentin  and
    Brendl{\'e}, Tom  and
    Vidal, Maxime},
      booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
      doi = {10.18653/v1/2020.findings-emnlp.107},
      editor = {Cohn, Trevor  and
    He, Yulan  and
    Liu, Yang},
      month = nov,
      pages = {1193--1208},
      publisher = {Association for Computational Linguistics},
      title = {{FQ}u{AD}: {F}rench Question Answering Dataset},
      url = {https://aclanthology.org/2020.findings-emnlp.107},
      year = {2020},
    }
    
    ```
    



#### `FaithDial` { .model-copy }

FaithDial is a faithful knowledge-grounded dialogue benchmark.It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). It consists of conversation histories along with manually labelled relevant passage. For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, as the gold passage associated with these instances is non-ambiguous.

**Dataset:** [`mteb/FaithDial`](https://huggingface.co/datasets/mteb/FaithDial) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/FaithDial)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{dziri2022faithdial,
      author = {Dziri, Nouha and Kamalloo, Ehsan and Milton, Sivan and Zaiane, Osmar and Yu, Mo and Ponti, Edoardo M and Reddy, Siva},
      doi = {10.1162/tacl_a_00529},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {12},
      pages = {1473--1490},
      publisher = {MIT Press},
      title = {{FaithDial: A Faithful Benchmark for Information-Seeking Dialogue}},
      volume = {10},
      year = {2022},
    }
    
    ```
    



#### `FeedbackQARetrieval` { .model-copy }

Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment

**Dataset:** [`mteb/FeedbackQARetrieval`](https://huggingface.co/datasets/mteb/FeedbackQARetrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.03025)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Government, Medical, Web, Written | human-annotated | created | precision_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{li-etal-2022-using,
      address = {Dublin, Ireland},
      author = {Li, Zichao  and
    Sharma, Prakhar  and
    Lu, Xing Han  and
    Cheung, Jackie  and
    Reddy, Siva},
      booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
      doi = {10.18653/v1/2022.findings-acl.75},
      editor = {Muresan, Smaranda  and
    Nakov, Preslav  and
    Villavicencio, Aline},
      month = may,
      pages = {926--937},
      publisher = {Association for Computational Linguistics},
      title = {Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment},
      url = {https://aclanthology.org/2022.findings-acl.75},
      year = {2022},
    }
    
    ```
    



#### `FiQA-PL` { .model-copy }

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/FiQA-PL`](https://huggingface.co/datasets/mteb/FiQA-PL) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Financial, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thakur2021beir,
      author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
      url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
      year = {2021},
    }
    
    ```
    



#### `FiQA2018` { .model-copy }

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/fiqa`](https://huggingface.co/datasets/mteb/fiqa) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Financial, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thakur2021beir,
      author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
      url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
      year = {2021},
    }
    
    ```
    



#### `FiQA2018-Fa` { .model-copy }

FiQA2018-Fa

**Dataset:** [`MCINext/fiqa-fa`](https://huggingface.co/datasets/MCINext/fiqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/fiqa-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `FiQA2018-Fa.v2` { .model-copy }

FiQA2018-Fa.v2

**Dataset:** [`MCINext/fiqa-fa-v2`](https://huggingface.co/datasets/MCINext/fiqa-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/fiqa-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `FiQA2018-NL` { .model-copy }

Financial Opinion Mining and Question Answering. FiQA2018-NL is a Dutch translation

**Dataset:** [`clips/beir-nl-fiqa`](https://huggingface.co/datasets/clips/beir-nl-fiqa) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fiqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `FiQA2018-VN` { .model-copy }

A translated dataset from Financial Opinion Mining and Question Answering The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fiqa-vn`](https://huggingface.co/datasets/GreenNode/fiqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Financial, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `FinQARetrieval` { .model-copy }

A financial retrieval task based on FinQA dataset containing numerical reasoning questions over financial documents. Each query is a financial question requiring numerical computation (e.g., 'What is the percentage change in operating expenses from 2019 to 2020?'), and the corpus contains financial document text with tables and numerical data. The task is to retrieve the correct financial information that enables answering the numerical question. Queries are numerical reasoning questions while the corpus contains financial text passages with embedded tables, figures, and quantitative financial data from earnings reports.

**Dataset:** [`embedding-benchmark/FinQA`](https://huggingface.co/datasets/embedding-benchmark/FinQA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinQA)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Financial | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{chen2021finqa,
      author = {Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
      journal = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      title = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
      year = {2021},
    }
    
    ```
    



#### `FinanceBenchRetrieval` { .model-copy }

A financial retrieval task based on FinanceBench dataset containing financial questions and answers. Each query is a financial question (e.g., 'What was the total revenue in Q3 2023?'), and the corpus contains financial document excerpts and annual reports. The task is to retrieve the correct financial information that answers the question. Queries are financial questions while the corpus contains relevant excerpts from financial documents, earnings reports, and SEC filings with detailed financial data and metrics.

**Dataset:** [`embedding-benchmark/FinanceBench`](https://huggingface.co/datasets/embedding-benchmark/FinanceBench) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinanceBench)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Financial | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{islam2023financebench,
      author = {Islam, Pranab and Kannappan, Anand and Kiela, Douwe and Fergus, Rob and Ott, Myle and Wang, Sam and Garimella, Aparna and Garcia, Nino},
      journal = {arXiv preprint arXiv:2311.11944},
      title = {FinanceBench: A New Benchmark for Financial Question Answering},
      year = {2023},
    }
    
    ```
    



#### `French1Retrieval` { .model-copy }

French general knowledge retrieval dataset with queries about celebrities, historical figures, and cultural topics. Contributed by Voyage AI.

**Dataset:** [`mteb-private/French1Retrieval`](https://huggingface.co/datasets/mteb-private/French1Retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/mteb-private/French1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Encyclopaedic, Written | derived | found | ndcg_at_10 |



#### `FrenchLegal1Retrieval` { .model-copy }

French legal document retrieval dataset with queries about administrative law, court decisions, and legal proceedings. Contributed by Voyage AI.

**Dataset:** [`mteb-private/FrenchLegal1Retrieval`](https://huggingface.co/datasets/mteb-private/FrenchLegal1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/FrenchLegal1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Legal, Written | derived | found | ndcg_at_10 |



#### `FreshStackRetrieval` { .model-copy }

A code retrieval task based on FreshStack dataset containing programming problems across multiple languages. Each query is a natural language description of a programming task (e.g., 'Write a function to reverse a string using recursion'), and the corpus contains code implementations in Python, JavaScript, and Go. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains function implementations with proper syntax and logic across different programming languages.

**Dataset:** [`embedding-benchmark/FreshStack_mteb`](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, go, javascript, python | Programming | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{thakur2025freshstackbuildingrealisticbenchmarks,
      archiveprefix = {arXiv},
      author = {Nandan Thakur and Jimmy Lin and Sam Havens and Michael Carbin and Omar Khattab and Andrew Drozdov},
      eprint = {2504.13128},
      primaryclass = {cs.IR},
      title = {FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents},
      url = {https://arxiv.org/abs/2504.13128},
      year = {2025},
    }
    
    ```
    



#### `GeorgianFAQRetrieval` { .model-copy }

Frequently asked questions (FAQs) and answers mined from Georgian websites via Common Crawl.

**Dataset:** [`jupyterjazz/georgian-faq`](https://huggingface.co/datasets/jupyterjazz/georgian-faq) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jupyterjazz/georgian-faq)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kat | Web, Written | derived | created | ndcg_at_10 |



#### `GerDaLIR` { .model-copy }

GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.

**Dataset:** [`mteb/GerDaLIR`](https://huggingface.co/datasets/mteb/GerDaLIR) • **License:** mit • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wrzalik-krechel-2021-gerdalir,
      address = {Punta Cana, Dominican Republic},
      author = {Wrzalik, Marco  and
    Krechel, Dirk},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
      month = nov,
      pages = {123--128},
      publisher = {Association for Computational Linguistics},
      title = {{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval},
      url = {https://aclanthology.org/2021.nllp-1.13},
      year = {2021},
    }
    
    ```
    



#### `GerDaLIRSmall` { .model-copy }

The dataset consists of documents, passages and relevance labels in German. In contrast to the original dataset, only documents that have corresponding queries in the query set are chosen to create a smaller corpus for evaluation purposes.

**Dataset:** [`mteb/GerDaLIRSmall`](https://huggingface.co/datasets/mteb/GerDaLIRSmall) • **License:** mit • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wrzalik-krechel-2021-gerdalir,
      address = {Punta Cana, Dominican Republic},
      author = {Wrzalik, Marco  and
    Krechel, Dirk},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
      month = nov,
      pages = {123--128},
      publisher = {Association for Computational Linguistics},
      title = {{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval},
      url = {https://aclanthology.org/2021.nllp-1.13},
      year = {2021},
    }
    
    ```
    



#### `German1Retrieval` { .model-copy }

German dialogue retrieval dataset with business conversations and workplace communication scenarios. Contributed by Voyage AI.

**Dataset:** [`mteb-private/German1Retrieval`](https://huggingface.co/datasets/mteb-private/German1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/German1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Non-fiction, Written | derived | found | ndcg_at_10 |



#### `GermanDPR` { .model-copy }

GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer

**Dataset:** [`mteb/GermanDPR`](https://huggingface.co/datasets/mteb/GermanDPR) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germandpr)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Non-fiction, Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{möller2021germanquad,
      archiveprefix = {arXiv},
      author = {Timo Möller and Julian Risch and Malte Pietsch},
      eprint = {2104.12741},
      primaryclass = {cs.CL},
      title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
      year = {2021},
    }
    
    ```
    



#### `GermanGovServiceRetrieval` { .model-copy }

LHM-Dienstleistungen-QA is a German question answering dataset for government services of the Munich city administration. It associates questions with a textual context containing the answer

**Dataset:** [`it-at-m/LHM-Dienstleistungen-QA`](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Government, Written | derived | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @software{lhm-dienstleistungen-qa,
      author = {Schröder, Leon Marius and
    Gutknecht, Clemens and
    Alkiddeh, Oubada and
    Susanne Weiß,
    Lukas, Leon},
      month = nov,
      publisher = {it@M},
      title = {LHM-Dienstleistungen-QA - german public domain question-answering dataset},
      url = {https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA},
      year = {2022},
    }
    
    ```
    



#### `GermanHealthcare1Retrieval` { .model-copy }

German medical consultation retrieval dataset with patient questions and doctor responses about various health conditions. Contributed by Voyage AI.

**Dataset:** [`mteb-private/GermanHealthcare1Retrieval`](https://huggingface.co/datasets/mteb-private/GermanHealthcare1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/GermanHealthcare1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Medical, Written | derived | found | ndcg_at_10 |



#### `GermanLegal1Retrieval` { .model-copy }

German educational regulation retrieval dataset with queries about university capacity calculations and academic administration. Contributed by Voyage AI.

**Dataset:** [`mteb-private/GermanLegal1Retrieval`](https://huggingface.co/datasets/mteb-private/GermanLegal1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/GermanLegal1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Legal, Written | derived | found | ndcg_at_10 |



#### `GermanQuAD-Retrieval` { .model-copy }

Context Retrieval for German Question Answering

**Dataset:** [`mteb/germanquad-retrieval`](https://huggingface.co/datasets/mteb/germanquad-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germanquad)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Non-fiction, Web, Written | human-annotated | found | mrr_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{möller2021germanquad,
      archiveprefix = {arXiv},
      author = {Timo Möller and Julian Risch and Malte Pietsch},
      eprint = {2104.12741},
      primaryclass = {cs.CL},
      title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
      year = {2021},
    }
    
    ```
    



#### `GovReport` { .model-copy }

A dataset for evaluating the ability of information retrieval models to retrieve lengthy US government reports from their summaries.

**Dataset:** [`isaacus/mteb-GovReport`](https://huggingface.co/datasets/isaacus/mteb-GovReport) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/launch/gov_report)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Government, Legal | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{huang-etal-2021-efficient,
      address = {Online},
      author = {Huang, Luyang  and
    Cao, Shuyang  and
    Parulian, Nikolaus  and
    Ji, Heng  and
    Wang, Lu},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2021.naacl-main.112},
      eprint = {2104.02112},
      month = jun,
      pages = {1419--1436},
      publisher = {Association for Computational Linguistics},
      title = {Efficient Attentions for Long Document Summarization},
      url = {https://aclanthology.org/2021.naacl-main.112},
      year = {2021},
    }
    
    ```
    



#### `GreekCivicsQA` { .model-copy }

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`ilsp/greek_civics_qa`](https://huggingface.co/datasets/ilsp/greek_civics_qa) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ell | Academic, Written | derived | found | ndcg_at_10 |



#### `GreenNodeTableMarkdownRetrieval` { .model-copy }

GreenNodeTable documents

**Dataset:** [`GreenNode/GreenNode-Table-Markdown-Retrieval-VN`](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN) • **License:** mit • [Learn more →](https://huggingface.co/GreenNode)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Financial, Non-fiction | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{10.1007/978-981-95-1746-6_17,
      abstract = {Information retrieval often comes in plain text, lacking semi-structured text such as HTML and markdown, retrieving data that contains rich format such as table became non-trivial. In this paper, we tackle this challenge by introducing a new dataset, GreenNode Table Retrieval VN (GN-TRVN), which is collected from a massive corpus, a wide range of topics, and a longer context compared to ViQuAD2.0. To evaluate the effectiveness of our proposed dataset, we introduce two versions, M3-GN-VN and M3-GN-VN-Mixed, by fine-tuning the M3-Embedding model on this dataset. Experimental results show that our models consistently outperform the baselines, including the base model, across most evaluation criteria on various datasets such as VieQuADRetrieval, ZacLegalTextRetrieval, and GN-TRVN. In general, we release a more comprehensive dataset and two model versions that improve response performance for Vietnamese Markdown Table Retrieval.},
      address = {Singapore},
      author = {Pham, Bao Loc
    and Hoang, Quoc Viet
    and Luu, Quy Tung
    and Vo, Trong Thu},
      booktitle = {Proceedings of the Fifth International Conference on Intelligent Systems and Networks},
      isbn = {978-981-95-1746-6},
      pages = {153--163},
      publisher = {Springer Nature Singapore},
      title = {GN-TRVN: A Benchmark for Vietnamese Table Markdown Retrieval Task},
      year = {2026},
    }
    
    ```
    



#### `HC3FinanceRetrieval` { .model-copy }

A financial retrieval task based on HC3 Finance dataset containing human vs AI-generated financial text detection. Each query is a financial question or prompt (e.g., 'Explain the impact of interest rate changes on bond prices'), and the corpus contains both human-written and AI-generated financial responses. The task is to retrieve the most relevant and accurate financial content that addresses the query. Queries are financial questions while the corpus contains detailed financial explanations, analysis, and educational content covering various financial concepts and market dynamics.

**Dataset:** [`embedding-benchmark/HC3Finance`](https://huggingface.co/datasets/embedding-benchmark/HC3Finance) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HC3Finance)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Financial | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{guo2023hc3,
      author = {Guo, Biyang and Zhang, Xin and Wang, Zhiyuan and Jiang, Mingyuan and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng},
      journal = {arXiv preprint arXiv:2301.07597},
      title = {How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
      year = {2023},
    }
    
    ```
    



#### `HagridRetrieval` { .model-copy }

HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)is a dataset for generative information-seeking scenarios. It consists of queriesalong with a set of manually labelled relevant passages

**Dataset:** [`mteb/HagridRetrieval`](https://huggingface.co/datasets/mteb/HagridRetrieval) • **License:** apache-2.0 • [Learn more →](https://github.com/project-miracl/hagrid)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{hagrid,
      author = {Ehsan Kamalloo and Aref Jafari and Xinyu Zhang and Nandan Thakur and Jimmy Lin},
      journal = {arXiv:2307.16883},
      title = {{HAGRID}: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution},
      year = {2023},
    }
    
    ```
    



#### `HellaSwag` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on HellaSwag.

**Dataset:** [`mteb/HellaSwag`](https://huggingface.co/datasets/mteb/HellaSwag) • **License:** mit • [Learn more →](https://rowanzellers.com/hellaswag/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    @article{zellers2019hellaswag,
      author = {Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
      journal = {arXiv preprint arXiv:1905.07830},
      title = {Hellaswag: Can a machine really finish your sentence?},
      year = {2019},
    }
    
    ```
    



#### `HotpotQA` { .model-copy }

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/hotpotqa`](https://huggingface.co/datasets/mteb/hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### `HotpotQA-Fa` { .model-copy }

HotpotQA-Fa

**Dataset:** [`MCINext/hotpotqa-fa`](https://huggingface.co/datasets/MCINext/hotpotqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/hotpotqa-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `HotpotQA-FaHardNegatives` { .model-copy }

HotpotQA-FaHardNegatives

**Dataset:** [`MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `HotpotQA-NL` { .model-copy }

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strongsupervision for supporting facts to enable more explainable question answering systems. HotpotQA-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-hotpotqa`](https://huggingface.co/datasets/clips/beir-nl-hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Web, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `HotpotQA-PL` { .model-copy }

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/HotpotQA-PL`](https://huggingface.co/datasets/mteb/HotpotQA-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `HotpotQA-PLHardNegatives` { .model-copy }

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA-PLHardNegatives`](https://huggingface.co/datasets/mteb/HotpotQA-PLHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `HotpotQA-VN` { .model-copy }

A translated dataset from HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/hotpotqa-vn`](https://huggingface.co/datasets/GreenNode/hotpotqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `HotpotQAHardNegatives` { .model-copy }

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### `HotpotQAHardNegatives.v2` { .model-copy }

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/HotpotQA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### `HumanEvalRetrieval` { .model-copy }

A code retrieval task based on 164 Python programming problems from HumanEval. Each query is a natural language description of a programming task (e.g., 'Check if in given list of numbers, are any two numbers closer to each other than given threshold'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper indentation and logic.

**Dataset:** [`embedding-benchmark/HumanEval`](https://huggingface.co/datasets/embedding-benchmark/HumanEval) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HumanEval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{chen2021evaluating,
      archiveprefix = {arXiv},
      author = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and Ray, Alex and Puri, Raul and Krueger, Gretchen and Petrov, Michael and Khlaaf, Heidy and Sastry, Girish and Mishkin, Pamela and Chan, Brooke and Gray, Scott and Ryder, Nick and Pavlov, Mikhail and Power, Alethea and Kaiser, Lukasz and Bavarian, Mohammad and Winter, Clemens and Tillet, Philippe and Such, Felipe Petroski and Cummings, Dave and Plappert, Matthias and Chantzis, Fotios and Barnes, Elizabeth and Herbert-Voss, Ariel and Guss, William Hebgen and Nichol, Alex and Paino, Alex and Tezak, Nikolas and Tang, Jie and Babuschkin, Igor and Balaji, Suchir and Jain, Shantanu and Saunders, William and Hesse, Christopher and Carr, Andrew N. and Leike, Jan and Achiam, Joshua and Misra, Vedant and Morikawa, Evan and Radford, Alec and Knight, Matthew and Brundage, Miles and Murati, Mira and Mayer, Katie and Welinder, Peter and McGrew, Bob and Amodei, Dario and McCandlish, Sam and Sutskever, Ilya and Zaremba, Wojciech},
      eprint = {2107.03374},
      primaryclass = {cs.LG},
      title = {Evaluating Large Language Models Trained on Code},
      year = {2021},
    }
    ```
    



#### `HunSum2AbstractiveRetrieval` { .model-copy }

HunSum-2-abstractive is a Hungarian dataset containing news articles along with lead, titles and metadata.

**Dataset:** [`SZTAKI-HLT/HunSum-2-abstractive`](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-2-abstractive) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2404.03555)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | hun | News, Written | derived | found | ndcg_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{barta2024news,
      archiveprefix = {arXiv},
      author = {Botond Barta and Dorina Lakatos and Attila Nagy and Milán Konor Nyist and Judit Ács},
      eprint = {2404.03555},
      primaryclass = {cs.CL},
      title = {From News to Summaries: Building a Hungarian Corpus for Extractive and Abstractive Summarization},
      year = {2024},
    }
    
    ```
    



#### `IndicQARetrieval` { .model-copy }

IndicQA is a manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages. It is repurposed retrieving relevant context for each question.

**Dataset:** [`mteb/IndicQARetrieval`](https://huggingface.co/datasets/mteb/IndicQARetrieval) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | asm, ben, guj, hin, kan, ... (11) | Web, Written | human-annotated | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{doddapaneni2022towards,
      author = {Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
      doi = {10.18653/v1/2023.acl-long.693},
      journal = {Annual Meeting of the Association for Computational Linguistics},
      title = {Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
      year = {2022},
    }
    
    ```
    



#### `JaCWIRRetrieval` { .model-copy }

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of 5000 question texts and approximately 500k web page titles and web page introductions or summaries (meta descriptions, etc.). The question texts are created based on one of the 500k web pages, and that data is used as a positive example for the question text.

**Dataset:** [`mteb/JaCWIRRetrieval`](https://huggingface.co/datasets/mteb/JaCWIRRetrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yuichi-tateno-2024-jacwir,
      author = {Yuichi Tateno},
      title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
    }
    
    ```
    



#### `JaCWIRRetrievalLite` { .model-copy }

JaCWIR (Japanese Casual Web IR) is a dataset consisting of questions and webpage meta descriptions collected from Hatena Bookmark. This is the lightweight version with a reduced corpus (302,638 documents) constructed using hard negatives from 5 high-performance models.

**Dataset:** [`mteb/JaCWIRRetrievalLite`](https://huggingface.co/datasets/mteb/JaCWIRRetrievalLite) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb_lite,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
    and Kawahara, Daisuke},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
      title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
      year = {2025},
    }
    
    @misc{yuichi-tateno-2024-jacwir,
      author = {Yuichi Tateno},
      title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
    }
    
    ```
    



#### `JaGovFaqsRetrieval` { .model-copy }

JaGovFaqs is a dataset consisting of FAQs manually extracted from the website of Japanese bureaus. The dataset consists of 22k FAQs, where the queries (questions) and corpus (answers) have been shuffled, and the goal is to match the answer with the question.

**Dataset:** [`mteb/JaGovFaqsRetrieval`](https://huggingface.co/datasets/mteb/JaGovFaqsRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | derived | found | ndcg_at_10 |



#### `JaQuADRetrieval` { .model-copy }

Human-annotated question-answer pairs for Japanese wikipedia pages.

**Dataset:** [`mteb/JaQuADRetrieval`](https://huggingface.co/datasets/mteb/JaQuADRetrieval) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/2202.01764)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{so2022jaquad,
      archiveprefix = {arXiv},
      author = {ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
      eprint = {2202.01764},
      primaryclass = {cs.CL},
      title = {{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
      year = {2022},
    }
    
    ```
    



#### `JapaneseCode1Retrieval` { .model-copy }

Japanese code retrieval dataset. Japanese natural language queries paired with Python code snippets for cross-lingual code retrieval evaluation. Contributed by Voyage AI.

**Dataset:** [`mteb-private/JapaneseCode1Retrieval`](https://huggingface.co/datasets/mteb-private/JapaneseCode1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/JapaneseCode1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Programming, Written | derived | found | ndcg_at_10 |



#### `JapaneseLegal1Retrieval` { .model-copy }

Japanese legal regulation retrieval dataset with queries about government regulations, ministry ordinances, and administrative law. Contributed by Voyage AI.

**Dataset:** [`mteb-private/JapaneseLegal1Retrieval`](https://huggingface.co/datasets/mteb-private/JapaneseLegal1Retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb-private/JapaneseLegal1Retrieval-sample)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Legal, Written | derived | found | ndcg_at_10 |



#### `JaqketRetrieval` { .model-copy }

JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset that is created based on quiz questions.

**Dataset:** [`mteb/jaqket`](https://huggingface.co/datasets/mteb/jaqket) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/kumapo/JAQKET-dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Kurihara_nlp2020,
      author = {鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也},
      booktitle = {言語処理学会第26回年次大会},
      note = {in Japanese},
      title = {JAQKET: クイズを題材にした日本語 QA データセットの構築},
      url = {https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf},
      year = {2020},
    }
    
    ```
    



#### `JaqketRetrievalLite` { .model-copy }

JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset created based on quiz questions. This is the lightweight version with a reduced corpus (65,802 documents) constructed using hard negatives from 5 high-performance models.

**Dataset:** [`mteb/JaqketRetrievalLite`](https://huggingface.co/datasets/mteb/JaqketRetrievalLite) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/kumapo/JAQKET-dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb_lite,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
    and Kawahara, Daisuke},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
      title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
      year = {2025},
    }
    
    @inproceedings{Kurihara_nlp2020,
      author = {鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也},
      booktitle = {言語処理学会第26回年次大会},
      note = {in Japanese},
      title = {JAQKET: クイズを題材にした日本語 QA データセットの構築},
      url = {https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf},
      year = {2020},
    }
    
    ```
    



#### `Ko-StrategyQA` { .model-copy }

Ko-StrategyQA

**Dataset:** [`taeminlee/Ko-StrategyQA`](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.02235)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kor | Encyclopaedic, Written | human-annotated | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{geva2021strategyqa,
      author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
      journal = {Transactions of the Association for Computational Linguistics (TACL)},
      title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
      year = {2021},
    }
    
    ```
    



#### `LEMBNarrativeQARetrieval` { .model-copy }

narrativeqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{kocisky-etal-2018-narrativeqa,
      address = {Cambridge, MA},
      author = {Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
    Schwarz, Jonathan  and
    Blunsom, Phil  and
    Dyer, Chris  and
    Hermann, Karl Moritz  and
    Melis, G{\'a}bor  and
    Grefenstette, Edward},
      doi = {10.1162/tacl_a_00023},
      editor = {Lee, Lillian  and
    Johnson, Mark  and
    Toutanova, Kristina  and
    Roark, Brian},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {317--328},
      publisher = {MIT Press},
      title = {The {N}arrative{QA} Reading Comprehension Challenge},
      url = {https://aclanthology.org/Q18-1023},
      volume = {6},
      year = {2018},
    }
    
    ```
    



#### `LEMBNeedleRetrieval` { .model-copy }

needle subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Blog, Written | derived | found | ndcg_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhu2024longembed,
      author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
      journal = {arXiv preprint arXiv:2404.12096},
      title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
      year = {2024},
    }
    
    ```
    



#### `LEMBPasskeyRetrieval` { .model-copy }

passkey subset of dwzhu/LongEmbed dataset.

**Dataset:** [`mteb/LEMBPasskeyRetrieval`](https://huggingface.co/datasets/mteb/LEMBPasskeyRetrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction, Written | derived | found | ndcg_at_1 |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhu2024longembed,
      author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
      journal = {arXiv preprint arXiv:2404.12096},
      title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
      year = {2024},
    }
    
    ```
    



#### `LEMBQMSumRetrieval` { .model-copy }

qmsum subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Spoken, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{zhong-etal-2021-qmsum,
      address = {Online},
      author = {Zhong, Ming  and
    Yin, Da  and
    Yu, Tao  and
    Zaidi, Ahmad  and
    Mutuma, Mutethia  and
    Jha, Rahul  and
    Awadallah, Ahmed Hassan  and
    Celikyilmaz, Asli  and
    Liu, Yang  and
    Qiu, Xipeng  and
    Radev, Dragomir},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2021.naacl-main.472},
      editor = {Toutanova, Kristina  and
    Rumshisky, Anna  and
    Zettlemoyer, Luke  and
    Hakkani-Tur, Dilek  and
    Beltagy, Iz  and
    Bethard, Steven  and
    Cotterell, Ryan  and
    Chakraborty, Tanmoy  and
    Zhou, Yichao},
      month = jun,
      pages = {5905--5921},
      publisher = {Association for Computational Linguistics},
      title = {{QMS}um: A New Benchmark for Query-based Multi-domain Meeting Summarization},
      url = {https://aclanthology.org/2021.naacl-main.472},
      year = {2021},
    }
    
    ```
    



#### `LEMBSummScreenFDRetrieval` { .model-copy }

summ_screen_fd subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Spoken, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{chen-etal-2022-summscreen,
      address = {Dublin, Ireland},
      author = {Chen, Mingda  and
    Chu, Zewei  and
    Wiseman, Sam  and
    Gimpel, Kevin},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/2022.acl-long.589},
      editor = {Muresan, Smaranda  and
    Nakov, Preslav  and
    Villavicencio, Aline},
      month = may,
      pages = {8602--8615},
      publisher = {Association for Computational Linguistics},
      title = {{S}umm{S}creen: A Dataset for Abstractive Screenplay Summarization},
      url = {https://aclanthology.org/2022.acl-long.589},
      year = {2022},
    }
    
    ```
    



#### `LEMBWikimQARetrieval` { .model-copy }

2wikimqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{ho2020constructing,
      author = {Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      pages = {6609--6625},
      title = {Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
      year = {2020},
    }
    
    ```
    



#### `LIMITRetrieval` { .model-copy }

A simple retrieval task designed to test all combinations of top-2 documents. This version includes all 50k docs.

**Dataset:** [`orionweller/LIMIT`](https://huggingface.co/datasets/orionweller/LIMIT) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction | human-annotated | created | recall_at_2 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2025theoreticallimit,
      archiveprefix = {arXiv},
      author = {Orion Weller and Michael Boratko and Iftekhar Naim and Jinhyuk Lee},
      eprint = {2508.21038},
      primaryclass = {cs.IR},
      title = {On the Theoretical Limitations of Embedding-Based Retrieval},
      url = {https://arxiv.org/abs/2508.21038},
      year = {2025},
    }
    ```
    



#### `LIMITSmallRetrieval` { .model-copy }

A simple retrieval task designed to test all combinations of top-2 documents. This version only includes the 46 documents that are relevant to the 1000 queries.

**Dataset:** [`orionweller/LIMIT-small`](https://huggingface.co/datasets/orionweller/LIMIT-small) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction | human-annotated | created | recall_at_2 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2025theoreticallimit,
      archiveprefix = {arXiv},
      author = {Orion Weller and Michael Boratko and Iftekhar Naim and Jinhyuk Lee},
      eprint = {2508.21038},
      primaryclass = {cs.IR},
      title = {On the Theoretical Limitations of Embedding-Based Retrieval},
      url = {https://arxiv.org/abs/2508.21038},
      year = {2025},
    }
    ```
    



#### `LawIRKo` { .model-copy }

This dataset assesses a model's ability to retrieve relevant legal articles from queries referencing specific Korean laws and provisions. The corpus comprises official legal texts including statutes, acts, and regulations, with each document representing a single article. Queries are derived from law titles paired and article identifiers. For instance the law title might be "건축법" (Building Act) and the article name "기술적 기준" (Technical Standards), which would become "건축법에 명시된 법률 중에 '기술적 기준'에 대해 설명하고 있는 세부 항목은 무엇입니까?" ("Which specific articles in the Building Act explain the 'technical standards'?").

**Dataset:** [`on-and-on/lawgov_ir-ko`](https://huggingface.co/datasets/on-and-on/lawgov_ir-ko) • **License:** mit • [Learn more →](https://huggingface.co/datasets/on-and-on/lawgov_ir-ko)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kor | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{law_ko_ir_khee,
      author = {kang-hyeun Lee},
      howpublished = {\url{https://huggingface.co/datasets/on-and-on/lawgov_ir-ko}},
      note = {A Benchmark Dataset for Korean Legal Information Retrieval and QA},
      year = {2026},
    }
    
    ```
    



#### `LeCaRDv2` { .model-copy }

The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.

**Dataset:** [`mteb/LeCaRDv2`](https://huggingface.co/datasets/mteb/LeCaRDv2) • **License:** mit • [Learn more →](https://github.com/THUIR/LeCaRDv2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | zho | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2023lecardv2,
      archiveprefix = {arXiv},
      author = {Haitao Li and Yunqiu Shao and Yueyue Wu and Qingyao Ai and Yixiao Ma and Yiqun Liu},
      eprint = {2310.17609},
      primaryclass = {cs.CL},
      title = {LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset},
      year = {2023},
    }
    
    ```
    



#### `LegalBenchConsumerContractsQA` { .model-copy }

The dataset includes questions and answers related to contracts.

**Dataset:** [`mteb/legalbench_consumer_contracts_qa`](https://huggingface.co/datasets/mteb/legalbench_consumer_contracts_qa) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{hendrycks2021cuad,
      author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
      journal = {arXiv preprint arXiv:2103.06268},
      title = {Cuad: An expert-annotated nlp dataset for legal contract review},
      year = {2021},
    }
    
    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }
    
    ```
    



#### `LegalBenchCorporateLobbying` { .model-copy }

The dataset includes bill titles and bill summaries related to corporate lobbying.

**Dataset:** [`mteb/legalbench_corporate_lobbying`](https://huggingface.co/datasets/mteb/legalbench_corporate_lobbying) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }
    
    @article{hendrycks2021cuad,
      author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
      journal = {arXiv preprint arXiv:2103.06268},
      title = {Cuad: An expert-annotated nlp dataset for legal contract review},
      year = {2021},
    }
    
    @article{holzenberger2021factoring,
      author = {Holzenberger, Nils and Van Durme, Benjamin},
      journal = {arXiv preprint arXiv:2105.07903},
      title = {Factoring statutory reasoning as language understanding challenges},
      year = {2021},
    }
    
    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }
    
    @article{lippi2019claudette,
      author = {Lippi, Marco and Pa{\l}ka, Przemys{\l}aw and Contissa, Giuseppe and Lagioia, Francesca and Micklitz, Hans-Wolfgang and Sartor, Giovanni and Torroni, Paolo},
      journal = {Artificial Intelligence and Law},
      pages = {117--139},
      publisher = {Springer},
      title = {CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service},
      volume = {27},
      year = {2019},
    }
    
    @article{ravichander2019question,
      author = {Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
      journal = {arXiv preprint arXiv:1911.00841},
      title = {Question answering for privacy policies: Combining computational and legal perspectives},
      year = {2019},
    }
    
    @article{wang2023maud,
      author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
      journal = {arXiv preprint arXiv:2301.00876},
      title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
      year = {2023},
    }
    
    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }
    
    @inproceedings{zheng2021does,
      author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
      booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
      pages = {159--168},
      title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
      year = {2021},
    }
    
    @article{zimmeck2019maps,
      author = {Zimmeck, Sebastian and Story, Peter and Smullen, Daniel and Ravichander, Abhilasha and Wang, Ziqi and Reidenberg, Joel R and Russell, N Cameron and Sadeh, Norman},
      journal = {Proc. Priv. Enhancing Tech.},
      pages = {66},
      title = {Maps: Scaling privacy compliance analysis to a million apps},
      volume = {2019},
      year = {2019},
    }
    
    ```
    



#### `LegalQANLRetrieval` { .model-copy }

To this end, we create and publish a Dutch legal QA dataset, consisting of question-answer pairs with attributions to Dutch law articles.

**Dataset:** [`clips/mteb-nl-legalqa-pr`](https://huggingface.co/datasets/clips/mteb-nl-legalqa-pr) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2024.nllp-1.12/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Legal, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{redelaar2024attributed,
      author = {Redelaar, Felicia and Van Drie, Romy and Verberne, Suzan and De Boer, Maaike},
      booktitle = {Proceedings of the natural legal language processing workshop 2024},
      pages = {154--165},
      title = {Attributed Question Answering for Preconditions in the Dutch Law},
      year = {2024},
    }
    
    ```
    



#### `LegalQuAD` { .model-copy }

The dataset consists of questions and legal documents in German.

**Dataset:** [`mteb/LegalQuAD`](https://huggingface.co/datasets/mteb/LegalQuAD) • **License:** cc-by-4.0 • [Learn more →](https://github.com/Christoph911/AIKE2021_Appendix)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{9723721,
      author = {Hoppe, Christoph and Pelkmann, David and Migenda, Nico and Hötte, Daniel and Schenck, Wolfram},
      booktitle = {2021 IEEE Fourth International Conference on Artificial Intelligence and Knowledge Engineering (AIKE)},
      doi = {10.1109/AIKE52691.2021.00011},
      keywords = {Knowledge engineering;Law;Semantic search;Conferences;Bit error rate;NLP;knowledge extraction;question-answering;semantic search;document retrieval;German language},
      number = {},
      pages = {29-32},
      title = {Towards Intelligent Legal Advisors for Document Retrieval and Question-Answering in German Legal Documents},
      volume = {},
      year = {2021},
    }
    
    ```
    



#### `LegalSummarization` { .model-copy }

The dataset consists of 439 pairs of contracts and their summarizations from https://tldrlegal.com and https://tosdr.org/.

**Dataset:** [`mteb/legal_summarization`](https://huggingface.co/datasets/mteb/legal_summarization) • **License:** apache-2.0 • [Learn more →](https://github.com/lauramanor/legal_summarization)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Legal, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{manor-li-2019-plain,
      address = {Minneapolis, Minnesota},
      author = {Manor, Laura  and
    Li, Junyi Jessy},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2019},
      month = jun,
      pages = {1--11},
      publisher = {Association for Computational Linguistics},
      title = {Plain {E}nglish Summarization of Contracts},
      url = {https://www.aclweb.org/anthology/W19-2201},
      year = {2019},
    }
    
    ```
    



#### `LitSearchRetrieval` { .model-copy }

The dataset contains the query set and retrieval corpus for the paper LitSearch: A Retrieval Benchmark for Scientific Literature Search. It introduces LitSearch, a retrieval benchmark comprising 597 realistic literature search queries about recent ML and NLP papers. LitSearch is constructed using a combination of (1) questions generated by GPT-4 based on paragraphs containing inline citations from research papers and (2) questions about recently published papers, manually written by their authors. All LitSearch questions were manually examined or edited by experts to ensure high quality.

**Dataset:** [`princeton-nlp/LitSearch`](https://huggingface.co/datasets/princeton-nlp/LitSearch) • **License:** mit • [Learn more →](https://github.com/princeton-nlp/LitSearch)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | LM-generated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{ajith2024litsearch,
      author = {Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu},
      title = {LitSearch: A Retrieval Benchmark for Scientific Literature Search},
      year = {2024},
    }
    
    ```
    



#### `LoTTE` { .model-copy }

LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing a challenging out-of-domain generalization benchmark.

**Dataset:** [`mteb/LoTTE`](https://huggingface.co/datasets/mteb/LoTTE) • **License:** mit • [Learn more →](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Social, Web | derived | found | hit_rate_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{santhanam-etal-2022-colbertv2,
      address = {Seattle, United States},
      author = {Santhanam, Keshav  and
    Khattab, Omar  and
    Saad-Falcon, Jon  and
    Potts, Christopher  and
    Zaharia, Matei},
      booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2022.naacl-main.272},
      editor = {Carpuat, Marine  and
    de Marneffe, Marie-Catherine  and
    Meza Ruiz, Ivan Vladimir},
      month = jul,
      pages = {3715--3734},
      publisher = {Association for Computational Linguistics},
      title = {{C}ol{BERT}v2: Effective and Efficient Retrieval via Lightweight Late Interaction},
      url = {https://aclanthology.org/2022.naacl-main.272/},
      year = {2022},
    }
    
    ```
    



#### `MBPPRetrieval` { .model-copy }

A code retrieval task based on 378 Python programming problems from MBPP (Mostly Basic Python Programming). Each query is a natural language description of a programming task (e.g., 'Write a function to find the shared elements from the given two lists'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper syntax and logic.

**Dataset:** [`embedding-benchmark/MBPP`](https://huggingface.co/datasets/embedding-benchmark/MBPP) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/MBPP)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, python | Programming | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{austin2021program,
      author = {Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
      journal = {arXiv preprint arXiv:2108.07732},
      title = {Program Synthesis with Large Language Models},
      year = {2021},
    }
    
    ```
    



#### `MIRACLJaRetrievalLite` { .model-copy }

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset. This is the lightweight Japanese version with a reduced corpus (105,064 documents) constructed using hard negatives from 5 high-performance models.

**Dataset:** [`mteb/MIRACLJaRetrievalLite`](https://huggingface.co/datasets/mteb/MIRACLJaRetrievalLite) • **License:** apache-2.0 • [Learn more →](https://project-miracl.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Written | expert-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David
    and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      volume = {11},
      year = {2023},
    }
    
    @misc{jmteb_lite,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
    and Kawahara, Daisuke},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
      title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
      year = {2025},
    }
    
    ```
    



#### `MIRACLRetrieval` { .model-copy }

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`mteb/MIRACLRetrieval`](https://huggingface.co/datasets/mteb/MIRACLRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      url = {https://doi.org/10.1162/tacl\_a\_00595},
      volume = {11},
      year = {2023},
    }
    
    ```
    



#### `MIRACLRetrievalHardNegatives` { .model-copy }

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MIRACLRetrievalHardNegatives`](https://huggingface.co/datasets/mteb/MIRACLRetrievalHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      url = {https://doi.org/10.1162/tacl\_a\_00595},
      volume = {11},
      year = {2023},
    }
    
    ```
    



#### `MIRACLRetrievalHardNegatives.v2` { .model-copy }

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/MIRACLRetrievalHardNegatives`](https://huggingface.co/datasets/mteb/MIRACLRetrievalHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      url = {https://doi.org/10.1162/tacl\_a\_00595},
      volume = {11},
      year = {2023},
    }
    
    ```
    



#### `MKQARetrieval` { .model-copy }

Multilingual Knowledge Questions & Answers (MKQA)contains 10,000 queries sampled from the Google Natural Questions dataset. For each query we collect new passage-independent answers. These queries and answers are then human translated into 25 Non-English languages.

**Dataset:** [`mteb/MKQARetrieval`](https://huggingface.co/datasets/mteb/MKQARetrieval) • **License:** cc-by-3.0 • [Learn more →](https://github.com/apple/ml-mkqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, dan, deu, eng, fin, ... (26) | Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{mkqa,
      author = {Shayne Longpre and Yi Lu and Joachim Daiber},
      title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
      url = {https://arxiv.org/pdf/2007.15207.pdf},
      year = {2020},
    }
    
    ```
    



#### `MLQARetrieval` { .model-copy }

MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance. MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between 4 different languages on average.

**Dataset:** [`mteb/MLQARetrieval`](https://huggingface.co/datasets/mteb/MLQARetrieval) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/mlqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, deu, eng, hin, spa, ... (7) | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{lewis2019mlqa,
      author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
      eid = {arXiv: 1910.07475},
      journal = {arXiv preprint arXiv:1910.07475},
      title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
      year = {2019},
    }
    
    ```
    



#### `MLQuestions` { .model-copy }

MLQuestions is a domain adaptation dataset for the machine learning domainIt consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)

**Dataset:** [`McGill-NLP/mlquestions`](https://huggingface.co/datasets/McGill-NLP/mlquestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/McGill-NLP/MLQuestions)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{kulshreshtha-etal-2021-back,
      address = {Online and Punta Cana, Dominican Republic},
      author = {Kulshreshtha, Devang  and
    Belfer, Robert  and
    Serban, Iulian Vlad  and
    Reddy, Siva},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      month = nov,
      pages = {7064--7078},
      publisher = {Association for Computational Linguistics},
      title = {Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval},
      url = {https://aclanthology.org/2021.emnlp-main.566},
      year = {2021},
    }
    
    ```
    



#### `MMarcoRetrieval` { .model-copy }

MMarcoRetrieval

**Dataset:** [`mteb/MMarcoRetrieval`](https://huggingface.co/datasets/mteb/MMarcoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2309.07597)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{xiao2024cpackpackagedresourcesadvance,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      url = {https://arxiv.org/abs/2309.07597},
      year = {2024},
    }
    
    ```
    



#### `MSMARCO` { .model-copy }

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/msmarco`](https://huggingface.co/datasets/mteb/msmarco) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### `MSMARCO-Fa` { .model-copy }

MSMARCO-Fa

**Dataset:** [`MCINext/msmarco-fa`](https://huggingface.co/datasets/MCINext/msmarco-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/msmarco-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `MSMARCO-FaHardNegatives` { .model-copy }

MSMARCO-FaHardNegatives

**Dataset:** [`MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `MSMARCO-PL` { .model-copy }

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/MSMARCO-PL`](https://huggingface.co/datasets/mteb/MSMARCO-PL) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `MSMARCO-PLHardNegatives` { .model-copy }

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO-PLHardNegatives`](https://huggingface.co/datasets/mteb/MSMARCO-PLHardNegatives) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `MSMARCO-VN` { .model-copy }

A translated dataset from MS MARCO is a collection of datasets focused on deep learning in search The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/msmarco-vn`](https://huggingface.co/datasets/GreenNode/msmarco-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `MSMARCOHardNegatives` { .model-copy }

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/MSMARCO_test_top_250_only_w_correct-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### `MSMARCOv2` { .model-copy }

MS MARCO is a collection of datasets focused on deep learning in search. This version is derived from BEIR

**Dataset:** [`mteb/msmarco-v2`](https://huggingface.co/datasets/mteb/msmarco-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### `MedicalQARetrieval` { .model-copy }

The dataset consists 2048 medical question and answer pairs.

**Dataset:** [`mteb/medical_qa`](https://huggingface.co/datasets/mteb/medical_qa) • **License:** cc0-1.0 • [Learn more →](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{BenAbacha-BMC-2019,
      author = {Asma, Ben Abacha and Dina, Demner{-}Fushman},
      journal = {{BMC} Bioinform.},
      number = {1},
      pages = {511:1--511:23},
      title = {A Question-Entailment Approach to Question Answering},
      url = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4},
      volume = {20},
      year = {2019},
    }
    
    ```
    



#### `MedicalRetrieval` { .model-copy }

MedicalRetrieval

**Dataset:** [`mteb/MedicalRetrieval`](https://huggingface.co/datasets/mteb/MedicalRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Medical, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### `MintakaRetrieval` { .model-copy }

We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers. 

**Dataset:** [`mteb/MintakaRetrieval`](https://huggingface.co/datasets/mteb/MintakaRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2022.coling-1.138)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, deu, fra, hin, ita, ... (8) | Encyclopaedic, Written | derived | human-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sen-etal-2022-mintaka,
      address = {Gyeongju, Republic of Korea},
      author = {Sen, Priyanka  and
    Aji, Alham Fikri  and
    Saffari, Amir},
      booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
      month = oct,
      pages = {1604--1619},
      publisher = {International Committee on Computational Linguistics},
      title = {Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering},
      url = {https://aclanthology.org/2022.coling-1.138},
      year = {2022},
    }
    
    ```
    



#### `MrTidyRetrieval` { .model-copy }

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.

**Dataset:** [`mteb/mrtidy`](https://huggingface.co/datasets/mteb/mrtidy) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/castorini/mr-tydi)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, ben, eng, fin, ind, ... (11) | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{mrtydi,
      author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
      journal = {arXiv:2108.08787},
      title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
      year = {2021},
    }
    
    ```
    



#### `MrTyDiJaRetrievalLite` { .model-copy }

Mr.TyDi is a multilingual benchmark dataset built on TyDi for document retrieval tasks. This is the lightweight Japanese version with a reduced corpus (93,382 documents) constructed using hard negatives from 5 high-performance models.

**Dataset:** [`mteb/MrTyDiJaRetrievalLite`](https://huggingface.co/datasets/mteb/MrTyDiJaRetrievalLite) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/castorini/mr-tydi)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb_lite,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
    and Kawahara, Daisuke},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
      title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
      year = {2025},
    }
    
    @article{mrtydi,
      author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
      journal = {arXiv:2108.08787},
      title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
      year = {2021},
    }
    
    ```
    



#### `MultiLongDocRetrieval` { .model-copy }

Multi Long Doc Retrieval (MLDR) 'is curated by the multilingual articles from Wikipedia, Wudao and mC4 (see Table 7), and NarrativeQA (Kocˇisky ́ et al., 2018; Gu ̈nther et al., 2023), which is only for English.' (Chen et al., 2024). It is constructed by sampling lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.

**Dataset:** [`mteb/MultiLongDocRetrieval`](https://huggingface.co/datasets/mteb/MultiLongDocRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.03216)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, cmn, deu, eng, fra, ... (13) | Encyclopaedic, Fiction, Non-fiction, Web, Written | LM-generated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{bge-m3,
      archiveprefix = {arXiv},
      author = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      eprint = {2402.03216},
      primaryclass = {cs.CL},
      title = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
      year = {2024},
    }
    
    ```
    



#### `NFCorpus` { .model-copy }

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nfcorpus`](https://huggingface.co/datasets/mteb/nfcorpus) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{boteva2016,
      author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
      city = {Padova},
      country = {Italy},
      journal = {Proceedings of the 38th European Conference on Information Retrieval},
      journal-abbrev = {ECIR},
      title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
      url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
      year = {2016},
    }
    
    ```
    



#### `NFCorpus-Fa` { .model-copy }

NFCorpus-Fa

**Dataset:** [`MCINext/nfcorpus-fa`](https://huggingface.co/datasets/MCINext/nfcorpus-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nfcorpus-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Medical | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `NFCorpus-NL` { .model-copy }

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-nfcorpus`](https://huggingface.co/datasets/clips/beir-nl-nfcorpus) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nfcorpus)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Medical, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `NFCorpus-NL.v2` { .model-copy }

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is a Dutch translation. This version adds a Dutch prompt to the dataset.

**Dataset:** [`clips/beir-nl-nfcorpus`](https://huggingface.co/datasets/clips/beir-nl-nfcorpus) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nfcorpus)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Medical, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `NFCorpus-PL` { .model-copy }

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/NFCorpus-PL`](https://huggingface.co/datasets/mteb/NFCorpus-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Medical, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `NFCorpus-VN` { .model-copy }

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nfcorpus-vn`](https://huggingface.co/datasets/GreenNode/nfcorpus-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Medical, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NLPJournalAbsArticleRetrieval` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`mteb/NLPJournalAbsArticleRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalAbsArticleRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalAbsArticleRetrieval.V2` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`mteb/NLPJournalAbsArticleRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalAbsArticleRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalAbsIntroRetrieval` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V1 dataset (last update 2020-06-15).

**Dataset:** [`mteb/NLPJournalAbsIntroRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalAbsIntroRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalAbsIntroRetrieval.V2` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V2 dataset (last update 2025-06-15).

**Dataset:** [`mteb/NLPJournalAbsIntroRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalAbsIntroRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalTitleAbsRetrieval` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`mteb/NLPJournalTitleAbsRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalTitleAbsRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalTitleAbsRetrieval.V2` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`mteb/NLPJournalTitleAbsRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalTitleAbsRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalTitleIntroRetrieval` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`mteb/NLPJournalTitleIntroRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalTitleIntroRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NLPJournalTitleIntroRetrieval.V2` { .model-copy }

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`mteb/NLPJournalTitleIntroRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalTitleIntroRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### `NQ` { .model-copy }

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nq`](https://huggingface.co/datasets/mteb/nq) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational
    Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    ```
    



#### `NQ-Fa` { .model-copy }

NQ-Fa

**Dataset:** [`MCINext/nq-fa`](https://huggingface.co/datasets/MCINext/nq-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nq-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `NQ-FaHardNegatives` { .model-copy }

NQ-FaHardNegatives

**Dataset:** [`MCINext/NQ_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/NQ_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/NQ_FA_test_top_250_only_w_correct-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Encyclopaedic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `NQ-NL` { .model-copy }

NQ-NL is a translation of NQ

**Dataset:** [`clips/beir-nl-nq`](https://huggingface.co/datasets/clips/beir-nl-nq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nq)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Encyclopaedic, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `NQ-PL` { .model-copy }

Natural Questions: A Benchmark for Question Answering Research

**Dataset:** [`mteb/NQ-PL`](https://huggingface.co/datasets/mteb/NQ-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Encyclopaedic, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `NQ-PLHardNegatives` { .model-copy }

Natural Questions: A Benchmark for Question Answering Research. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ-PLHardNegatives`](https://huggingface.co/datasets/mteb/NQ-PLHardNegatives) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Encyclopaedic, Written | human-annotated | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `NQ-VN` { .model-copy }

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nq-vn`](https://huggingface.co/datasets/GreenNode/nq-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NQHardNegatives` { .model-copy }

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/NQ_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational
    Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    ```
    



#### `NanoArguAnaRetrieval` { .model-copy }

NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.

**Dataset:** [`zeta-alpha-ai/NanoArguAna`](https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna) • **License:** cc-by-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Social, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wachsmuth2018retrieval,
      author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
      booktitle = {ACL},
      title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
      year = {2018},
    }
    
    ```
    



#### `NanoClimateFEVER-VN` { .model-copy }

NanoClimateFEVERVN is a small version of A translated dataset from CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nano-climate-fever-vn`](https://huggingface.co/datasets/GreenNode/nano-climate-fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NanoClimateFeverRetrieval` { .model-copy }

NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.

**Dataset:** [`zeta-alpha-ai/NanoClimateFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2012.00614)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, News, Non-fiction | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### `NanoDBPedia-VN` { .model-copy }

NanoDBPediaVN is a small version of A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nano-dbpedia-vn`](https://huggingface.co/datasets/GreenNode/nano-dbpedia-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NanoDBPediaRetrieval` { .model-copy }

NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.

**Dataset:** [`zeta-alpha-ai/NanoDBPedia`](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{lehmann2015dbpedia,
      author = {Lehmann, Jens and et al.},
      journal = {Semantic Web},
      title = {DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia},
      year = {2015},
    }
    
    ```
    



#### `NanoFEVER-VN` { .model-copy }

NanoFEVERVN is a small version of A translated dataset from FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nano-fever-vn`](https://huggingface.co/datasets/GreenNode/nano-fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://fever.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NanoFEVERRetrieval` { .model-copy }

NanoFEVER is a smaller version of FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`zeta-alpha-ai/NanoFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER) • **License:** cc-by-4.0 • [Learn more →](https://fever.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Encyclopaedic | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### `NanoFiQA2018Retrieval` { .model-copy }

NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset.

**Dataset:** [`zeta-alpha-ai/NanoFiQA2018`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018) • **License:** cc-by-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Social | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thakur2021beir,
      author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
      url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
      year = {2021},
    }
    
    ```
    



#### `NanoHotpotQA-VN` { .model-copy }

NanoHotpotQAVN is a small version of A translated dataset from HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nano-hotpotqa-vn`](https://huggingface.co/datasets/GreenNode/nano-hotpotqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NanoHotpotQARetrieval` { .model-copy }

NanoHotpotQARetrieval is a smaller subset of the HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`zeta-alpha-ai/NanoHotpotQA`](https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA) • **License:** cc-by-4.0 • [Learn more →](https://hotpotqa.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### `NanoMSMARCO-VN` { .model-copy }

NanoMSMARCOVN is a small version of A translated dataset from MS MARCO is a collection of datasets focused on deep learning in search The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nano-msmarco-vn`](https://huggingface.co/datasets/GreenNode/nano-msmarco-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NanoMSMARCORetrieval` { .model-copy }

NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.

**Dataset:** [`zeta-alpha-ai/NanoMSMARCO`](https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO) • **License:** cc-by-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### `NanoNFCorpusRetrieval` { .model-copy }

NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.

**Dataset:** [`zeta-alpha-ai/NanoNFCorpus`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus) • **License:** cc-by-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{boteva2016,
      author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
      city = {Padova},
      country = {Italy},
      journal = {Proceedings of the 38th European Conference on Information Retrieval},
      journal-abbrev = {ECIR},
      title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
      url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
      year = {2016},
    }
    
    ```
    



#### `NanoNQ-VN` { .model-copy }

NanoNQVN is a small version of A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nano-nq-vn`](https://huggingface.co/datasets/GreenNode/nano-nq-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `NanoNQRetrieval` { .model-copy }

NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.

**Dataset:** [`zeta-alpha-ai/NanoNQ`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ) • **License:** cc-by-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Web | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational
    Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    ```
    



#### `NanoQuoraRetrieval` { .model-copy }

NanoQuoraRetrieval is a smaller subset of the QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`zeta-alpha-ai/NanoQuoraRetrieval`](https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Social | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### `NanoSCIDOCSRetrieval` { .model-copy }

NanoFiQA2018 is a smaller subset of SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`zeta-alpha-ai/NanoSCIDOCS`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### `NanoSciFactRetrieval` { .model-copy }

NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`zeta-alpha-ai/NanoSciFact`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact) • **License:** cc-by-4.0 • [Learn more →](https://github.com/allenai/scifact)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### `NanoTouche2020Retrieval` { .model-copy }

NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.

**Dataset:** [`zeta-alpha-ai/NanoTouche2020`](https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020) • **License:** cc-by-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{potthast_2022_6862281,
      author = {Potthast, Martin and
    Gienapp, Lukas and
    Wachsmuth, Henning and
    Hagen, Matthias and
    Fröbe, Maik and
    Bondarenko, Alexander and
    Ajjour, Yamen and
    Stein, Benno},
      doi = {10.5281/zenodo.6862281},
      month = jul,
      publisher = {Zenodo},
      title = {{Touché20-Argument-Retrieval-for-Controversial-
    Questions}},
      url = {https://doi.org/10.5281/zenodo.6862281},
      year = {2022},
    }
    
    ```
    



#### `NarrativeQARetrieval` { .model-copy }

NarrativeQA is a dataset for the task of question answering on long narratives. It consists of realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. 

**Dataset:** [`deepmind/narrativeqa`](https://huggingface.co/datasets/deepmind/narrativeqa) • **License:** apache-2.0 • [Learn more →](https://metatext.io/datasets/narrativeqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Fiction, Non-fiction, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{kocisky-etal-2018-narrativeqa,
      address = {Cambridge, MA},
      author = {Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
    Schwarz, Jonathan  and
    Blunsom, Phil  and
    Dyer, Chris  and
    Hermann, Karl Moritz  and
    Melis, G{\'a}bor  and
    Grefenstette, Edward},
      doi = {10.1162/tacl_a_00023},
      editor = {Lee, Lillian  and
    Johnson, Mark  and
    Toutanova, Kristina  and
    Roark, Brian},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {317--328},
      publisher = {MIT Press},
      title = {The {N}arrative{QA} Reading Comprehension Challenge},
      url = {https://aclanthology.org/Q18-1023},
      volume = {6},
      year = {2018},
    }
    
    ```
    



#### `NeuCLIR2022Retrieval` { .model-copy }

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/NeuCLIR2022Retrieval`](https://huggingface.co/datasets/mteb/NeuCLIR2022Retrieval) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas, rus, zho | News, Written | expert-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @article{lawrie2023overview,
      author = {Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
      journal = {arXiv preprint arXiv:2304.12367},
      title = {Overview of the TREC 2022 NeuCLIR track},
      year = {2023},
    }
    
    ```
    



#### `NeuCLIR2022RetrievalHardNegatives` { .model-copy }

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NeuCLIR2022RetrievalHardNegatives`](https://huggingface.co/datasets/mteb/NeuCLIR2022RetrievalHardNegatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas, rus, zho | News, Written | expert-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @article{lawrie2023overview,
      author = {Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
      journal = {arXiv preprint arXiv:2304.12367},
      title = {Overview of the TREC 2022 NeuCLIR track},
      year = {2023},
    }
    
    ```
    



#### `NeuCLIR2023Retrieval` { .model-copy }

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/NeuCLIR2022Retrieval`](https://huggingface.co/datasets/mteb/NeuCLIR2022Retrieval) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas, rus, zho | News, Written | expert-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lawrie2024overview,
      archiveprefix = {arXiv},
      author = {Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
      eprint = {2404.08071},
      primaryclass = {cs.IR},
      title = {Overview of the TREC 2023 NeuCLIR Track},
      year = {2024},
    }
    
    ```
    



#### `NeuCLIR2023RetrievalHardNegatives` { .model-copy }

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NeuCLIR2023RetrievalHardNegatives`](https://huggingface.co/datasets/mteb/NeuCLIR2023RetrievalHardNegatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas, rus, zho | News, Written | expert-annotated | found | ndcg_at_20 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lawrie2024overview,
      archiveprefix = {arXiv},
      author = {Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
      eprint = {2404.08071},
      primaryclass = {cs.IR},
      title = {Overview of the TREC 2023 NeuCLIR Track},
      year = {2024},
    }
    
    ```
    



#### `NorQuadRetrieval` { .model-copy }

Human-created question for Norwegian wikipedia passages.

**Dataset:** [`mteb/norquad_retrieval`](https://huggingface.co/datasets/mteb/norquad_retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.17/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nob | Encyclopaedic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{ivanova-etal-2023-norquad,
      address = {T{\'o}rshavn, Faroe Islands},
      author = {Ivanova, Sardana  and
    Andreassen, Fredrik  and
    Jentoft, Matias  and
    Wold, Sondre  and
    {\O}vrelid, Lilja},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Alum{\"a}e, Tanel  and
    Fishel, Mark},
      month = may,
      pages = {159--168},
      publisher = {University of Tartu Library},
      title = {{N}or{Q}u{AD}: {N}orwegian Question Answering Dataset},
      url = {https://aclanthology.org/2023.nodalida-1.17},
      year = {2023},
    }
    
    ```
    



#### `OpenTenderRetrieval` { .model-copy }

This dataset contains Belgian and Dutch tender calls from OpenTender in Dutch

**Dataset:** [`clips/mteb-nl-opentender-ret`](https://huggingface.co/datasets/clips/mteb-nl-opentender-ret) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2509.12340)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Government, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2025mtebnle5nlembeddingbenchmark,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      eprint = {2509.12340},
      primaryclass = {cs.CL},
      title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
      url = {https://arxiv.org/abs/2509.12340},
      year = {2025},
    }
    
    ```
    



#### `PIQA` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.

**Dataset:** [`mteb/PIQA`](https://huggingface.co/datasets/mteb/PIQA) • **License:** afl-3.0 • [Learn more →](https://arxiv.org/abs/1911.11641)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{bisk2020piqa,
      author = {Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
      booktitle = {Proceedings of the AAAI conference on artificial intelligence},
      number = {05},
      pages = {7432--7439},
      title = {Piqa: Reasoning about physical commonsense in natural language},
      volume = {34},
      year = {2020},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `PUGGRetrieval` { .model-copy }

Information Retrieval PUGG dataset for the Polish language.

**Dataset:** [`clarin-pl/PUGG_IR`](https://huggingface.co/datasets/clarin-pl/PUGG_IR) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2024.findings-acl.652/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web | human-annotated | multiple | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sawczyn-etal-2024-developing,
      address = {Bangkok, Thailand},
      author = {Sawczyn, Albert  and
    Viarenich, Katsiaryna  and
    Wojtasik, Konrad  and
    Domoga{\l}a, Aleksandra  and
    Oleksy, Marcin  and
    Piasecki, Maciej  and
    Kajdanowicz, Tomasz},
      booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
      doi = {10.18653/v1/2024.findings-acl.652},
      editor = {Ku, Lun-Wei  and
    Martins, Andre  and
    Srikumar, Vivek},
      month = aug,
      pages = {10978--10996},
      publisher = {Association for Computational Linguistics},
      title = {Developing {PUGG} for {P}olish: A Modern Approach to {KBQA}, {MRC}, and {IR} Dataset Construction},
      url = {https://aclanthology.org/2024.findings-acl.652/},
      year = {2024},
    }
    
    ```
    



#### `PersianWebDocumentRetrieval` { .model-copy }

Persian dataset designed specifically for the task of text information retrieval through the web.

**Dataset:** [`MCINext/persian-web-document-retrieval`](https://huggingface.co/datasets/MCINext/persian-web-document-retrieval) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/10553090)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `PublicHealthQA` { .model-copy }

A multilingual dataset for public health question answering, based on FAQ sourced from CDC and WHO.

**Dataset:** [`xhluca/publichealth-qa`](https://huggingface.co/datasets/xhluca/publichealth-qa) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/xhluca/publichealth-qa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, eng, fra, kor, rus, ... (8) | Government, Medical, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{xing_han_lu_2024,
      author = { {Xing Han Lu} },
      doi = { 10.57967/hf/2247 },
      publisher = { Hugging Face },
      title = { publichealth-qa (Revision 3b67b6b) },
      url = { https://huggingface.co/datasets/xhluca/publichealth-qa },
      year = {2024},
    }
    
    ```
    



#### `Quail` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on Quail.

**Dataset:** [`mteb/Quail`](https://huggingface.co/datasets/mteb/Quail) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://text-machine.cs.uml.edu/lab2/projects/quail/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{rogers2020getting,
      author = {Rogers, Anna and Kovaleva, Olga and Downey, Matthew and Rumshisky, Anna},
      booktitle = {Proceedings of the AAAI conference on artificial intelligence},
      number = {05},
      pages = {8722--8731},
      title = {Getting closer to AI complete question answering: A set of prerequisite real tasks},
      volume = {34},
      year = {2020},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `Quora-NL` { .model-copy }

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. QuoraRetrieval-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-quora`](https://huggingface.co/datasets/clips/beir-nl-quora) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-quora)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `Quora-PL` { .model-copy }

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/Quora-PL`](https://huggingface.co/datasets/mteb/Quora-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `Quora-PLHardNegatives` { .model-copy }

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/Quora-PLHardNegatives`](https://huggingface.co/datasets/mteb/Quora-PLHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `Quora-VN` { .model-copy }

A translated dataset from QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/quora-vn`](https://huggingface.co/datasets/GreenNode/quora-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Blog, Web, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `QuoraRetrieval` { .model-copy }

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/quora`](https://huggingface.co/datasets/mteb/quora) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### `QuoraRetrieval-Fa` { .model-copy }

QuoraRetrieval-Fa

**Dataset:** [`MCINext/quora-fa`](https://huggingface.co/datasets/MCINext/quora-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/quora-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `QuoraRetrieval-Fa.v2` { .model-copy }

QuoraRetrieval-Fa.v2

**Dataset:** [`MCINext/quora-fa-v2`](https://huggingface.co/datasets/MCINext/quora-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/quora-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `QuoraRetrievalHardNegatives` { .model-copy }

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/QuoraRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/QuoraRetrieval_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### `QuoraRetrievalHardNegatives.v2` { .model-copy }

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/QuoraRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/QuoraRetrieval_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### `R2MEDBioinformaticsRetrieval` { .model-copy }

Bioinformatics retrieval dataset.

**Dataset:** [`R2MED/Bioinformatics`](https://huggingface.co/datasets/R2MED/Bioinformatics) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Bioinformatics)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDBiologyRetrieval` { .model-copy }

Biology retrieval dataset.

**Dataset:** [`R2MED/Biology`](https://huggingface.co/datasets/R2MED/Biology) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Biology)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDIIYiClinicalRetrieval` { .model-copy }

IIYi-Clinical retrieval dataset.

**Dataset:** [`R2MED/IIYi-Clinical`](https://huggingface.co/datasets/R2MED/IIYi-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/IIYi-Clinical)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDMedQADiagRetrieval` { .model-copy }

MedQA-Diag retrieval dataset.

**Dataset:** [`R2MED/MedQA-Diag`](https://huggingface.co/datasets/R2MED/MedQA-Diag) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedQA-Diag)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDMedXpertQAExamRetrieval` { .model-copy }

MedXpertQA-Exam retrieval dataset.

**Dataset:** [`R2MED/MedXpertQA-Exam`](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDMedicalSciencesRetrieval` { .model-copy }

Medical-Sciences retrieval dataset.

**Dataset:** [`R2MED/Medical-Sciences`](https://huggingface.co/datasets/R2MED/Medical-Sciences) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Medical-Sciences)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDPMCClinicalRetrieval` { .model-copy }

PMC-Clinical retrieval dataset.

**Dataset:** [`R2MED/PMC-Clinical`](https://huggingface.co/datasets/R2MED/PMC-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Clinical)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `R2MEDPMCTreatmentRetrieval` { .model-copy }

PMC-Treatment retrieval dataset.

**Dataset:** [`R2MED/PMC-Treatment`](https://huggingface.co/datasets/R2MED/PMC-Treatment) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Treatment)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | LM-generated and reviewed | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### `RARbCode` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.

**Dataset:** [`mteb/RARbCode`](https://huggingface.co/datasets/mteb/RARbCode) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.06347)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{husain2019codesearchnet,
      author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
      journal = {arXiv preprint arXiv:1909.09436},
      title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
      year = {2019},
    }
    
    @article{muennighoff2023octopack,
      author = {Muennighoff, Niklas and Liu, Qian and Zebaze, Armel and Zheng, Qinkai and Hui, Binyuan and Zhuo, Terry Yue and Singh, Swayam and Tang, Xiangru and Von Werra, Leandro and Longpre, Shayne},
      journal = {arXiv preprint arXiv:2308.07124},
      title = {Octopack: Instruction tuning code large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `RARbMath` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.

**Dataset:** [`mteb/RARbMath`](https://huggingface.co/datasets/mteb/RARbMath) • **License:** mit • [Learn more →](https://arxiv.org/abs/2404.06347)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{cobbe2021training,
      author = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
      journal = {arXiv preprint arXiv:2110.14168},
      title = {Training verifiers to solve math word problems},
      year = {2021},
    }
    
    @article{hendrycks2021measuring,
      author = {Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
      journal = {arXiv preprint arXiv:2103.03874},
      title = {Measuring mathematical problem solving with the math dataset},
      year = {2021},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    @article{yu2023metamath,
      author = {Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
      journal = {arXiv preprint arXiv:2309.12284},
      title = {Metamath: Bootstrap your own mathematical questions for large language models},
      year = {2023},
    }
    
    ```
    



#### `RiaNewsRetrieval` { .model-copy }

News article retrieval by headline. Based on Rossiya Segodnya dataset.

**Dataset:** [`ai-forever/ria-news-retrieval`](https://huggingface.co/datasets/ai-forever/ria-news-retrieval) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | News, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{gavrilov2018self,
      author = {Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
      booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
      title = {Self-Attentive Model for Headline Generation},
      year = {2019},
    }
    
    ```
    



#### `RiaNewsRetrievalHardNegatives` { .model-copy }

News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | News, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{gavrilov2018self,
      author = {Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
      booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
      title = {Self-Attentive Model for Headline Generation},
      year = {2019},
    }
    
    ```
    



#### `RiaNewsRetrievalHardNegatives.v2` { .model-copy }

News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)

**Dataset:** [`mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | News, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{gavrilov2018self,
      author = {Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
      booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
      title = {Self-Attentive Model for Headline Generation},
      year = {2019},
    }
    
    ```
    



#### `RuBQRetrieval` { .model-copy }

Paragraph retrieval based on RuBQ 2.0. Retrieve paragraphs from Wikipedia that answer the question.

**Dataset:** [`ai-forever/rubq-retrieval`](https://huggingface.co/datasets/ai-forever/rubq-retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | Encyclopaedic, Written | human-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{RuBQ2021,
      author = {Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
      booktitle = {ESWC},
      pages = {532--547},
      title = {RuBQ 2.0: An Innovated Russian Question Answering Dataset},
      year = {2021},
    }
    
    ```
    



#### `RuSciBenchCiteRetrieval` { .model-copy }

This task is focused on Direct Citation Prediction for scientific papers from eLibrary, Russia's largest electronic library of scientific publications. Given a query paper (title and abstract), the goal is to retrieve papers that are directly cited by it from a larger corpus of papers. The dataset for this task consists of 3,000 query papers, 15,000 relevant (cited) papers, and 75,000 irrelevant papers. The task is available for both Russian and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, rus | Academic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{vatolin2024ruscibench,
      author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
      doi = {10.1134/S1064562424602191},
      issn = {1531-8362},
      journal = {Doklady Mathematics},
      month = {12},
      number = {1},
      pages = {S251--S260},
      title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
      url = {https://doi.org/10.1134/S1064562424602191},
      volume = {110},
      year = {2024},
    }
    
    ```
    



#### `RuSciBenchCociteRetrieval` { .model-copy }

This task focuses on Co-citation Prediction for scientific papers from eLibrary, Russia's largest electronic library of scientific publications. Given a query paper (title and abstract), the goal is to retrieve other papers that are co-cited with it. Two papers are considered co-cited if they are both cited by at least 5 of the same other papers. Similar to the Direct Citation task, this task employs a retrieval setup: for a given query paper, all other papers in the corpus that are not co-cited with it are considered negative examples. The task is available for both Russian and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, rus | Academic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{vatolin2024ruscibench,
      author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
      doi = {10.1134/S1064562424602191},
      issn = {1531-8362},
      journal = {Doklady Mathematics},
      month = {12},
      number = {1},
      pages = {S251--S260},
      title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
      url = {https://doi.org/10.1134/S1064562424602191},
      volume = {110},
      year = {2024},
    }
    
    ```
    



#### `SCIDOCS` { .model-copy }

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/scidocs`](https://huggingface.co/datasets/mteb/scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### `SCIDOCS-Fa` { .model-copy }

SCIDOCS-Fa

**Dataset:** [`MCINext/scidocs-fa`](https://huggingface.co/datasets/MCINext/scidocs-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scidocs-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Academic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SCIDOCS-Fa.v2` { .model-copy }

SCIDOCS-Fa.v2

**Dataset:** [`MCINext/scidocs-fa-v2`](https://huggingface.co/datasets/MCINext/scidocs-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scidocs-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Academic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SCIDOCS-NL` { .model-copy }

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-scidocs`](https://huggingface.co/datasets/clips/beir-nl-scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `SCIDOCS-NL.v2` { .model-copy }

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch translation. This version adds a Dutch prompt to the dataset.

**Dataset:** [`clips/beir-nl-scidocs`](https://huggingface.co/datasets/clips/beir-nl-scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Non-fiction, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `SCIDOCS-PL` { .model-copy }

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/SCIDOCS-PL`](https://huggingface.co/datasets/mteb/SCIDOCS-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `SCIDOCS-VN` { .model-copy }

A translated dataset from SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scidocs-vn`](https://huggingface.co/datasets/GreenNode/scidocs-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `SIQA` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SIQA.

**Dataset:** [`mteb/SIQA`](https://huggingface.co/datasets/mteb/SIQA) • **License:** not specified • [Learn more →](https://leaderboard.allenai.org/socialiqa/submissions/get-started)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{sap2019socialiqa,
      author = {Sap, Maarten and Rashkin, Hannah and Chen, Derek and LeBras, Ronan and Choi, Yejin},
      journal = {arXiv preprint arXiv:1904.09728},
      title = {Socialiqa: Commonsense reasoning about social interactions},
      year = {2019},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `SKQuadRetrieval` { .model-copy }

Retrieval SK Quad evaluates Slovak search performance using questions and answers derived from the SK-QuAD dataset. It measures relevance with scores assigned to answers based on their relevancy to corresponding questions, which is vital for improving Slovak language search systems.

**Dataset:** [`TUKE-KEMT/retrieval-skquad`](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | slk | Encyclopaedic | human-annotated | found | ndcg_at_10 |



#### `SNLRetrieval` { .model-copy }

Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.

**Dataset:** [`adrlau/navjordj-SNL_summarization_copy`](https://huggingface.co/datasets/adrlau/navjordj-SNL_summarization_copy) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLRetrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nob | Encyclopaedic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### `SQuADKorV1Retrieval` { .model-copy }

Korean translation of SQuAD v1.0 dataset for retrieval task, based on Korean Wikipedia articles.

**Dataset:** [`yjoonjang/squad_kor_v1`](https://huggingface.co/datasets/yjoonjang/squad_kor_v1) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/yjoonjang/squad_kor_v1)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kor | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{rajpurkar-etal-2016-squad,
      address = {Austin, Texas},
      author = {Rajpurkar, Pranav  and
    Zhang, Jian  and
    Lopyrev, Konstantin  and
    Liang, Percy},
      booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D16-1264},
      editor = {Su, Jian  and
    Duh, Kevin  and
    Carreras, Xavier},
      month = nov,
      pages = {2383--2392},
      publisher = {Association for Computational Linguistics},
      title = {{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text},
      url = {https://aclanthology.org/D16-1264},
      year = {2016},
    }
    
    ```
    



#### `SadeemQuestionRetrieval` { .model-copy }

SadeemQuestion: A Benchmark Data Set for Community Question-Retrieval Research

**Dataset:** [`sadeem-ai/sadeem-ar-eval-retrieval-questions`](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara | Written, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sadeem-2024-ar-retrieval-questions,
      author = {abubakr.soliman@sadeem.app},
      title = {SadeemQuestionRetrieval: A New Benchmark for Arabic questions-based Articles Searching.},
    }
    
    ```
    



#### `SciFact` { .model-copy }

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact) • **License:** cc-by-nc-4.0 • [Learn more →](https://github.com/allenai/scifact)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### `SciFact-Fa` { .model-copy }

SciFact-Fa

**Dataset:** [`MCINext/scifact-fa`](https://huggingface.co/datasets/MCINext/scifact-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scifact-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Academic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SciFact-Fa.v2` { .model-copy }

SciFact-Fa.v2

**Dataset:** [`MCINext/scifact-fa-v2`](https://huggingface.co/datasets/MCINext/scifact-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scifact-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Academic | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SciFact-NL` { .model-copy }

SciFactNL verifies scientific claims in Dutch using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`clips/beir-nl-scifact`](https://huggingface.co/datasets/clips/beir-nl-scifact) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scifact)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Medical, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `SciFact-NL.v2` { .model-copy }

SciFactNL verifies scientific claims in Dutch using evidence from the research literature containing scientific paper abstracts. This version adds a Dutch prompt to the dataset.

**Dataset:** [`clips/beir-nl-scifact`](https://huggingface.co/datasets/clips/beir-nl-scifact) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scifact)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Medical, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `SciFact-PL` { .model-copy }

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/SciFact-PL`](https://huggingface.co/datasets/mteb/SciFact-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/allenai/scifact)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Medical, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `SciFact-VN` { .model-copy }

A translated dataset from SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scifact-vn`](https://huggingface.co/datasets/GreenNode/scifact-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/allenai/scifact)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Medical, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `SlovakSumRetrieval` { .model-copy }

SlovakSum, a Slovak news summarization dataset consisting of over 200 thousand news articles with titles and short abstracts obtained from multiple Slovak newspapers. Originally intended as a summarization task, but since no human annotations were provided here reformulated to a retrieval task.

**Dataset:** [`NaiveNeuron/slovaksum`](https://huggingface.co/datasets/NaiveNeuron/slovaksum) • **License:** openrail • [Learn more →](https://huggingface.co/datasets/NaiveNeuron/slovaksum)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | slk | News, Social, Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{OndrejowaSlovakSum24,
      author = {Ondrejová, Viktória and Šuppa, Marek},
      booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
      date = {2024},
      title = {SlovakSum: A Large Scale Slovak Summarization Dataset},
    }
    
    ```
    



#### `SpanishPassageRetrievalS2P` { .model-copy }

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`mteb/SpanishPassageRetrievalS2P`](https://huggingface.co/datasets/mteb/SpanishPassageRetrievalS2P) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | spa | Medical, Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{10.1007/978-3-030-15719-7_19,
      address = {Cham},
      author = {Kamateri, Eleni
    and Tsikrika, Theodora
    and Symeonidis, Spyridon
    and Vrochidis, Stefanos
    and Minker, Wolfgang
    and Kompatsiaris, Yiannis},
      booktitle = {Advances in Information Retrieval},
      editor = {Azzopardi, Leif
    and Stein, Benno
    and Fuhr, Norbert
    and Mayr, Philipp
    and Hauff, Claudia
    and Hiemstra, Djoerd},
      isbn = {978-3-030-15719-7},
      pages = {148--154},
      publisher = {Springer International Publishing},
      title = {A Test Collection for Passage Retrieval Evaluation of Spanish Health-Related Resources},
      year = {2019},
    }
    
    ```
    



#### `SpanishPassageRetrievalS2S` { .model-copy }

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`mteb/SpanishPassageRetrievalS2S`](https://huggingface.co/datasets/mteb/SpanishPassageRetrievalS2S) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | spa | Medical, Web, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{10.1007/978-3-030-15719-7_19,
      address = {Cham},
      author = {Kamateri, Eleni
    and Tsikrika, Theodora
    and Symeonidis, Spyridon
    and Vrochidis, Stefanos
    and Minker, Wolfgang
    and Kompatsiaris, Yiannis},
      booktitle = {Advances in Information Retrieval},
      editor = {Azzopardi, Leif
    and Stein, Benno
    and Fuhr, Norbert
    and Mayr, Philipp
    and Hauff, Claudia
    and Hiemstra, Djoerd},
      isbn = {978-3-030-15719-7},
      pages = {148--154},
      publisher = {Springer International Publishing},
      title = {A Test Collection for Passage Retrieval Evaluation of Spanish Health-Related Resources},
      year = {2019},
    }
    
    ```
    



#### `SpartQA` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SpartQA.

**Dataset:** [`mteb/SpartQA`](https://huggingface.co/datasets/mteb/SpartQA) • **License:** mit • [Learn more →](https://github.com/HLR/SpartQA_generation)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{mirzaee2021spartqa,
      author = {Mirzaee, Roshanak and Faghihi, Hossein Rajaby and Ning, Qiang and Kordjmashidi, Parisa},
      journal = {arXiv preprint arXiv:2104.05832},
      title = {Spartqa:: A textual question answering benchmark for spatial reasoning},
      year = {2021},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `StackOverflowQA` { .model-copy }

The dataset is a collection of natural language queries and their corresponding response which may include some text mixed with code snippets. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`mteb/StackOverflowQA`](https://huggingface.co/datasets/mteb/StackOverflowQA) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2024coircomprehensivebenchmarkcode,
      archiveprefix = {arXiv},
      author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
      eprint = {2407.02883},
      primaryclass = {cs.IR},
      title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      url = {https://arxiv.org/abs/2407.02883},
      year = {2024},
    }
    
    ```
    



#### `StatcanDialogueDatasetRetrieval` { .model-copy }

A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.

**Dataset:** [`mteb/StatcanDialogueDatasetRetrieval`](https://huggingface.co/datasets/mteb/StatcanDialogueDatasetRetrieval) • **License:** https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md • [Learn more →](https://mcgill-nlp.github.io/statcan-dialogue-dataset/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, fra | Government, Web, Written | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{lu-etal-2023-statcan,
      address = {Dubrovnik, Croatia},
      author = {Lu, Xing Han  and
    Reddy, Siva  and
    de Vries, Harm},
      booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
      month = may,
      pages = {2799--2829},
      publisher = {Association for Computational Linguistics},
      title = {The {S}tat{C}an Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents},
      url = {https://arxiv.org/abs/2304.01412},
      year = {2023},
    }
    
    ```
    



#### `SweFaqRetrieval` { .model-copy }

A Swedish QA dataset derived from FAQ

**Dataset:** [`mteb/SweFaqRetrieval`](https://huggingface.co/datasets/mteb/SweFaqRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | swe | Government, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{berdivcevskis2023superlim,
      author = {Berdi{\v{c}}evskis, Aleksandrs and Bouma, Gerlof and Kurtz, Robin and Morger, Felix and {\"O}hman, Joey and Adesam, Yvonne and Borin, Lars and Dann{\'e}lls, Dana and Forsberg, Markus and Isbister, Tim and others},
      booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages = {8137--8153},
      title = {Superlim: A Swedish language understanding evaluation benchmark},
      year = {2023},
    }
    
    ```
    



#### `SwednRetrieval` { .model-copy }

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure

**Dataset:** [`mteb/SwednRetrieval`](https://huggingface.co/datasets/mteb/SwednRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | swe | News, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    
    ```
    



#### `SynPerChatbotRAGFAQRetrieval` { .model-copy }

Synthetic Persian Chatbot RAG FAQ Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-faq-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | LM-generated | LM-generated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SynPerChatbotRAGTopicsRetrieval` { .model-copy }

Synthetic Persian Chatbot RAG Topics Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-topics-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | LM-generated | LM-generated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SynPerChatbotTopicsRetrieval` { .model-copy }

Synthetic Persian Chatbot Topics Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-topics-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | LM-generated | LM-generated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SynPerQARetrieval` { .model-copy }

Synthetic Persian QA Retrieval

**Dataset:** [`MCINext/synthetic-persian-qa-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Web | LM-generated | LM-generated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `SyntecRetrieval` { .model-copy }

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`lyon-nlp/mteb-fr-retrieval-syntec-s2p`](https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Legal, Written | human-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{ciancone2024extending,
      archiveprefix = {arXiv},
      author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      eprint = {2405.20468},
      primaryclass = {cs.CL},
      title = {Extending the Massive Text Embedding Benchmark to French},
      year = {2024},
    }
    
    ```
    



#### `SyntheticText2SQL` { .model-copy }

The dataset is a collection of natural language queries and their corresponding sql snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/synthetic-text2sql`](https://huggingface.co/datasets/CoIR-Retrieval/synthetic-text2sql) • **License:** mit • [Learn more →](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, sql | Programming, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @software{gretel-synthetic-text-to-sql-2024,
      author = {Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew},
      month = {April},
      title = {{Synthetic-Text-To-SQL}: A synthetic dataset for training language models to generate SQL queries from natural language prompts},
      url = {https://huggingface.co/datasets/gretelai/synthetic-text-to-sql},
      year = {2024},
    }
    
    ```
    



#### `T2Retrieval` { .model-copy }

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Retrieval`](https://huggingface.co/datasets/mteb/T2Retrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2304.03679)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Academic, Financial, Government, Medical, Non-fiction | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{xie2023t2ranking,
      archiveprefix = {arXiv},
      author = {Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
      eprint = {2304.03679},
      primaryclass = {cs.IR},
      title = {T2Ranking: A large-scale Chinese Benchmark for Passage Ranking},
      year = {2023},
    }
    
    ```
    



#### `TRECCOVID` { .model-copy }

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/trec-covid`](https://huggingface.co/datasets/mteb/trec-covid) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Medical, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{roberts2021searching,
      archiveprefix = {arXiv},
      author = {Kirk Roberts and Tasmeer Alam and Steven Bedrick and Dina Demner-Fushman and Kyle Lo and Ian Soboroff and Ellen Voorhees and Lucy Lu Wang and William R Hersh},
      eprint = {2104.09632},
      primaryclass = {cs.IR},
      title = {Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID},
      year = {2021},
    }
    
    ```
    



#### `TRECCOVID-Fa` { .model-copy }

TRECCOVID-Fa

**Dataset:** [`MCINext/trec-covid-fa`](https://huggingface.co/datasets/MCINext/trec-covid-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/trec-covid-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Medical | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `TRECCOVID-Fa.v2` { .model-copy }

TRECCOVID-Fa.v2

**Dataset:** [`MCINext/trec-covid-fa-v2`](https://huggingface.co/datasets/MCINext/trec-covid-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/trec-covid-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Medical | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `TRECCOVID-NL` { .model-copy }

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic. TRECCOVID-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-trec-covid`](https://huggingface.co/datasets/clips/beir-nl-trec-covid) • **License:** cc-by-4.0 • [Learn more →](https://colab.research.google.com/drive/1R99rjeAGt8S9IfAIRR3wS052sNu3Bjo-#scrollTo=4HduGW6xHnrZ)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Medical, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `TRECCOVID-PL` { .model-copy }

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/TRECCOVID-PL`](https://huggingface.co/datasets/mteb/TRECCOVID-PL) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic, Medical, Non-fiction, Written | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `TRECCOVID-VN` { .model-copy }

A translated dataset from TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/trec-covid-vn`](https://huggingface.co/datasets/GreenNode/trec-covid-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic, Medical, Written | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `TRECDL2019` { .model-copy }

TREC Deep Learning Track 2019 passage ranking task. The task involves retrieving relevant passages from the MS MARCO collection given web search queries. Queries have multi-graded relevance judgments.

**Dataset:** [`whybe-choi/trec-dl-2019`](https://huggingface.co/datasets/whybe-choi/trec-dl-2019) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{craswell2020overview,
      author = {Craswell, Nick and Mitra, Bhaskar and Yilmaz, Emine and Campos, Daniel and Voorhees, Ellen M},
      booktitle = {Proceedings of the 28th Text REtrieval Conference (TREC 2019)},
      organization = {NIST},
      title = {Overview of the TREC 2019 deep learning track},
      year = {2020},
    }
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### `TRECDL2020` { .model-copy }

TREC Deep Learning Track 2020 passage ranking task. The task involves retrieving relevant passages from the MS MARCO collection given web search queries. Queries have multi-graded relevance judgments.

**Dataset:** [`whybe-choi/trec-dl-2020`](https://huggingface.co/datasets/whybe-choi/trec-dl-2020) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{craswell2021overview,
      author = {Craswell, Nick and Mitra, Bhaskar and Yilmaz, Emine and Campos, Daniel and Voorhees, Ellen M},
      booktitle = {Proceedings of the 29th Text REtrieval Conference (TREC 2020)},
      organization = {NIST},
      title = {Overview of the TREC 2020 deep learning track},
      year = {2021},
    }
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### `TV2Nordretrieval` { .model-copy }

News Article and corresponding summaries extracted from the Danish newspaper TV2 Nord.

**Dataset:** [`alexandrainst/nordjylland-news-summarization`](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan | News, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
      address = {Marseille, France},
      author = {Flansmose Mikkelsen, Linea  and
    Kinch, Oliver  and
    Jess Pedersen, Anders  and
    Lacroix, Oph{\'e}lie},
      booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
      editor = {Calzolari, Nicoletta  and
    B{\'e}chet, Fr{\'e}d{\'e}ric  and
    Blache, Philippe  and
    Choukri, Khalid  and
    Cieri, Christopher  and
    Declerck, Thierry  and
    Goggi, Sara  and
    Isahara, Hitoshi  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Mazo, H{\'e}l{\`e}ne  and
    Odijk, Jan  and
    Piperidis, Stelios},
      month = jun,
      pages = {2440--2445},
      publisher = {European Language Resources Association},
      title = {{DD}is{C}o: A Discourse Coherence Dataset for {D}anish},
      url = {https://aclanthology.org/2022.lrec-1.260},
      year = {2022},
    }
    
    ```
    



#### `TVPLRetrieval` { .model-copy }

A Vietnamese dataset for evaluating legal text retrieval. From Thu vien phap luat (TVPL) dataset: Optimizing Answer Generator in Vietnamese Legal Question Answering Systems Using Language Models.

**Dataset:** [`GreenNode/TVPL-Retrieval-VN`](https://huggingface.co/datasets/GreenNode/TVPL-Retrieval-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.coling-main.233.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Legal | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1145/3732938,
      address = {New York, NY, USA},
      author = {Le, Huong and Luu, Ngoc and Nguyen, Thanh and Dao, Tuan and Dinh, Sang},
      doi = {10.1145/3732938},
      issn = {2375-4699},
      journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
      publisher = {Association for Computing Machinery},
      title = {Optimizing Answer Generator in Vietnamese Legal Question Answering Systems Using Language Models},
      url = {https://doi.org/10.1145/3732938},
      year = {2025},
    }
    
    ```
    



#### `TempReasonL1` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l1.

**Dataset:** [`mteb/TempReasonL1`](https://huggingface.co/datasets/mteb/TempReasonL1) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TempReasonL2Context` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-context.

**Dataset:** [`mteb/TempReasonL2Context`](https://huggingface.co/datasets/mteb/TempReasonL2Context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TempReasonL2Fact` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-fact.

**Dataset:** [`mteb/TempReasonL2Fact`](https://huggingface.co/datasets/mteb/TempReasonL2Fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TempReasonL2Pure` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-pure.

**Dataset:** [`mteb/TempReasonL2Pure`](https://huggingface.co/datasets/mteb/TempReasonL2Pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TempReasonL3Context` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-context.

**Dataset:** [`mteb/TempReasonL3Context`](https://huggingface.co/datasets/mteb/TempReasonL3Context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TempReasonL3Fact` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-fact.

**Dataset:** [`mteb/TempReasonL3Fact`](https://huggingface.co/datasets/mteb/TempReasonL3Fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TempReasonL3Pure` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-pure.

**Dataset:** [`mteb/TempReasonL3Pure`](https://huggingface.co/datasets/mteb/TempReasonL3Pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `TopiOCQA` { .model-copy }

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage.

**Dataset:** [`mteb/TopiOCQA`](https://huggingface.co/datasets/mteb/TopiOCQA) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{adlakha2022topiocqa,
      archiveprefix = {arXiv},
      author = {Vaibhav Adlakha and Shehzaad Dhuliawala and Kaheer Suleman and Harm de Vries and Siva Reddy},
      eprint = {2110.00768},
      primaryclass = {cs.CL},
      title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
      year = {2022},
    }
    
    ```
    



#### `TopiOCQAHardNegatives` { .model-copy }

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/TopiOCQA_validation_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/TopiOCQA_validation_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{adlakha2022topiocqa,
      archiveprefix = {arXiv},
      author = {Vaibhav Adlakha and Shehzaad Dhuliawala and Kaheer Suleman and Harm de Vries and Siva Reddy},
      eprint = {2110.00768},
      primaryclass = {cs.CL},
      title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
      year = {2022},
    }
    
    ```
    



#### `Touche2020` { .model-copy }

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/touche2020`](https://huggingface.co/datasets/mteb/touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{potthast_2022_6862281,
      author = {Potthast, Martin and
    Gienapp, Lukas and
    Wachsmuth, Henning and
    Hagen, Matthias and
    Fröbe, Maik and
    Bondarenko, Alexander and
    Ajjour, Yamen and
    Stein, Benno},
      doi = {10.5281/zenodo.6862281},
      month = jul,
      publisher = {Zenodo},
      title = {{Touché20-Argument-Retrieval-for-Controversial-
    Questions}},
      url = {https://doi.org/10.5281/zenodo.6862281},
      year = {2022},
    }
    
    ```
    



#### `Touche2020-Fa` { .model-copy }

Touche2020-Fa

**Dataset:** [`MCINext/touche2020-fa`](https://huggingface.co/datasets/MCINext/touche2020-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/touche2020-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `Touche2020-Fa.v2` { .model-copy }

Touche2020-Fa.v2

**Dataset:** [`MCINext/webis-touche2020-v3-fa`](https://huggingface.co/datasets/MCINext/webis-touche2020-v3-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/touche2020-fa-v2)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }
    ```
    



#### `Touche2020-NL` { .model-copy }

Touché Task 1: Argument Retrieval for Controversial Questions. Touche2020-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-webis-touche2020`](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Non-fiction | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### `Touche2020-PL` { .model-copy }

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/Touche2020-PL`](https://huggingface.co/datasets/mteb/Touche2020-PL) • **License:** not specified • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Academic | derived | machine-translated | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### `Touche2020-VN` { .model-copy }

A translated dataset from Touché Task 1: Argument Retrieval for Controversial Questions The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/webis-touche2020-vn`](https://huggingface.co/datasets/GreenNode/webis-touche2020-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Academic | derived | machine-translated and LM verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }
    
    ```
    



#### `Touche2020Retrieval.v3` { .model-copy }

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/webis-touche2020-v3`](https://huggingface.co/datasets/mteb/webis-touche2020-v3) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/castorini/touche-error-analysis)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Academic | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Thakur_etal_SIGIR2024,
      address_ = {Washington, D.C.},
      author = {Nandan Thakur and Luiz Bonifacio and Maik {Fr\"{o}be} and Alexander Bondarenko and Ehsan Kamalloo and Martin Potthast and Matthias Hagen and Jimmy Lin},
      booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      title = {Systematic Evaluation of Neural Retrieval Models on the {Touch\'{e}} 2020 Argument Retrieval Subset of {BEIR}},
      year = {2024},
    }
    
    ```
    



#### `TurHistQuadRetrieval` { .model-copy }

Question Answering dataset on Ottoman History in Turkish

**Dataset:** [`asparius/TurHistQuAD`](https://huggingface.co/datasets/asparius/TurHistQuAD) • **License:** mit • [Learn more →](https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | tur | Academic, Encyclopaedic, Non-fiction, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{9559013,
      author = {Soygazi, Fatih and Çiftçi, Okan and Kök, Uğurcan and Cengiz, Soner},
      booktitle = {2021 6th International Conference on Computer Science and Engineering (UBMK)},
      doi = {10.1109/UBMK52708.2021.9559013},
      keywords = {Computer science;Computational modeling;Neural networks;Knowledge discovery;Information retrieval;Natural language processing;History;question answering;information retrieval;natural language understanding;deep learning;contextualized word embeddings},
      number = {},
      pages = {215-220},
      title = {THQuAD: Turkish Historic Question Answering Dataset for Reading Comprehension},
      volume = {},
      year = {2021},
    }
    
    ```
    



#### `TwitterHjerneRetrieval` { .model-copy }

Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer.

**Dataset:** [`mteb/TwitterHjerneRetrieval`](https://huggingface.co/datasets/mteb/TwitterHjerneRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan | Social, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{holm2024gllms,
      author = {Holm, Soren Vejlgaard},
      title = {Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish},
      year = {2024},
    }
    
    ```
    



#### `VABBRetrieval` { .model-copy }

This dataset contains the fourteenth edition of the Flemish Academic Bibliography for the Social Sciences and Humanities (VABB-SHW), a database of academic publications from the social sciences and humanities authored by researchers affiliated to Flemish universities (more information). Publications in the database are used as one of the parameters of the Flemish performance-based research funding system

**Dataset:** [`clips/mteb-nl-vabb-ret`](https://huggingface.co/datasets/clips/mteb-nl-vabb-ret) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://zenodo.org/records/14214806)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Academic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{aspeslagh2024vabb,
      author = {Aspeslagh, Pieter and Guns, Raf and Engels, Tim C. E.},
      doi = {10.5281/zenodo.14214806},
      publisher = {Zenodo},
      title = {VABB-SHW: Dataset of Flemish Academic Bibliography for the Social Sciences and Humanities (edition 14)},
      url = {https://doi.org/10.5281/zenodo.14214806},
      year = {2024},
    }
    
    ```
    



#### `VDRMultilingualRetrieval` { .model-copy }

Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German

**Dataset:** [`llamaindex/vdr-multilingual-test`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | deu, eng, fra, ita, spa | Web | LM-generated | found | ndcg_at_5 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{llamaindex2024vdrmultilingual,
      author = {LlamaIndex},
      howpublished = {https://huggingface.co/datasets/llamaindex/vdr-multilingual-test},
      title = {Visual Document Retrieval Goes Multilingual},
      year = {2025},
    }
    
    ```
    



#### `VideoRetrieval` { .model-copy }

VideoRetrieval

**Dataset:** [`mteb/VideoRetrieval`](https://huggingface.co/datasets/mteb/VideoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Entertainment, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### `VieQuADRetrieval` { .model-copy }

A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.

**Dataset:** [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) • **License:** mit • [Learn more →](https://aclanthology.org/2020.coling-main.233.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Encyclopaedic, Non-fiction, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{nguyen-etal-2020-vietnamese,
      address = {Barcelona, Spain (Online)},
      author = {Nguyen, Kiet  and
    Nguyen, Vu  and
    Nguyen, Anh  and
    Nguyen, Ngan},
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      doi = {10.18653/v1/2020.coling-main.233},
      editor = {Scott, Donia  and
    Bel, Nuria  and
    Zong, Chengqing},
      month = dec,
      pages = {2595--2605},
      publisher = {International Committee on Computational Linguistics},
      title = {A Vietnamese Dataset for Evaluating Machine Reading Comprehension},
      url = {https://aclanthology.org/2020.coling-main.233},
      year = {2020},
    }
    
    ```
    



#### `VisRAGRetArxivQA` { .model-copy }

evaluate vision-based retrieval and generation on scientific figures and their surrounding context to preserve complex layouts and mathematical notations.

**Dataset:** [`mteb/VisRAGRetArxivQA`](https://huggingface.co/datasets/mteb/VisRAGRetArxivQA) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2403.00231)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Academic, Non-fiction | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    @misc{li2024multimodalarxivdatasetimproving,
      archiveprefix = {arXiv},
      author = {Lei Li and Yuqi Wang and Runxin Xu and Peiyi Wang and Xiachong Feng and Lingpeng Kong and Qi Liu},
      eprint = {2403.00231},
      primaryclass = {cs.CV},
      title = {Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models},
      url = {https://arxiv.org/abs/2403.00231},
      year = {2024},
    }
    ```
    



#### `VisRAGRetChartQA` { .model-copy }

Assess end-to-end vision-based RAG performance on real-world charts requiring complex logical and visual reasoning from retrieved images.

**Dataset:** [`mteb/VisRAGRetChartQA`](https://huggingface.co/datasets/mteb/VisRAGRetChartQA) • **License:** gpl-3.0 • [Learn more →](https://arxiv.org/abs/2203.10244)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Non-fiction, Web | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    @misc{masry2022chartqabenchmarkquestionanswering,
      archiveprefix = {arXiv},
      author = {Ahmed Masry and Do Xuan Long and Jia Qing Tan and Shafiq Joty and Enamul Hoque},
      eprint = {2203.10244},
      primaryclass = {cs.CL},
      title = {ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
      url = {https://arxiv.org/abs/2203.10244},
      year = {2022},
    }
    ```
    



#### `VisRAGRetInfoVQA` { .model-copy }

Evaluate the retrieval and understanding of complex infographics where layout and graphical elements are essential for cross-modal question answering.

**Dataset:** [`mteb/VisRAGRetInfoVQA`](https://huggingface.co/datasets/mteb/VisRAGRetInfoVQA) • **License:** mit • [Learn more →](https://arxiv.org/abs/2104.12756)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    @misc{mathew2021infographicvqa,
      archiveprefix = {arXiv},
      author = {Minesh Mathew and Viraj Bagal and Rubèn Pérez Tito and Dimosthenis Karatzas and Ernest Valveny and C. V Jawahar},
      eprint = {2104.12756},
      primaryclass = {cs.CV},
      title = {InfographicVQA},
      url = {https://arxiv.org/abs/2104.12756},
      year = {2021},
    }
    ```
    



#### `VisRAGRetMPDocVQA` { .model-copy }

Benchmark the ability to retrieve specific relevant pages from multi-page documents and generate answers based on visual evidence.

**Dataset:** [`mteb/VisRAGRetMPDocVQA`](https://huggingface.co/datasets/mteb/VisRAGRetMPDocVQA) • **License:** mit • [Learn more →](https://arxiv.org/abs/2212.05935)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Non-fiction, Web | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    @misc{tito2023hierarchicalmultimodaltransformersmultipage,
      archiveprefix = {arXiv},
      author = {Rubèn Tito and Dimosthenis Karatzas and Ernest Valveny},
      eprint = {2212.05935},
      primaryclass = {cs.CV},
      title = {Hierarchical multimodal transformers for Multi-Page DocVQA},
      url = {https://arxiv.org/abs/2212.05935},
      year = {2023},
    }
    ```
    



#### `VisRAGRetPlotQA` { .model-copy }

Execute vision-based retrieval and numerical reasoning over scientific plots to answer questions without relying on structured data parsing.

**Dataset:** [`mteb/VisRAGRetPlotQA`](https://huggingface.co/datasets/mteb/VisRAGRetPlotQA) • **License:** mit • [Learn more →](https://arxiv.org/abs/1909.00997)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Non-fiction, Web | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    @misc{methani2020plotqareasoningscientificplots,
      archiveprefix = {arXiv},
      author = {Nitesh Methani and Pritha Ganguly and Mitesh M. Khapra and Pratyush Kumar},
      eprint = {1909.00997},
      primaryclass = {cs.CV},
      title = {PlotQA: Reasoning over Scientific Plots},
      url = {https://arxiv.org/abs/1909.00997},
      year = {2020},
    }
    ```
    



#### `VisRAGRetSlideVQA` { .model-copy }

Retrieve and reason across multiple slide images within a deck to answer multi-hop questions in a vision-centric retrieval-augmented generation pipeline.

**Dataset:** [`mteb/VisRAGRetSlideVQA`](https://huggingface.co/datasets/mteb/VisRAGRetSlideVQA) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2301.04883)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to image (t2i) | eng | Web | derived | found | recall_at_10 |



??? quote "Citation"

    
    ```bibtex
    @misc{tanaka2023slidevqadatasetdocumentvisual,
      archiveprefix = {arXiv},
      author = {Ryota Tanaka and Kyosuke Nishida and Kosuke Nishida and Taku Hasegawa and Itsumi Saito and Kuniko Saito},
      eprint = {2301.04883},
      primaryclass = {cs.CL},
      title = {SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images},
      url = {https://arxiv.org/abs/2301.04883},
      year = {2023},
    }
    ```
    



#### `WebFAQRetrieval` { .model-copy }

WebFAQ is a broad-coverage corpus of natural question-answer pairs in 75 languages, gathered from FAQ pages on the web.

**Dataset:** [`mteb/WebFAQRetrieval`](https://huggingface.co/datasets/mteb/WebFAQRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, aze, ben, bul, cat, ... (51) | Web, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @misc{dinzinger2025webfaq,
      archiveprefix = {arXiv},
      author = {Michael Dinzinger and Laura Caspari and Kanishka Ghosh Dastidar and Jelena Mitrović and Michael Granitzer},
      eprint = {2502.20936},
      primaryclass = {cs.CL},
      title = {WebFAQ: A Multilingual Collection of Natural Q&amp;A Datasets for Dense Retrieval},
      url = {https://arxiv.org/abs/2502.20936},
      year = {2025},
    }
    
    ```
    



#### `WikiSQLRetrieval` { .model-copy }

A code retrieval task based on WikiSQL dataset with natural language questions and corresponding SQL queries. Each query is a natural language question (e.g., 'What is the name of the team that has scored the most goals?'), and the corpus contains SQL query implementations. The task is to retrieve the correct SQL query that answers the natural language question. Queries are natural language questions while the corpus contains SQL SELECT statements with proper syntax and logic for querying database tables.

**Dataset:** [`embedding-benchmark/WikiSQL_mteb`](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb) • **License:** bsd-3-clause • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, sql | Programming | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhong2017seq2sql,
      archiveprefix = {arXiv},
      author = {Zhong, Victor and Xiong, Caiming and Socher, Richard},
      eprint = {1709.00103},
      primaryclass = {cs.CL},
      title = {Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning},
      year = {2017},
    }
    
    ```
    



#### `WikipediaRetrievalMultilingual` { .model-copy }

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRetrievalMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRetrievalMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ben, bul, ces, dan, deu, ... (16) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified | ndcg_at_10 |



#### `WinoGrande` { .model-copy }

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on winogrande.

**Dataset:** [`mteb/WinoGrande`](https://huggingface.co/datasets/mteb/WinoGrande) • **License:** not specified • [Learn more →](https://winogrande.allenai.org/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{sakaguchi2021winogrande,
      author = {Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
      journal = {Communications of the ACM},
      number = {9},
      pages = {99--106},
      publisher = {ACM New York, NY, USA},
      title = {Winogrande: An adversarial winograd schema challenge at scale},
      volume = {64},
      year = {2021},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### `XMarket` { .model-copy }

XMarket

**Dataset:** [`mteb/XMarket`](https://huggingface.co/datasets/mteb/XMarket) • **License:** not specified • [Learn more →](http://dx.doi.org/10.1145/3459637.3482493)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu, eng, spa | Reviews, Written | derived | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Bonab_2021,
      author = {Bonab, Hamed and Aliannejadi, Mohammad and Vardasbi, Ali and Kanoulas, Evangelos and Allan, James},
      booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
      collection = {CIKM ’21},
      doi = {10.1145/3459637.3482493},
      month = oct,
      publisher = {ACM},
      series = {CIKM ’21},
      title = {Cross-Market Product Recommendation},
      url = {http://dx.doi.org/10.1145/3459637.3482493},
      year = {2021},
    }
    
    ```
    



#### `XPQARetrieval` { .model-copy }

XPQARetrieval

**Dataset:** [`mteb/XPQARetrieval`](https://huggingface.co/datasets/mteb/XPQARetrieval) • **License:** cdla-sharing-1.0 • [Learn more →](https://arxiv.org/abs/2305.09249)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, cmn, deu, eng, fra, ... (13) | Reviews, Written | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{shen2023xpqa,
      author = {Shen, Xiaoyu and Asai, Akari and Byrne, Bill and De Gispert, Adria},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track)},
      pages = {103--115},
      title = {xPQA: Cross-Lingual Product Question Answering in 12 Languages},
      year = {2023},
    }
    
    ```
    



#### `XQuADRetrieval` { .model-copy }

XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question.

**Dataset:** [`google/xquad`](https://huggingface.co/datasets/google/xquad) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/xquad)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | arb, deu, ell, eng, hin, ... (12) | Web, Written | human-annotated | created | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{Artetxe:etal:2019,
      archiveprefix = {arXiv},
      author = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      eprint = {1910.11856},
      journal = {CoRR},
      title = {On the cross-lingual transferability of monolingual representations},
      volume = {abs/1910.11856},
      year = {2019},
    }
    
    @inproceedings{dumitrescu2021liro,
      author = {Stefan Daniel Dumitrescu and Petru Rebeja and Beata Lorincz and Mihaela Gaman and Andrei Avram and Mihai Ilie and Andrei Pruteanu and Adriana Stan and Lorena Rosia and Cristina Iacobescu and Luciana Morogan and George Dima and Gabriel Marchidan and Traian Rebedea and Madalina Chitez and Dani Yogatama and Sebastian Ruder and Radu Tudor Ionescu and Razvan Pascanu and Viorica Patraucean},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
      title = {LiRo: Benchmark and leaderboard for Romanian language tasks},
      url = {https://openreview.net/forum?id=JH61CD7afTv},
      year = {2021},
    }
    
    ```
    



#### `ZacLegalTextRetrieval` { .model-copy }

Zalo Legal Text documents

**Dataset:** [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) • **License:** mit • [Learn more →](https://challenge.zalo.ai/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | vie | Legal | human-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{10.1007/978-981-95-1746-6_17,
      address = {Singapore},
      author = {Pham, Bao Loc
    and Hoang, Quoc Viet
    and Luu, Quy Tung
    and Vo, Trong Thu},
      booktitle = {Proceedings of the Fifth International Conference on Intelligent Systems and Networks},
      isbn = {978-981-95-1746-6},
      pages = {153--163},
      publisher = {Springer Nature Singapore},
      title = {GN-TRVN: A Benchmark for Vietnamese Table Markdown Retrieval Task},
      year = {2026},
    }
    
    ```
    



#### `bBSARDNLRetrieval` { .model-copy }

Building on the Belgian Statutory Article Retrieval Dataset (BSARD) in French, we introduce the bilingual version of this dataset, bBSARD. The dataset contains parallel Belgian statutory articles in both French and Dutch, along with legal questions from BSARD and their Dutch translation.

**Dataset:** [`clips/mteb-nl-bbsard`](https://huggingface.co/datasets/clips/mteb-nl-bbsard) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2025.regnlp-1.3.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Legal, Written | expert-annotated | found | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{lotfi2025bilingual,
      author = {Lotfi, Ehsan and Banar, Nikolay and Yuzbashyan, Nerses and Daelemans, Walter},
      journal = {COLING 2025},
      pages = {10},
      title = {Bilingual BSARD: Extending Statutory Article Retrieval to Dutch},
      year = {2025},
    }
    
    ```
    



#### `mMARCO-NL` { .model-copy }

mMARCO is a multi-lingual (translated) collection of datasets focused on deep learning in search

**Dataset:** [`clips/beir-nl-mmarco`](https://huggingface.co/datasets/clips/beir-nl-mmarco) • **License:** apache-2.0 • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | Web, Written | derived | machine-translated and verified | ndcg_at_10 |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/abs-2108-13897,
      author = {Luiz Bonifacio and
    Israel Campiotti and
    Roberto de Alencar Lotufo and
    Rodrigo Frassetto Nogueira},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-2108-13897.bib},
      eprint = {2108.13897},
      eprinttype = {arXiv},
      journal = {CoRR},
      timestamp = {Mon, 20 Mar 2023 15:35:34 +0100},
      title = {mMARCO: {A} Multilingual Version of {MS} {MARCO} Passage Ranking Dataset},
      url = {https://arxiv.org/abs/2108.13897},
      volume = {abs/2108.13897},
      year = {2021},
    }
    
    ```




## VisionCentricQA

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 6

#### `BLINKIT2IMultiChoice` { .model-copy }

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`mteb/blink-it2i-multi`](https://huggingface.co/datasets/mteb/blink-it2i-multi) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2404.12390)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to image (it2i) | eng | Encyclopaedic | derived | found | accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @article{fu2024blink,
      author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
      journal = {arXiv preprint arXiv:2404.12390},
      title = {Blink: Multimodal large language models can see but not perceive},
      year = {2024},
    }
    
    ```
    



#### `BLINKIT2TMultiChoice` { .model-copy }

Retrieve the correct text answer based on images and specific retrieval instructions.

**Dataset:** [`mteb/blink-it2t-multi`](https://huggingface.co/datasets/mteb/blink-it2t-multi) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2404.12390)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Encyclopaedic | derived | found | accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @article{fu2024blink,
      author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
      journal = {arXiv preprint arXiv:2404.12390},
      title = {Blink: Multimodal large language models can see but not perceive},
      year = {2024},
    }
    
    ```
    



#### `CVBenchCount` { .model-copy }

count the number of objects in the image.

**Dataset:** [`mteb/CV-Bench`](https://huggingface.co/datasets/mteb/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Academic | derived | found | accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @article{tong2024cambrian,
      author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
      journal = {arXiv preprint arXiv:2406.16860},
      title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
      year = {2024},
    }
    
    ```
    



#### `CVBenchDepth` { .model-copy }

judge the depth of the objects in the image with similarity matching.

**Dataset:** [`mteb/CV-Bench`](https://huggingface.co/datasets/mteb/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Academic | derived | found | accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @article{tong2024cambrian,
      author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
      journal = {arXiv preprint arXiv:2406.16860},
      title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
      year = {2024},
    }
    
    ```
    



#### `CVBenchDistance` { .model-copy }

judge the distance of the objects in the image with similarity matching.

**Dataset:** [`mteb/CV-Bench`](https://huggingface.co/datasets/mteb/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Academic | derived | found | accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @article{tong2024cambrian,
      author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
      journal = {arXiv preprint arXiv:2406.16860},
      title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
      year = {2024},
    }
    
    ```
    



#### `CVBenchRelation` { .model-copy }

decide the relation of the objects in the image.

**Dataset:** [`mteb/CV-Bench`](https://huggingface.co/datasets/mteb/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image, text to text (it2t) | eng | Academic | derived | found | accuracy |



??? quote "Citation"

    
    ```bibtex
    
    @article{tong2024cambrian,
      author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
      journal = {arXiv preprint arXiv:2406.16860},
      title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
      year = {2024},
    }
    
    ```

<!-- END-TASKS -->