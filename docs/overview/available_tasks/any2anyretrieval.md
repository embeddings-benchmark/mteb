
# Any2AnyRetrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 87

#### AudioCapsA2TRetrieval

Natural language description for any kind of audio in the wild.

**Dataset:** [`mteb/audiocaps_a2t`](https://huggingface.co/datasets/mteb/audiocaps_a2t) • **License:** mit • [Learn more →](https://audiocaps.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng, zxx | Encyclopaedic, Written | derived | found |



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




#### AudioCapsT2ARetrieval

Natural language description for any kind of audio in the wild.

**Dataset:** [`mteb/audiocaps_t2a`](https://huggingface.co/datasets/mteb/audiocaps_t2a) • **License:** mit • [Learn more →](https://audiocaps.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng, zxx | Encyclopaedic, Written | derived | found |



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




#### AudioSetStrongA2TRetrieval

Retrieve all temporally-strong labeled events within 10s audio clips from the AudioSet Strongly-Labeled subset.

**Dataset:** [`mteb/audioset_strong_a2t`](https://huggingface.co/datasets/mteb/audioset_strong_a2t) • **License:** cc-by-4.0 • [Learn more →](https://research.google.com/audioset/download_strong.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | AudioScene | derived | found |



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




#### AudioSetStrongT2ARetrieval

Retrieve audio segments corresponding to a given sound event label from the AudioSet Strongly-Labeled 10s clips.

**Dataset:** [`mteb/audioset_strong_t2a`](https://huggingface.co/datasets/mteb/audioset_strong_t2a) • **License:** cc-by-4.0 • [Learn more →](https://research.google.com/audioset/download_strong.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | AudioScene | derived | found |



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




#### BLINKIT2IRetrieval

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`JamieSJS/blink-it2i`](https://huggingface.co/datasets/JamieSJS/blink-it2i) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.12390)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{fu2024blink,
      author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
      journal = {arXiv preprint arXiv:2404.12390},
      title = {Blink: Multimodal large language models can see but not perceive},
      year = {2024},
    }

    ```




#### BLINKIT2TRetrieval

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`JamieSJS/blink-it2t`](https://huggingface.co/datasets/JamieSJS/blink-it2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.12390)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_1 | eng | Encyclopaedic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{fu2024blink,
      author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
      journal = {arXiv preprint arXiv:2404.12390},
      title = {Blink: Multimodal large language models can see but not perceive},
      year = {2024},
    }

    ```




#### CIRRIT2IRetrieval

Retrieve images based on texts and images.

**Dataset:** [`MRBench/mbeir_cirr_task7`](https://huggingface.co/datasets/MRBench/mbeir_cirr_task7) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### CMUArcticA2TRetrieval

Retrieve the correct transcription for an English speech segment. The dataset is derived from the phonetically balanced CMU Arctic single-speaker TTS corpora. The corpora contains 1150 samples based on read-aloud segments from books, which are out of copyright and derived from the Gutenberg project.

**Dataset:** [`mteb/CMU_Arctic_a2t`](https://huggingface.co/datasets/mteb/CMU_Arctic_a2t) • **License:** cc0-1.0 • [Learn more →](http://festvox.org/cmu_arctic/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Spoken | derived | found |



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




#### CMUArcticT2ARetrieval

Retrieve the correct audio segment for an English transcription. The dataset is derived from the phonetically balanced CMU Arctic single-speaker TTS corpora. The corpora contains 1150 audio-text pairs based on read-aloud segments from public domain books originally sourced from the Gutenberg project.

**Dataset:** [`mteb/CMU_Arctic_t2a`](https://huggingface.co/datasets/mteb/CMU_Arctic_t2a) • **License:** cc0-1.0 • [Learn more →](http://festvox.org/cmu_arctic/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Spoken | derived | found |



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




#### CUB200I2IRetrieval

Retrieve bird images from 200 classes.

**Dataset:** [`isaacchung/cub200_retrieval`](https://huggingface.co/datasets/isaacchung/cub200_retrieval) • **License:** not specified • [Learn more →](https://www.florian-schroff.de/publications/CUB-200.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



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




#### ClothoA2TRetrieval

An audio captioning datasetst containing audio clips and their corresponding captions.

**Dataset:** [`CLAPv2/Clotho`](https://huggingface.co/datasets/CLAPv2/Clotho) • **License:** mit • [Learn more →](https://github.com/audio-captioning/clotho-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Encyclopaedic, Written | derived | found |



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




#### ClothoT2ARetrieval

An audio captioning datasetst containing audio clips from the Freesound platform and their corresponding captions.

**Dataset:** [`CLAPv2/Clotho`](https://huggingface.co/datasets/CLAPv2/Clotho) • **License:** mit • [Learn more →](https://github.com/audio-captioning/clotho-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Encyclopaedic, Written | derived | found |



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




#### CommonVoice17A2TRetrieval

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_17_0_mini`](https://huggingface.co/datasets/mteb/common_voice_17_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | ara, ast, bel, ben, bre, ... (50) | Spoken | human-annotated | found |



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




#### CommonVoice17T2ARetrieval

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_17_0_mini`](https://huggingface.co/datasets/mteb/common_voice_17_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | ara, ast, bel, ben, bre, ... (50) | Spoken | human-annotated | found |



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




#### CommonVoice21A2TRetrieval

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_21_0_mini`](https://huggingface.co/datasets/mteb/common_voice_21_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | abk, afr, amh, ara, asm, ... (114) | Spoken | human-annotated | found |



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




#### CommonVoice21T2ARetrieval

Speech recordings with corresponding text transcriptions from CommonVoice dataset.

**Dataset:** [`mteb/common_voice_21_0_mini`](https://huggingface.co/datasets/mteb/common_voice_21_0_mini) • **License:** cc0-1.0 • [Learn more →](https://commonvoice.mozilla.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | abk, afr, amh, ara, asm, ... (114) | Spoken | human-annotated | found |



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




#### EDIST2ITRetrieval

Retrieve news images and titles based on news content.

**Dataset:** [`MRBench/mbeir_edis_task2`](https://huggingface.co/datasets/MRBench/mbeir_edis_task2) • **License:** apache-2.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.297/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image, text (t2it) | ndcg_at_10 | eng | News | derived | created |



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




#### EmoVDBA2TRetrieval

Natural language emotional captions for speech segments from the EmoV-DB emotional voices database.

**Dataset:** [`mteb/EmoV_DB_a2t`](https://huggingface.co/datasets/mteb/EmoV_DB_a2t) • **License:** https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md • [Learn more →](https://github.com/numediart/EmoV-DB?tab=readme-ov-file)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @article{adigwe2018emotional,
      author = {Adigwe, Adaeze and Tits, No{\'e} and Haddad, Kevin El and Ostadabbas, Sarah and Dutoit, Thierry},
      journal = {arXiv preprint arXiv:1806.09514},
      title = {The emotional voices database: Towards controlling the emotion dimension in voice generation systems},
      year = {2018},
    }

    ```




#### EmoVDBT2ARetrieval

Natural language emotional captions for speech segments from the EmoV-DB emotional voices database.

**Dataset:** [`mteb/EmoV_DB_t2a`](https://huggingface.co/datasets/mteb/EmoV_DB_t2a) • **License:** https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md • [Learn more →](https://github.com/numediart/EmoV-DB?tab=readme-ov-file)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @article{adigwe2018emotional,
      author = {Adigwe, Adaeze and Tits, No{\'e} and Haddad, Kevin El and Ostadabbas, Sarah and Dutoit, Thierry},
      journal = {arXiv preprint arXiv:1806.09514},
      title = {The emotional voices database: Towards controlling the emotion dimension in voice generation systems},
      year = {2018},
    }

    ```




#### EncyclopediaVQAIT2ITRetrieval

Retrieval Wiki passage and image and passage to answer query about an image.

**Dataset:** [`izhx/UMRB-EncyclopediaVQA`](https://huggingface.co/datasets/izhx/UMRB-EncyclopediaVQA) • **License:** cc-by-4.0 • [Learn more →](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | cv_recall_at_5 | eng | Encyclopaedic | derived | created |



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




#### FORBI2IRetrieval

Retrieve flat object images from 8 classes.

**Dataset:** [`isaacchung/forb_retrieval`](https://huggingface.co/datasets/isaacchung/forb_retrieval) • **License:** not specified • [Learn more →](https://github.com/pxiangwu/FORB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



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




#### Fashion200kI2TRetrieval

Retrieve clothes based on descriptions.

**Dataset:** [`MRBench/mbeir_fashion200k_task3`](https://huggingface.co/datasets/MRBench/mbeir_fashion200k_task3) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### Fashion200kT2IRetrieval

Retrieve clothes based on descriptions.

**Dataset:** [`MRBench/mbeir_fashion200k_task0`](https://huggingface.co/datasets/MRBench/mbeir_fashion200k_task0) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### FashionIQIT2IRetrieval

Retrieve clothes based on descriptions.

**Dataset:** [`MRBench/mbeir_fashioniq_task7`](https://huggingface.co/datasets/MRBench/mbeir_fashioniq_task7) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### FleursA2TRetrieval

Speech recordings with corresponding text transcriptions from the FLEURS dataset.

**Dataset:** [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) • **License:** apache-2.0 • [Learn more →](https://github.com/google-research-datasets/fleurs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | afr, amh, ara, asm, ast, ... (102) | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex
    @inproceedings{conneau2023fleurs,
      author = {Conneau, Alexis and Kocmi, Tom and Ruder, Sebastian and Sainz, Oscar and Chaudhary, Vishrav and Guzmán, Francisco and Joulin, Armand and Khandelwal, Kartikay and Kumar, Shubham and Moehs, Florian and Pino, Juan and Poncelas, Alberto and Seedat, Saadia and Stojanovski, Daan and Wang, Jingfei and Wang, Mona and Wenzek, Guillaume and Wrona, Piotr and Zhou, Wei},
      booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association (INTERSPEECH 2023)},
      year = {2023},
    }

    ```




#### FleursT2ARetrieval

Speech recordings with corresponding text transcriptions from the FLEURS dataset.

**Dataset:** [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) • **License:** apache-2.0 • [Learn more →](https://github.com/google-research-datasets/fleurs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | afr, amh, ara, asm, ast, ... (102) | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex
    @inproceedings{conneau2023fleurs,
      author = {Conneau, Alexis and Kocmi, Tom and Ruder, Sebastian and Sainz, Oscar and Chaudhary, Vishrav and Guzmán, Francisco and Joulin, Armand and Khandelwal, Kartikay and Kumar, Shubham and Moehs, Florian and Pino, Juan and Poncelas, Alberto and Seedat, Saadia and Stojanovski, Daan and Wang, Jingfei and Wang, Mona and Wenzek, Guillaume and Wrona, Piotr and Zhou, Wei},
      booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association (INTERSPEECH 2023)},
      year = {2023},
    }

    ```




#### Flickr30kI2TRetrieval

Retrieve captions based on images.

**Dataset:** [`isaacchung/flickr30ki2t`](https://huggingface.co/datasets/isaacchung/flickr30ki2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Web, Written | derived | found |



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




#### Flickr30kT2IRetrieval

Retrieve images based on captions.

**Dataset:** [`isaacchung/flickr30kt2i`](https://huggingface.co/datasets/isaacchung/flickr30kt2i) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Web, Written | derived | found |



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




#### GLDv2I2IRetrieval

Retrieve names of landmarks based on their image.

**Dataset:** [`gowitheflow/gld-v2`](https://huggingface.co/datasets/gowitheflow/gld-v2) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### GLDv2I2TRetrieval

Retrieve names of landmarks based on their image.

**Dataset:** [`JamieSJS/gld-v2-i2t`](https://huggingface.co/datasets/JamieSJS/gld-v2-i2t) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### GigaSpeechA2TRetrieval

Given an English speech segment, retrieve its correct transcription. Audio comes from the 10 000‑hour training subset of GigaSpeech, which originates from ≈40 000 hours of transcribed audiobooks, podcasts, and YouTube.

**Dataset:** [`mteb/gigaspeech_a2t`](https://huggingface.co/datasets/mteb/gigaspeech_a2t) • **License:** https://github.com/SpeechColab/GigaSpeech/blob/main/LICENSE • [Learn more →](https://github.com/SpeechColab/GigaSpeech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{GigaSpeech2021,
      author = {Chen, Guoguo and Chai, Shuzhou and Wang, Guanbo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and Wang, Yujun and You, Zhao and Yan, Zhiyong},
      booktitle = {Proc. Interspeech 2021},
      title = {GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio},
      year = {2021},
    }

    ```




#### GigaSpeechT2ARetrieval

Given an English transcription, retrieve its corresponding audio segment. Audio comes from the 10 000‑hour training subset of GigaSpeech, sourced from ≈40 000 hours of transcribed audiobooks, podcasts, and YouTube.

**Dataset:** [`mteb/gigaspeech_t2a`](https://huggingface.co/datasets/mteb/gigaspeech_t2a) • **License:** https://github.com/SpeechColab/GigaSpeech/blob/main/LICENSE • [Learn more →](https://github.com/SpeechColab/GigaSpeech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{GigaSpeech2021,
      author = {Chen, Guoguo and Chai, Shuzhou and Wang, Guanbo and Du, Jiayu and Zhang, Wei-Qiang and Weng, Chao and Su, Dan and Povey, Daniel and Trmal, Jan and Zhang, Junbo and Jin, Mingjie and Khudanpur, Sanjeev and Watanabe, Shinji and Zhao, Shuaijiang and Zou, Wei and Li, Xiangang and Yao, Xuchen and Wang, Yongqing and Wang, Yujun and You, Zhao and Yan, Zhiyong},
      booktitle = {Proc. Interspeech 2021},
      title = {GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio},
      year = {2021},
    }

    ```




#### GoogleSVQA2TRetrieval

Multilingual audio-to-text retrieval using the Simple Voice Questions (SVQ) dataset. Given an audio query, retrieve the corresponding text transcription.

**Dataset:** [`google/svq`](https://huggingface.co/datasets/google/svq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/google/svq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | acm, apc, arq, arz, ben, ... (20) | Spoken | human-annotated | found |



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




#### GoogleSVQT2ARetrieval

Multilingual text-to-audio retrieval using the Simple Voice Questions (SVQ) dataset. Given a text query, retrieve the corresponding audio recording.

**Dataset:** [`google/svq`](https://huggingface.co/datasets/google/svq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/google/svq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | acm, apc, arq, arz, ben, ... (20) | Spoken | human-annotated | found |



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




#### HatefulMemesI2TRetrieval

Retrieve captions based on memes to assess OCR abilities.

**Dataset:** [`Ahren09/MMSoc_HatefulMemes`](https://huggingface.co/datasets/Ahren09/MMSoc_HatefulMemes) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2005.04790)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### HatefulMemesT2IRetrieval

Retrieve captions based on memes to assess OCR abilities.

**Dataset:** [`Ahren09/MMSoc_HatefulMemes`](https://huggingface.co/datasets/Ahren09/MMSoc_HatefulMemes) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2005.04790)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### HiFiTTSA2TRetrieval

Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.

**Dataset:** [`mteb/hifi-tts_a2t`](https://huggingface.co/datasets/mteb/hifi-tts_a2t) • **License:** cc-by-4.0 • [Learn more →](https://openslr.org/109/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @article{bakhturina2021hi,
      author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
      journal = {arXiv preprint arXiv:2104.01497},
      title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
      year = {2021},
    }

    ```




#### HiFiTTST2ARetrieval

Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.

**Dataset:** [`mteb/hifi-tts_t2a`](https://huggingface.co/datasets/mteb/hifi-tts_t2a) • **License:** cc-by-4.0 • [Learn more →](https://openslr.org/109/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @article{bakhturina2021hi,
      author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
      journal = {arXiv preprint arXiv:2104.01497},
      title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
      year = {2021},
    }

    ```




#### ImageCoDeT2IRetrieval

Retrieve a specific video frame based on a precise caption.

**Dataset:** [`JamieSJS/imagecode`](https://huggingface.co/datasets/JamieSJS/imagecode) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.acl-long.241.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | cv_recall_at_3 | eng | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{krojer2022image,
      author = {Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
      journal = {arXiv preprint arXiv:2203.15867},
      title = {Image retrieval from contextual descriptions},
      year = {2022},
    }

    ```




#### InfoSeekIT2ITRetrieval

Retrieve source text and image information to answer questions about images.

**Dataset:** [`mteb/InfoSeekIT2ITRetrieval`](https://huggingface.co/datasets/mteb/InfoSeekIT2ITRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.925)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### InfoSeekIT2TRetrieval

Retrieve source information to answer questions about images.

**Dataset:** [`MRBench/mbeir_infoseek_task6`](https://huggingface.co/datasets/MRBench/mbeir_infoseek_task6) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.925)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### JLCorpusA2TRetrieval

Emotional speech segments from the JL-Corpus, balanced over long vowels and annotated for primary and secondary emotions.

**Dataset:** [`mteb/jl_corpus_a2t`](https://huggingface.co/datasets/mteb/jl_corpus_a2t) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/tli725/jl-corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{james2018open,
      author = {James, Jesin and Li, Tian and Watson, Catherine},
      booktitle = {Proc. Interspeech 2018},
      title = {An Open Source Emotional Speech Corpus for Human Robot Interaction Applications},
      year = {2018},
    }

    ```




#### JLCorpusT2ARetrieval

Emotional speech segments from the JL-Corpus, balanced over long vowels and annotated for primary and secondary emotions.

**Dataset:** [`mteb/jl_corpus_t2a`](https://huggingface.co/datasets/mteb/jl_corpus_t2a) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/tli725/jl-corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{james2018open,
      author = {James, Jesin and Li, Tian and Watson, Catherine},
      booktitle = {Proc. Interspeech 2018},
      title = {An Open Source Emotional Speech Corpus for Human Robot Interaction Applications},
      year = {2018},
    }

    ```




#### JamAltArtistA2ARetrieval

Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset

**Dataset:** [`jamendolyrics/jam-alt-lines`](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | ndcg_at_10 | deu, eng, fra, spa | Music | derived | found |



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




#### JamAltLyricA2TRetrieval

From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset

**Dataset:** [`jamendolyrics/jam-alt-lines`](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | ndcg_at_10 | deu, eng, fra, spa | Music | derived | found |



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




#### JamAltLyricT2ARetrieval

From textual lyrics (query), retrieve corresponding audio clips of songs from the Jam-Alt-Lines dataset

**Dataset:** [`jamendolyrics/jam-alt-lines`](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jamendolyrics/jam-alt-lines)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | ndcg_at_10 | deu, eng, fra, spa | Music | derived | found |



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




#### LLaVAIT2TRetrieval

Retrieve responses to answer questions about images.

**Dataset:** [`izhx/UMRB-LLaVA`](https://huggingface.co/datasets/izhx/UMRB-LLaVA) • **License:** cc-by-4.0 • [Learn more →](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_5 | eng | Encyclopaedic | derived | found |



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




#### LibriTTSA2TRetrieval

Given audiobook speech segments from the multi‑speaker LibriTTS corpus, retrieve the correct text transcription. LibriTTS is a 585‑hour, 24 kHz, multi‑speaker English TTS corpus derived from LibriVox (audio) and Project Gutenberg (text).

**Dataset:** [`mteb/LibriTTS_a2t`](https://huggingface.co/datasets/mteb/LibriTTS_a2t) • **License:** cc-by-4.0 • [Learn more →](https://www.openslr.org/60/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | Spoken | derived | found |



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




#### LibriTTST2ARetrieval

Given an English text transcription, retrieve its corresponding audiobook speech segment from the multi‑speaker LibriTTS corpus. LibriTTS is a 585‑hour, 24 kHz, multi‑speaker English TTS corpus derived from LibriVox and Project Gutenberg.

**Dataset:** [`mteb/LibriTTS_t2a`](https://huggingface.co/datasets/mteb/LibriTTS_t2a) • **License:** cc-by-4.0 • [Learn more →](https://www.openslr.org/60/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Spoken | derived | found |



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




#### MACSA2TRetrieval

Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset.

**Dataset:** [`mteb/MACS_a2t`](https://huggingface.co/datasets/mteb/MACS_a2t) • **License:** https://zenodo.org/records/5114771 • [Learn more →](https://zenodo.org/records/5114771)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | eng | AudioScene | human-annotated | found |



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




#### MACST2ARetrieval

Audio captions and tags for urban acoustic scenes in TAU Urban Acoustic Scenes 2019 development dataset.

**Dataset:** [`mteb/MACS_t2a`](https://huggingface.co/datasets/mteb/MACS_t2a) • **License:** https://zenodo.org/records/5114771 • [Learn more →](https://zenodo.org/records/5114771)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | AudioScene | human-annotated | found |



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




#### METI2IRetrieval

Retrieve photos of more than 224k artworks.

**Dataset:** [`JamieSJS/met`](https://huggingface.co/datasets/JamieSJS/met) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2202.01747)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{ypsilantis2021met,
      author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {The met dataset: Instance-level recognition for artworks},
      year = {2021},
    }

    ```




#### MSCOCOI2TRetrieval

Retrieve captions based on images.

**Dataset:** [`MRBench/mbeir_mscoco_task3`](https://huggingface.co/datasets/MRBench/mbeir_mscoco_task3) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### MSCOCOT2IRetrieval

Retrieve images based on captions.

**Dataset:** [`MRBench/mbeir_mscoco_task0`](https://huggingface.co/datasets/MRBench/mbeir_mscoco_task0) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### MemotionI2TRetrieval

Retrieve captions based on memes.

**Dataset:** [`Ahren09/MMSoc_Memotion`](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion) • **License:** mit • [Learn more →](https://aclanthology.org/2020.semeval-1.99/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### MemotionT2IRetrieval

Retrieve memes based on captions.

**Dataset:** [`Ahren09/MMSoc_Memotion`](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion) • **License:** mit • [Learn more →](https://aclanthology.org/2020.semeval-1.99/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### MusicCapsA2TRetrieval

Natural language description for music audio.

**Dataset:** [`mteb/MusicCaps_a2t`](https://huggingface.co/datasets/mteb/MusicCaps_a2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/nateraw/download-musiccaps-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | zxx | Music | human-annotated | found |



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




#### MusicCapsT2ARetrieval

Natural language description for music audio.

**Dataset:** [`mteb/MusicCaps_t2a`](https://huggingface.co/datasets/mteb/MusicCaps_t2a) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/nateraw/download-musiccaps-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | zxx | Music | human-annotated | found |



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




#### NIGHTSI2IRetrieval

Retrieval identical image to the given image.

**Dataset:** [`MRBench/mbeir_nights_task4`](https://huggingface.co/datasets/MRBench/mbeir_nights_task4) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### OKVQAIT2TRetrieval

Retrieval a Wiki passage to answer query about an image.

**Dataset:** [`izhx/UMRB-OKVQA`](https://huggingface.co/datasets/izhx/UMRB-OKVQA) • **License:** cc-by-4.0 • [Learn more →](https://okvqa.allenai.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_10 | eng | Encyclopaedic | derived | created |



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




#### OVENIT2ITRetrieval

Retrieval a Wiki image and passage to answer query about an image.

**Dataset:** [`MRBench/mbeir_oven_task8`](https://huggingface.co/datasets/MRBench/mbeir_oven_task8) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### OVENIT2TRetrieval

Retrieval a Wiki passage to answer query about an image.

**Dataset:** [`MRBench/mbeir_oven_task6`](https://huggingface.co/datasets/MRBench/mbeir_oven_task6) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### ROxfordEasyI2IRetrieval

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`JamieSJS/r-oxford-easy-multi`](https://huggingface.co/datasets/JamieSJS/r-oxford-easy-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



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




#### ROxfordHardI2IRetrieval

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`JamieSJS/r-oxford-hard-multi`](https://huggingface.co/datasets/JamieSJS/r-oxford-hard-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



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




#### ROxfordMediumI2IRetrieval

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`JamieSJS/r-oxford-medium-multi`](https://huggingface.co/datasets/JamieSJS/r-oxford-medium-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



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




#### RP2kI2IRetrieval

Retrieve photos of 39457 products.

**Dataset:** [`JamieSJS/rp2k`](https://huggingface.co/datasets/JamieSJS/rp2k) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2006.12634)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Web | derived | created |



??? quote "Citation"


    ```bibtex

    @article{peng2020rp2k,
      author = {Peng, Jingtian and Xiao, Chang and Li, Yifan},
      journal = {arXiv preprint arXiv:2006.12634},
      title = {RP2K: A large-scale retail product dataset for fine-grained image classification},
      year = {2020},
    }

    ```




#### RParisEasyI2IRetrieval

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`JamieSJS/r-paris-easy-multi`](https://huggingface.co/datasets/JamieSJS/r-paris-easy-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



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




#### RParisHardI2IRetrieval

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`JamieSJS/r-paris-hard-multi`](https://huggingface.co/datasets/JamieSJS/r-paris-hard-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



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




#### RParisMediumI2IRetrieval

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`JamieSJS/r-paris-medium-multi`](https://huggingface.co/datasets/JamieSJS/r-paris-medium-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



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




#### ReMuQIT2TRetrieval

Retrieval of a Wiki passage to answer a query about an image.

**Dataset:** [`izhx/UMRB-ReMuQ`](https://huggingface.co/datasets/izhx/UMRB-ReMuQ) • **License:** cc0-1.0 • [Learn more →](https://github.com/luomancs/ReMuQ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_5 | eng | Encyclopaedic | derived | created |



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




#### SOPI2IRetrieval

Retrieve product photos of 22634 online products.

**Dataset:** [`JamieSJS/stanford-online-products`](https://huggingface.co/datasets/JamieSJS/stanford-online-products) • **License:** not specified • [Learn more →](https://paperswithcode.com/dataset/stanford-online-products)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



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




#### SciMMIRI2TRetrieval

Retrieve captions based on figures and tables.

**Dataset:** [`m-a-p/SciMMIR`](https://huggingface.co/datasets/m-a-p/SciMMIR) • **License:** mit • [Learn more →](https://aclanthology.org/2024.findings-acl.746/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{wu2024scimmir,
      author = {Wu, Siwei and Li, Yizhi and Zhu, Kang and Zhang, Ge and Liang, Yiming and Ma, Kaijing and Xiao, Chenghao and Zhang, Haoran and Yang, Bohao and Chen, Wenhu and others},
      journal = {arXiv preprint arXiv:2401.13478},
      title = {SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval},
      year = {2024},
    }

    ```




#### SciMMIRT2IRetrieval

Retrieve figures and tables based on captions.

**Dataset:** [`m-a-p/SciMMIR`](https://huggingface.co/datasets/m-a-p/SciMMIR) • **License:** mit • [Learn more →](https://aclanthology.org/2024.findings-acl.746/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{wu2024scimmir,
      author = {Wu, Siwei and Li, Yizhi and Zhu, Kang and Zhang, Ge and Liang, Yiming and Ma, Kaijing and Xiao, Chenghao and Zhang, Haoran and Yang, Bohao and Chen, Wenhu and others},
      journal = {arXiv preprint arXiv:2401.13478},
      title = {SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval},
      year = {2024},
    }

    ```




#### SketchyI2IRetrieval

Retrieve photos from sketches.

**Dataset:** [`JamieSJS/sketchy`](https://huggingface.co/datasets/JamieSJS/sketchy) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2202.01747)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{ypsilantis2021met,
      author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {The met dataset: Instance-level recognition for artworks},
      year = {2021},
    }

    ```




#### SoundDescsA2TRetrieval

Natural language description for different audio sources from the BBC Sound Effects webpage.

**Dataset:** [`mteb/sounddescs_a2t`](https://huggingface.co/datasets/mteb/sounddescs_a2t) • **License:** apache-2.0 • [Learn more →](https://github.com/akoepke/audio-retrieval-benchmark)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | zxx | Encyclopaedic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Koepke2022,
      author = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
      booktitle = {IEEE Transactions on Multimedia},
      title = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
      year = {2022},
    }

    ```




#### SoundDescsT2ARetrieval

Natural language description for different audio sources from the BBC Sound Effects webpage.

**Dataset:** [`mteb/sounddescs_t2a`](https://huggingface.co/datasets/mteb/sounddescs_t2a) • **License:** apache-2.0 • [Learn more →](https://github.com/akoepke/audio-retrieval-benchmark)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | zxx | Encyclopaedic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Koepke2022,
      author = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
      booktitle = {IEEE Transactions on Multimedia},
      title = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
      year = {2022},
    }

    ```




#### SpokenSQuADT2ARetrieval

Text-to-audio retrieval task based on SpokenSQuAD dataset. Given a text question, retrieve relevant audio segments that contain the answer. Questions are derived from SQuAD reading comprehension dataset with corresponding spoken passages.

**Dataset:** [`arteemg/spoken-squad-t2a`](https://huggingface.co/datasets/arteemg/spoken-squad-t2a) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/chiuwy/Spoken-SQuAD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | eng | Academic, Encyclopaedic, Non-fiction | derived | found |



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




#### StanfordCarsI2IRetrieval

Retrieve car images from 196 makes.

**Dataset:** [`isaacchung/stanford_cars_retrieval`](https://huggingface.co/datasets/isaacchung/stanford_cars_retrieval) • **License:** not specified • [Learn more →](https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{Krause2013CollectingAL,
      author = {Jonathan Krause and Jia Deng and Michael Stark and Li Fei-Fei},
      title = {Collecting a Large-scale Dataset of Fine-grained Cars},
      url = {https://api.semanticscholar.org/CorpusID:16632981},
      year = {2013},
    }

    ```




#### TUBerlinT2IRetrieval

Retrieve sketch images based on text descriptions.

**Dataset:** [`gowitheflow/tu-berlin`](https://huggingface.co/datasets/gowitheflow/tu-berlin) • **License:** cc-by-sa-4.0 • [Learn more →](https://dl.acm.org/doi/pdf/10.1145/2185520.2185540)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



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




#### UrbanSound8KA2TRetrieval

UrbanSound8K: Audio-to-text retrieval of urban sound events.

**Dataset:** [`mteb/Urbansound8K_a2t`](https://huggingface.co/datasets/mteb/Urbansound8K_a2t) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/CLAPv2/Urbansound8K)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | cv_recall_at_5 | zxx | AudioScene | human-annotated | found |



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




#### UrbanSound8KT2ARetrieval

UrbanSound8K: Text-to-audio retrieval of urban sound events.

**Dataset:** [`mteb/Urbansound8K_t2a`](https://huggingface.co/datasets/mteb/Urbansound8K_t2a) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/CLAPv2/Urbansound8K)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to audio (t2a) | cv_recall_at_5 | zxx | AudioScene | human-annotated | found |



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




#### VQA2IT2TRetrieval

Retrieve the correct answer for a question about an image.

**Dataset:** [`JamieSJS/vqa-2`](https://huggingface.co/datasets/JamieSJS/vqa-2) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Web | derived | found |



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




#### VisualNewsI2TRetrieval

Retrieval entity-rich captions for news images.

**Dataset:** [`MRBench/mbeir_visualnews_task3`](https://huggingface.co/datasets/MRBench/mbeir_visualnews_task3) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.emnlp-main.542/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### VisualNewsT2IRetrieval

Retrieve news images with captions.

**Dataset:** [`MRBench/mbeir_visualnews_task0`](https://huggingface.co/datasets/MRBench/mbeir_visualnews_task0) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.emnlp-main.542/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### VizWizIT2TRetrieval

Retrieve the correct answer for a question about an image.

**Dataset:** [`JamieSJS/vizwiz`](https://huggingface.co/datasets/JamieSJS/vizwiz) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Web | derived | found |



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




#### WebQAT2ITRetrieval

Retrieve sources of information based on questions.

**Dataset:** [`MRBench/mbeir_webqa_task2`](https://huggingface.co/datasets/MRBench/mbeir_webqa_task2) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image, text (t2it) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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




#### WebQAT2TRetrieval

Retrieve sources of information based on questions.

**Dataset:** [`MRBench/mbeir_webqa_task1`](https://huggingface.co/datasets/MRBench/mbeir_webqa_task1) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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
