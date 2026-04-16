
# AudioClassification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 35

#### AmbientAcousticContext

The Ambient Acoustic Context dataset contains 1-second segments for activities that occur in a workplace setting. This is a downsampled version with ~100 train and ~50 test samples per class.

**Dataset:** [`mteb/ambient-acoustic-context-small`](https://huggingface.co/datasets/mteb/ambient-acoustic-context-small) • **License:** not specified • [Learn more →](https://dl.acm.org/doi/10.1145/3379503.3403535)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Speech, Spoken | human-annotated | found |



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




#### BeijingOpera

Audio classification of percussion instruments into one of 4 classes: `Bangu`, `Naobo`, `Daluo`, and `Xiaoluo`

**Dataset:** [`mteb/beijing-opera`](https://huggingface.co/datasets/mteb/beijing-opera) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/silky1708/BeijingOpera)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Music | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{6853981,
      author = {Tian, Mi and Srinivasamurthy, Ajay and Sandler, Mark and Serra, Xavier},
      booktitle = {2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      doi = {10.1109/ICASSP.2014.6853981},
      keywords = {Decision support systems;Conferences;Acoustics;Speech;Speech processing;Time-frequency analysis;Beijing Opera;Onset Detection;Drum Transcription;Non-negative matrix factorization},
      number = {},
      pages = {2159-2163},
      title = {A study of instrument-wise onset detection in Beijing Opera percussion ensembles},
      volume = {},
      year = {2014},
    }

    ```




#### BirdCLEF

BirdCLEF+ 2025 dataset for species identification from audio, focused on birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia. Downsampled to 50 classes with 20 samples each.

**Dataset:** [`mteb/birdclef25-mini`](https://huggingface.co/datasets/mteb/birdclef25-mini) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/christopher/birdclef-2025)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Bioacoustics, Speech, Spoken | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @dataset{birdclef2025,
      author = {Christopher},
      publisher = {Hugging Face},
      title = {BirdCLEF+ 2025},
      url = {https://huggingface.co/datasets/christopher/birdclef-2025},
      year = {2025},
    }

    ```




#### CREMA_D

Emotion classification of audio into one of 6 classes: Anger, Disgust, Fear, Happy, Neutral, Sad.

**Dataset:** [`mteb/crema-d`](https://huggingface.co/datasets/mteb/crema-d) • **License:** http://opendatacommons.org/licenses/odbl/1.0/ • [Learn more →](https://huggingface.co/datasets/silky1708/CREMA-D)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech | human-annotated | created |



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




#### CSTRVCTKAccentID

Gender classification from CSTR-VCTK dataset. This is a stratified and downsampled version of the original dataset. The dataset was recorded with 2 different microphones, and this mini version uniformly samples data from the 2 microphone types.

**Dataset:** [`mteb/cstr-vctk-accent-mini`](https://huggingface.co/datasets/mteb/cstr-vctk-accent-mini) • **License:** cc-by-4.0 • [Learn more →](https://datashare.ed.ac.uk/handle/10283/3443)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Yamagishi2019CSTRVC,
      author = {Junichi Yamagishi and Christophe Veaux and Kirsten MacDonald},
      title = {CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)},
      url = {https://api.semanticscholar.org/CorpusID:213060286},
      year = {2019},
    }

    ```




#### CSTRVCTKGender

Gender classification from CSTR-VCTK dataset. This is a stratified and downsampled version of the original dataset. The dataset was recorded with 2 different microphones, and this mini version uniformly samples data from the 2 microphone types.

**Dataset:** [`mteb/cstr-vctk-gender-mini`](https://huggingface.co/datasets/mteb/cstr-vctk-gender-mini) • **License:** cc-by-4.0 • [Learn more →](https://datashare.ed.ac.uk/handle/10283/3443)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Yamagishi2019CSTRVC,
      author = {Junichi Yamagishi and Christophe Veaux and Kirsten MacDonald},
      title = {CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)},
      url = {https://api.semanticscholar.org/CorpusID:213060286},
      year = {2019},
    }

    ```




#### CommonLanguageAgeDetection

Age Classification. This is a stratified subsampled version of the original CommonLanguage dataset.

**Dataset:** [`mteb/commonlanguage-age-mini`](https://huggingface.co/datasets/mteb/commonlanguage-age-mini) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/speechbrain/common_language)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Scene, Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @dataset{ganesh_sinisetty_2021_5036977,
      author = {Ganesh Sinisetty and
    Pavlo Ruban and
    Oleksandr Dymov and
    Mirco Ravanelli},
      doi = {10.5281/zenodo.5036977},
      month = jun,
      publisher = {Zenodo},
      title = {CommonLanguage},
      url = {https://doi.org/10.5281/zenodo.5036977},
      version = {0.1},
      year = {2021},
    }

    ```




#### CommonLanguageGenderDetection

Gender Classification. This is a stratified subsampled version of the original CommonLanguage datasets.

**Dataset:** [`mteb/commonlanguage-gender-mini`](https://huggingface.co/datasets/mteb/commonlanguage-gender-mini) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/speechbrain/common_language)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Scene, Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @dataset{ganesh_sinisetty_2021_5036977,
      author = {Ganesh Sinisetty and
    Pavlo Ruban and
    Oleksandr Dymov and
    Mirco Ravanelli},
      doi = {10.5281/zenodo.5036977},
      month = jun,
      publisher = {Zenodo},
      title = {CommonLanguage},
      url = {https://doi.org/10.5281/zenodo.5036977},
      version = {0.1},
      year = {2021},
    }

    ```




#### CommonLanguageLanguageDetection

Language Classification. This is a stratified subsampled version of the original CommonLanguage dataset.

**Dataset:** [`mteb/commonlanguage-lang-mini`](https://huggingface.co/datasets/mteb/commonlanguage-lang-mini) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/speechbrain/common_language)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Scene, Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @dataset{ganesh_sinisetty_2021_5036977,
      author = {Ganesh Sinisetty and
    Pavlo Ruban and
    Oleksandr Dymov and
    Mirco Ravanelli},
      doi = {10.5281/zenodo.5036977},
      month = jun,
      publisher = {Zenodo},
      title = {CommonLanguage},
      url = {https://doi.org/10.5281/zenodo.5036977},
      version = {0.1},
      year = {2021},
    }

    ```




#### ESC50

Environmental Sound Classification Dataset.

**Dataset:** [`mteb/esc50`](https://huggingface.co/datasets/mteb/esc50) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ashraq/esc50)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Spoken | human-annotated | found |



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




#### ExpressoConv

Multiclass expressive speech style classification. This is a stratfied and downsampled version of the original dataset that contains 40 hours of speech. The original dataset has two subsets - read speech and conversational speech, each having their own set of style labels. This task only includes the conversational speech subset.

**Dataset:** [`mteb/expresso-conv-mini`](https://huggingface.co/datasets/mteb/expresso-conv-mini) • **License:** cc-by-nc-4.0 • [Learn more →](https://speechbot.github.io/expresso/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Speech, Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{nguyen2023expresso,
      author = {Nguyen, Tu Anh and Hsu, Wei-Ning and d'Avirro, Antony and Shi, Bowen and Gat, Itai and Fazel-Zarani, Maryam and Remez, Tal and Copet, Jade and Synnaeve, Gabriel and Hassid, Michael and others},
      booktitle = {INTERSPEECH 2023-24th Annual Conference of the International Speech Communication Association},
      pages = {4823--4827},
      title = {Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis},
      year = {2023},
    }

    ```




#### ExpressoRead

Multiclass expressive speech style classification. This is a stratfied and downsampled version of the original dataset that contains 40 hours of speech. The original dataset has two subsets - read speech and conversational speech, each having their own set of style labels. This task only includes the read speech subset.

**Dataset:** [`mteb/expresso-read-mini`](https://huggingface.co/datasets/mteb/expresso-read-mini) • **License:** cc-by-nc-4.0 • [Learn more →](https://speechbot.github.io/expresso/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Speech, Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{nguyen2023expresso,
      author = {Nguyen, Tu Anh and Hsu, Wei-Ning and d'Avirro, Antony and Shi, Bowen and Gat, Itai and Fazel-Zarani, Maryam and Remez, Tal and Copet, Jade and Synnaeve, Gabriel and Hassid, Michael and others},
      booktitle = {INTERSPEECH 2023-24th Annual Conference of the International Speech Communication Association},
      pages = {4823--4827},
      title = {Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis},
      year = {2023},
    }

    ```




#### FSDD

Spoken digit classification of audio into one of 10 classes: 0-9

**Dataset:** [`mteb/free-spoken-digit-dataset`](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/silky1708/Free-Spoken-Digit-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Music | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{zohar2018free,
      author = {J. Zohar and S. Cãar and F. Jason and P. Yuxin and N. Hereman and T. Adhish},
      month = {aug},
      title = {Jakobovski/Free-Spoken-Digit-Dataset: V1.0.8},
      url = {https://doi.org/10.5281/zenodo.1342401},
      year = {2018},
    }

    ```




#### GLOBEV2Age

Age classification from the GLOBE v2 dataset (sampled and enhanced from CommonVoice dataset for TTS purpose). This dataset is a stratified and downsampled version of the original dataset, containing about 535 hours of speech data across 164 accents. We use the age column as the target label for audio classification.

**Dataset:** [`mteb/globe-v2-age-mini`](https://huggingface.co/datasets/mteb/globe-v2-age-mini) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/MushanW/GLOBE_V2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{wang2024globe,
      archiveprefix = {arXiv},
      author = {Wenbin Wang and Yang Song and Sanjay Jha},
      eprint = {2406.14875},
      title = {GLOBE: A High-quality English Corpus with Global Accents for Zero-shot Speaker Adaptive Text-to-Speech},
      year = {2024},
    }

    ```




#### GLOBEV2Gender

Gender classification from the GLOBE v2 dataset (sampled and enhanced from CommonVoice dataset for TTS purpose). This dataset is a stratified and downsampled version of the original dataset, containing about 535 hours of speech data across 164 accents. We use the gender column as the target label for audio classification.

**Dataset:** [`mteb/globe-v2-gender-mini`](https://huggingface.co/datasets/mteb/globe-v2-gender-mini) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/MushanW/GLOBE_V2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to text (a2t) | accuracy | eng | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{wang2024globe,
      archiveprefix = {arXiv},
      author = {Wenbin Wang and Yang Song and Sanjay Jha},
      eprint = {2406.14875},
      title = {GLOBE: A High-quality English Corpus with Global Accents for Zero-shot Speaker Adaptive Text-to-Speech},
      year = {2024},
    }

    ```




#### GTZANGenre

Music Genre Classification (10 classes)

**Dataset:** [`mteb/gtzan-genre`](https://huggingface.co/datasets/mteb/gtzan-genre) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/silky1708/GTZAN-Genre)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Music | human-annotated | found |



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




#### GunshotTriangulation

Classifying a weapon based on its muzzle blast

**Dataset:** [`mteb/GunshotTriangulationHear`](https://huggingface.co/datasets/mteb/GunshotTriangulationHear) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/anime-sh/GunshotTriangulationHEAR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{raponi2021soundgunsdigitalforensics,
      archiveprefix = {arXiv},
      author = {Simone Raponi and Isra Ali and Gabriele Oligeri},
      eprint = {2004.07948},
      primaryclass = {eess.AS},
      title = {Sound of Guns: Digital Forensics of Gun Audio Samples meets Artificial Intelligence},
      url = {https://arxiv.org/abs/2004.07948},
      year = {2021},
    }

    ```




#### IEMOCAPEmotion

Classification of speech samples into emotions (angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted) from interactive emotional dyadic conversations.

**Dataset:** [`mteb/iemocap`](https://huggingface.co/datasets/mteb/iemocap) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://doi.org/10.1007/s10579-008-9076-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech, Spoken | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @article{busso2008iemocap,
      author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
      journal = {Language resources and evaluation},
      number = {4},
      pages = {335--359},
      publisher = {Springer},
      title = {IEMOCAP: Interactive emotional dyadic motion capture database},
      volume = {42},
      year = {2008},
    }

    ```




#### IEMOCAPGender

Classification of speech samples by speaker gender (male/female) from the IEMOCAP database of interactive emotional dyadic conversations.

**Dataset:** [`mteb/iemocap`](https://huggingface.co/datasets/mteb/iemocap) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://doi.org/10.1007/s10579-008-9076-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech, Spoken | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @article{busso2008iemocap,
      author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
      journal = {Language resources and evaluation},
      number = {4},
      pages = {335--359},
      publisher = {Springer},
      title = {IEMOCAP: Interactive emotional dyadic motion capture database},
      volume = {42},
      year = {2008},
    }

    ```




#### LibriCount

Multiclass speaker count identification. Dataset contains audio recordings with between 0 to 10 speakers.

**Dataset:** [`mteb/libricount`](https://huggingface.co/datasets/mteb/libricount) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/silky1708/LibriCount)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech | algorithmic | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{Stoter_2018,
      author = {Stoter, Fabian-Robert and Chakrabarty, Soumitro and Edler, Bernd and Habets, Emanuel A. P.},
      booktitle = {2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      doi = {10.1109/icassp.2018.8462159},
      month = apr,
      pages = {436-440},
      publisher = {IEEE},
      title = {Classification vs. Regression in Supervised Learning for Single Channel Speaker Count Estimation},
      url = {http://dx.doi.org/10.1109/ICASSP.2018.8462159},
      year = {2018},
    }

    ```




#### MInDS14

MInDS-14 is an evaluation resource for intent detection with spoken data in 14 diverse languages.

**Dataset:** [`mteb/minds14-multilingual`](https://huggingface.co/datasets/mteb/minds14-multilingual) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2104.08524)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | ces, deu, eng, fra, ita, ... (12) | Speech, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{DBLP:journals/corr/abs-2104-08524,
      author = {Daniela Gerz and Pei{-}Hao Su and Razvan Kusztos and Avishek Mondal and Michal Lis and Eshan Singhal and Nikola Mrkšić and Tsung{-}Hsien Wen and Ivan Vulic},
      eprint = {2104.08524},
      eprinttype = {arXiv},
      journal = {CoRR},
      title = {Multilingual and Cross-Lingual Intent Detection from Spoken Data},
      url = {https://arxiv.org/abs/2104.08524},
      volume = {abs/2104.08524},
      year = {2021},
    }

    ```




#### MridinghamStroke

Stroke classification of Mridingham (a pitched percussion instrument) into one of 10 classes: ["bheem", "cha", "dheem", "dhin", "num", "tham", "ta", "tha", "thi", "thom"]

**Dataset:** [`mteb/mridingham-stroke`](https://huggingface.co/datasets/mteb/mridingham-stroke) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/silky1708/Mridingham-Stroke)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Music | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{6637633,
      author = {Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A},
      booktitle = {2013 IEEE International Conference on Acoustics, Speech and Signal Processing},
      doi = {10.1109/ICASSP.2013.6637633},
      keywords = {Instruments;Vectors;Hidden Markov models;Harmonic analysis;Modal analysis;Dictionaries;Music;Modal Analysis;Mridangam;automatic transcription;Non-negative Matrix Factorization;Hidden Markov models},
      number = {},
      pages = {181-185},
      title = {Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization},
      volume = {},
      year = {2013},
    }

    ```




#### MridinghamTonic

Tonic classification of Mridingham (a pitched percussion instrument) into one of 6 classes: B,C,C#,D,D#,E

**Dataset:** [`mteb/mridingham-tonic`](https://huggingface.co/datasets/mteb/mridingham-tonic) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/silky1708/Mridingham-Tonic)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Music | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{6637633,
      author = {Anantapadmanabhan, Akshay and Bellur, Ashwin and Murthy, Hema A},
      booktitle = {2013 IEEE International Conference on Acoustics, Speech and Signal Processing},
      doi = {10.1109/ICASSP.2013.6637633},
      keywords = {Instruments;Vectors;Hidden Markov models;Harmonic analysis;Modal analysis;Dictionaries;Music;Modal Analysis;Mridangam;automatic transcription;Non-negative Matrix Factorization;Hidden Markov models},
      number = {},
      pages = {181-185},
      title = {Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization},
      volume = {},
      year = {2013},
    }

    ```




#### NSynth

Instrument Source Classification: one of acoustic, electronic, or synthetic.

**Dataset:** [`mteb/nsynth-mini`](https://huggingface.co/datasets/mteb/nsynth-mini) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/anime-sh/NSYNTH_PITCH_HEAR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | Music | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{engel2017neuralaudiosynthesismusical,
      archiveprefix = {arXiv},
      author = {Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Douglas Eck and Karen Simonyan and Mohammad Norouzi},
      eprint = {1704.01279},
      primaryclass = {cs.LG},
      title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
      url = {https://arxiv.org/abs/1704.01279},
      year = {2017},
    }

    ```




#### SpeechCommands

A set of one-second .wav audio files, each containing a single spoken English word or background noise. To keep evaluation fast, we use a downsampled version of the original dataset by keeping ~50 samples per class for training.

**Dataset:** [`mteb/speech-commands-mini`](https://huggingface.co/datasets/mteb/speech-commands-mini) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/1804.03209)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{speechcommands2018,
      author = {Pete Warden},
      journal = {arXiv preprint arXiv:1804.03209},
      title = {Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
      year = {2018},
    }

    ```




#### SpokeNEnglish

Human Sound Classification Dataset.

**Dataset:** [`mteb/SpokeN-100-English`](https://huggingface.co/datasets/mteb/SpokeN-100-English) • **License:** cc-by-sa-4.0 • [Learn more →](https://zenodo.org/records/10810044)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Spoken | LM-generated | found |



??? quote "Citation"


    ```bibtex

    @misc{groh2024spoken100crosslingualbenchmarkingdataset,
      archiveprefix = {arXiv},
      author = {René Groh and Nina Goes and Andreas M. Kist},
      eprint = {2403.09753},
      primaryclass = {cs.SD},
      title = {SpokeN-100: A Cross-Lingual Benchmarking Dataset for The Classification of Spoken Numbers in Different Languages},
      url = {https://arxiv.org/abs/2403.09753},
      year = {2024},
    }

    ```




#### SpokenQAForIC

SpokenQA dataset reformulated as Intent Classification (IC) task

**Dataset:** [`mteb/SpokenQA_SLUE`](https://huggingface.co/datasets/mteb/SpokenQA_SLUE) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/DynamicSuperb/SpokenQA_SLUE)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Spoken | human-annotated | multiple |



??? quote "Citation"


    ```bibtex

    @misc{shon2023sluephase2benchmarksuite,
      archiveprefix = {arXiv},
      author = {Suwon Shon and Siddhant Arora and Chyi-Jiunn Lin and Ankita Pasad and Felix Wu and Roshan Sharma and Wei-Lun Wu and Hung-Yi Lee and Karen Livescu and Shinji Watanabe},
      eprint = {2212.10525},
      primaryclass = {cs.CL},
      title = {SLUE Phase-2: A Benchmark Suite of Diverse Spoken Language Understanding Tasks},
      url = {https://arxiv.org/abs/2212.10525},
      year = {2023},
    }

    ```




#### TAUAcousticScenes2022Mobile

TAU Urban Acoustic Scenes 2022 Mobile, development dataset consists of 1-second audio recordings from 12 European cities in 10 different acoustic scenes using 4 different devices. This is a stratified subsampled version of the evaluation_setup subset of the original dataset.

**Dataset:** [`mteb/tau-acoustic-scenes-2022-mobile-mini`](https://huggingface.co/datasets/mteb/tau-acoustic-scenes-2022-mobile-mini) • **License:** not specified • [Learn more →](https://zenodo.org/records/6337421)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | AudioScene | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @dataset{heittola_2022_6337421,
      author = {Toni Heittola and Annamaria Mesaros and Tuomas Virtanen},
      publisher = {Zenodo},
      title = {TAU Urban Acoustic Scenes 2022 Mobile, Development Dataset},
      url = {https://doi.org/10.5281/zenodo.6337421},
      year = {2022},
    }

    ```




#### TUTAcousticScenes

TUT Urban Acoustic Scenes 2018 dataset consists of 10-second audio segments from 10 acoustic scenes recorded in six European cities. This is a stratified subsampled version of the original dataset.

**Dataset:** [`mteb/tut-acoustic-scenes-mini`](https://huggingface.co/datasets/mteb/tut-acoustic-scenes-mini) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/record/1228142)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | zxx | AudioScene | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Mesaros2018_DCASE,
      address = {Tampere, Finland},
      author = {Annamaria Mesaros and Toni Heittola and Tuomas Virtanen},
      booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
      publisher = {Tampere University of Technology},
      title = {A Multi-Device Dataset for Urban Acoustic Scene Classification},
      url = {https://arxiv.org/abs/1807.09840},
      year = {2018},
    }

    ```




#### VocalSound

Human Vocal Sound Classification Dataset.

**Dataset:** [`mteb/vocalsound`](https://huggingface.co/datasets/mteb/vocalsound) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/lmms-lab/vocalsound)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Spoken | human-annotated | found |



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




#### VoxCelebSA

VoxCeleb dataset augmented for Sentiment Analysis task

**Dataset:** [`mteb/voxceleb-sentiment`](https://huggingface.co/datasets/mteb/voxceleb-sentiment) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Spoken | human-annotated | found |



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




#### VoxLingua107_Top10

Spoken Language Identification for a given audio samples (10 classes/languages)

**Dataset:** [`mteb/voxlingua107-top10`](https://huggingface.co/datasets/mteb/voxlingua107-top10) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/silky1708/VoxLingua107-Top-10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech | automatic-and-reviewed | found |



??? quote "Citation"


    ```bibtex

    @misc{valk2020voxlingua107datasetspokenlanguage,
      archiveprefix = {arXiv},
      author = {Jörgen Valk and Tanel Alumäe},
      eprint = {2011.12998},
      primaryclass = {eess.AS},
      title = {VoxLingua107: a Dataset for Spoken Language Recognition},
      url = {https://arxiv.org/abs/2011.12998},
      year = {2020},
    }

    ```




#### VoxPopuliAccentID

Classification of English speech samples into one of 15 non-native accents from European Parliament recordings. This is a stratified subsampled version of the original VoxPopuli dataset.

**Dataset:** [`mteb/voxpopuli-accent-mini`](https://huggingface.co/datasets/mteb/voxpopuli-accent-mini) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/facebook/voxpopuli)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | eng | Speech, Spoken | human-annotated | found |



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




#### VoxPopuliGenderID

Subsampled Dataset Classification of speech samples by speaker gender (male/female) from European Parliament recordings.

**Dataset:** [`mteb/voxpopuli-mini`](https://huggingface.co/datasets/mteb/voxpopuli-mini) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/facebook/voxpopuli)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | deu, eng, fra, pol, spa | Speech, Spoken | human-annotated | found |



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




#### VoxPopuliLanguageID

Subsampled Dataset for classification of speech samples into one of 5 European languages (English, German, French, Spanish, Polish) from European Parliament recordings.

**Dataset:** [`mteb/voxpopuli-mini`](https://huggingface.co/datasets/mteb/voxpopuli-mini) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/facebook/voxpopuli)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to category (a2c) | accuracy | deu, eng, fra, pol, spa | Speech, Spoken | human-annotated | found |



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
