---
icon: lucide/chart-network
title: "Clustering"
---

<style>
.nowrap-table th {
  white-space: nowrap;
}
</style>

# Clustering

<!-- The following sections are auto-generated, please edit the construction script -->

<!-- START-TASKS -->

## AudioClustering

- **Number of tasks:** 10

#### AmbientAcousticContextClustering

Clustering task based on a subset of the Ambient Acoustic Context dataset containing 1-second segments for workplace activities.

**Dataset:** [`mteb/ambient-acoustic-context-small`](https://huggingface.co/datasets/mteb/ambient-acoustic-context-small) • **License:** not specified • [Learn more →](https://dl.acm.org/doi/10.1145/3379503.3403535)

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

**Dataset:** [`mteb/crema-d`](https://huggingface.co/datasets/mteb/crema-d) • **License:** http://opendatacommons.org/licenses/odbl/1.0/ • [Learn more →](https://ieeexplore.ieee.org/document/6849440)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| audio to audio (a2a) | v_measure | eng | Speech | human-annotated | created |



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
    



#### ESC50Clustering

The ESC-50 dataset contains 2,000 labeled environmental audio recordings evenly distributed across 50 classes (40 clips per class). These classes are organized into 5 broad categories: animal sounds, natural soundscapes & water sounds, human (non-speech) sounds, interior/domestic sounds, and exterior/urban noises. This task evaluates unsupervised clustering performance on environmental audio recordings.

**Dataset:** [`mteb/esc50`](https://huggingface.co/datasets/mteb/esc50) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ashraq/esc50)

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

**Dataset:** [`mteb/gtzan-genre`](https://huggingface.co/datasets/mteb/gtzan-genre) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/silky1708/GTZAN-Genre)

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

**Dataset:** [`mteb/Vehicle_sounds_classification_dataset`](https://huggingface.co/datasets/mteb/Vehicle_sounds_classification_dataset) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/DynamicSuperb/Vehicle_sounds_classification_dataset)

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

**Dataset:** [`mteb/VoiceGenderClustering`](https://huggingface.co/datasets/mteb/VoiceGenderClustering) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mmn3690/voice-gender-clustering)

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

**Dataset:** [`mteb/Sentiment_Analysis_SLUE-VoxCeleb`](https://huggingface.co/datasets/mteb/Sentiment_Analysis_SLUE-VoxCeleb) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb)

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

**Dataset:** [`mteb/mini-voxpopuli`](https://huggingface.co/datasets/mteb/mini-voxpopuli) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/facebook/voxpopuli)

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



## Clustering

- **Number of tasks:** 109

#### AlloProfClusteringP2P

Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`mteb/AlloProfClusteringP2P`](https://huggingface.co/datasets/mteb/AlloProfClusteringP2P) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



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
    



#### AlloProfClusteringP2P.v2

Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`mteb/AlloProfClusteringP2P.v2`](https://huggingface.co/datasets/mteb/AlloProfClusteringP2P.v2) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



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
    



#### AlloProfClusteringS2S

Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`mteb/AlloProfClusteringS2S`](https://huggingface.co/datasets/mteb/AlloProfClusteringS2S) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



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
    



#### AlloProfClusteringS2S.v2

Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`mteb/AlloProfClusteringS2S.v2`](https://huggingface.co/datasets/mteb/AlloProfClusteringS2S.v2) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



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
    



#### ArXivHierarchicalClusteringP2P

Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/arxiv-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



#### ArXivHierarchicalClusteringS2S

Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



#### ArxivClusteringP2P

Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/arxiv-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{arxiv_org_submitters_2024,
      author = {arXiv.org submitters},
      doi = {10.34740/KAGGLE/DSV/7548853},
      publisher = {Kaggle},
      title = {arXiv Dataset},
      url = {https://www.kaggle.com/dsv/7548853},
      year = {2024},
    }
    
    ```
    



#### ArxivClusteringP2P.v2

Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/arxiv-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{arxiv_org_submitters_2024,
      author = {arXiv.org submitters},
      doi = {10.34740/KAGGLE/DSV/7548853},
      publisher = {Kaggle},
      title = {arXiv Dataset},
      url = {https://www.kaggle.com/dsv/7548853},
      year = {2024},
    }
    
    ```
    



#### ArxivClusteringS2S

Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{arxiv_org_submitters_2024,
      author = {arXiv.org submitters},
      doi = {10.34740/KAGGLE/DSV/7548853},
      publisher = {Kaggle},
      title = {arXiv Dataset},
      url = {https://www.kaggle.com/dsv/7548853},
      year = {2024},
    }
    
    ```
    



#### BeytooteClustering

Beytoote Website Articles Clustering

**Dataset:** [`MCINext/beytoote-clustering`](https://huggingface.co/datasets/MCINext/beytoote-clustering) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | News | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### BigPatentClustering

Clustering of documents from the Big Patent dataset. Test set only includes documents belonging to a single category, with a total of 9 categories.

**Dataset:** [`jinaai/big-patent-clustering`](https://huggingface.co/datasets/jinaai/big-patent-clustering) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NortheasternUniversity/big_patent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/abs-1906-03741,
      author = {Eva Sharma and
    Chen Li and
    Lu Wang},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-1906-03741.bib},
      eprint = {1906.03741},
      eprinttype = {arXiv},
      journal = {CoRR},
      timestamp = {Wed, 26 Jun 2019 07:14:58 +0200},
      title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
      url = {http://arxiv.org/abs/1906.03741},
      volume = {abs/1906.03741},
      year = {2019},
    }
    
    ```
    



#### BigPatentClustering.v2

Clustering of documents from the Big Patent dataset. Test set only includes documents belonging to a single category, with a total of 9 categories.

**Dataset:** [`mteb/big-patent`](https://huggingface.co/datasets/mteb/big-patent) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NortheasternUniversity/big_patent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/abs-1906-03741,
      author = {Eva Sharma and
    Chen Li and
    Lu Wang},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-1906-03741.bib},
      eprint = {1906.03741},
      eprinttype = {arXiv},
      journal = {CoRR},
      timestamp = {Wed, 26 Jun 2019 07:14:58 +0200},
      title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
      url = {http://arxiv.org/abs/1906.03741},
      volume = {abs/1906.03741},
      year = {2019},
    }
    
    ```
    



#### BiorxivClusteringP2P

Clustering of titles+abstract from biorxiv. Clustering of 10 sets, based on the main category.

**Dataset:** [`mteb/biorxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/biorxiv-clustering-p2p) • **License:** https://www.biorxiv.org/content/about-biorxiv • [Learn more →](https://api.biorxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | created |



#### BiorxivClusteringP2P.v2

Clustering of titles+abstract from biorxiv across 26 categories.

**Dataset:** [`mteb/biorxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/biorxiv-clustering-p2p) • **License:** https://www.biorxiv.org/content/about-biorxiv • [Learn more →](https://api.biorxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | created |



#### BiorxivClusteringS2S

Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.

**Dataset:** [`mteb/biorxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/biorxiv-clustering-s2s) • **License:** https://www.biorxiv.org/content/about-biorxiv • [Learn more →](https://api.biorxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | created |



#### BiorxivClusteringS2S.v2

Clustering of titles from biorxiv across 26 categories.

**Dataset:** [`mteb/biorxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/biorxiv-clustering-s2s) • **License:** https://www.biorxiv.org/content/about-biorxiv • [Learn more →](https://api.biorxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | created |



#### BlurbsClusteringP2P

Clustering of book titles+blurbs. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p) • **License:** cc-by-nc-4.0 • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Remus2019GermEval2T,
      author = {Steffen Remus and Rami Aly and Chris Biemann},
      booktitle = {Conference on Natural Language Processing},
      title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
      url = {https://api.semanticscholar.org/CorpusID:208334484},
      year = {2019},
    }
    
    ```
    



#### BlurbsClusteringP2P.v2

Clustering of book titles+blurbs. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p) • **License:** cc-by-nc-4.0 • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Remus2019GermEval2T,
      author = {Steffen Remus and Rami Aly and Chris Biemann},
      booktitle = {Conference on Natural Language Processing},
      title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
      url = {https://api.semanticscholar.org/CorpusID:208334484},
      year = {2019},
    }
    
    ```
    



#### BlurbsClusteringS2S

Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s) • **License:** cc-by-nc-4.0 • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Remus2019GermEval2T,
      author = {Steffen Remus and Rami Aly and Chris Biemann},
      booktitle = {Conference on Natural Language Processing},
      title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
      url = {https://api.semanticscholar.org/CorpusID:208334484},
      year = {2019},
    }
    
    ```
    



#### BlurbsClusteringS2S.v2

Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s) • **License:** cc-by-nc-4.0 • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Remus2019GermEval2T,
      author = {Steffen Remus and Rami Aly and Chris Biemann},
      booktitle = {Conference on Natural Language Processing},
      title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
      url = {https://api.semanticscholar.org/CorpusID:208334484},
      year = {2019},
    }
    
    ```
    



#### BuiltBenchClusteringP2P

Clustering of built asset item descriptions based on categories identified within industry classification systems such as IFC, Uniclass, etc.

**Dataset:** [`mehrzad-shahin/BuiltBench-clustering-p2p`](https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-clustering-p2p) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Engineering, Written | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    
    ```
    



#### BuiltBenchClusteringS2S

Clustering of built asset names/titles based on categories identified within industry classification systems such as IFC, Uniclass, etc.

**Dataset:** [`mehrzad-shahin/BuiltBench-clustering-s2s`](https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-clustering-s2s) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Engineering, Written | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    
    ```
    



#### CLSClusteringP2P

Clustering of titles + abstract from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringP2P`](https://huggingface.co/datasets/C-MTEB/CLSClusteringP2P) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2022csl,
      author = {Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
      journal = {arXiv preprint arXiv:2209.05034},
      title = {CSL: A large-scale Chinese scientific literature dataset},
      year = {2022},
    }
    
    ```
    



#### CLSClusteringP2P.v2

Clustering of titles + abstract from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`mteb/CLSClusteringP2P.v2`](https://huggingface.co/datasets/mteb/CLSClusteringP2P.v2) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2022csl,
      archiveprefix = {arXiv},
      author = {Yudong Li and Yuqing Zhang and Zhe Zhao and Linlin Shen and Weijie Liu and Weiquan Mao and Hui Zhang},
      eprint = {2209.05034},
      primaryclass = {cs.CL},
      title = {CSL: A Large-scale Chinese Scientific Literature Dataset},
      year = {2022},
    }
    
    ```
    



#### CLSClusteringS2S

Clustering of titles from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringS2S`](https://huggingface.co/datasets/C-MTEB/CLSClusteringS2S) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2022csl,
      author = {Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
      journal = {arXiv preprint arXiv:2209.05034},
      title = {CSL: A large-scale Chinese scientific literature dataset},
      year = {2022},
    }
    
    ```
    



#### CLSClusteringS2S.v2

Clustering of titles from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringS2S`](https://huggingface.co/datasets/C-MTEB/CLSClusteringS2S) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2022csl,
      archiveprefix = {arXiv},
      author = {Yudong Li and Yuqing Zhang and Zhe Zhao and Linlin Shen and Weijie Liu and Weiquan Mao and Hui Zhang},
      eprint = {2209.05034},
      primaryclass = {cs.CL},
      title = {CSL: A Large-scale Chinese Scientific Literature Dataset},
      year = {2022},
    }
    
    ```
    



#### ClusTREC-Covid

A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.

**Dataset:** [`Uri-ka/ClusTREC-Covid`](https://huggingface.co/datasets/Uri-ka/ClusTREC-Covid) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Medical, Written | expert-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{katz-etal-2024-knowledge,
      address = {Miami, Florida, USA},
      author = {Katz, Uri  and
    Levy, Mosh  and
    Goldberg, Yoav},
      booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
      month = nov,
      pages = {8838--8855},
      publisher = {Association for Computational Linguistics},
      title = {Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature},
      url = {https://aclanthology.org/2024.findings-emnlp.516},
      year = {2024},
    }
    
    ```
    



#### DigikalamagClustering

A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.

**Dataset:** [`mteb/DigikalamagClustering`](https://huggingface.co/datasets/mteb/DigikalamagClustering) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/tc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### DutchNewsArticlesClusteringP2P

This dataset contains all the articles published by the NOS as of the 1st of January 2010. The data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news organizations in the Netherlands.

**Dataset:** [`clips/mteb-nl-news-articles-cls`](https://huggingface.co/datasets/clips/mteb-nl-news-articles-cls) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | News, Written | derived | found |



#### DutchNewsArticlesClusteringS2S

This dataset contains all the articles published by the NOS as of the 1st of January 2010. The data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news organizations in the Netherlands.

**Dataset:** [`clips/mteb-nl-news-articles-cls`](https://huggingface.co/datasets/clips/mteb-nl-news-articles-cls) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | News, Written | derived | found |



#### EightTagsClustering

Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, food, medicine, motorization, work, sport and technology.

**Dataset:** [`PL-MTEB/8tags-clustering`](https://huggingface.co/datasets/PL-MTEB/8tags-clustering) • **License:** gpl-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Social, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{dadas-etal-2020-evaluation,
      address = {Marseille, France},
      author = {Dadas, Slawomir  and
    Pere{\\l}kiewicz, Micha{\\l}  and
    Po{\\'s}wiata, Rafa{\\l}},
      booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
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
    Mazo, H{\\'e}l{\\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      pages = {1674--1680},
      publisher = {European Language Resources Association},
      title = {Evaluation of Sentence Representations in {P}olish},
      url = {https://aclanthology.org/2020.lrec-1.207},
      year = {2020},
    }
    
    ```
    



#### EightTagsClustering.v2

Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, food, medicine, motorization, work, sport and technology.

**Dataset:** [`PL-MTEB/8tags-clustering`](https://huggingface.co/datasets/PL-MTEB/8tags-clustering) • **License:** gpl-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Social, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{dadas-etal-2020-evaluation,
      address = {Marseille, France},
      author = {Dadas, Slawomir  and
    Pere{\\l}kiewicz, Micha{\\l}  and
    Po{\\'s}wiata, Rafa{\\l}},
      booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
      editor = {Calzolari, Nicoletta  and
    B{\\'e}chet, Fr{\\'e}d{\\'e}ric  and
    Blache, Philippe  and
    Choukri, Khalid  and
    Cieri, Christopher  and
    Declerck, Thierry  and
    Goggi, Sara  and
    Isahara, Hitoshi  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Mazo, H{\\'e}l{\\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      pages = {1674--1680},
      publisher = {European Language Resources Association},
      title = {Evaluation of Sentence Representations in {P}olish},
      url = {https://aclanthology.org/2020.lrec-1.207},
      year = {2020},
    }
    
    ```
    



#### GeoreviewClusteringP2P

Review clustering based on Yandex Georeview dataset

**Dataset:** [`ai-forever/georeview-clustering-p2p`](https://huggingface.co/datasets/ai-forever/georeview-clustering-p2p) • **License:** mit • [Learn more →](https://github.com/yandex/geo-reviews-dataset-2023)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | rus | Reviews, Written | derived | found |



#### HALClusteringS2S

Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)

**Dataset:** [`lyon-nlp/clustering-hal-s2s`](https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Academic, Written | human-annotated | found |



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
    



#### HALClusteringS2S.v2

Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)

**Dataset:** [`mteb/HALClusteringS2S.v2`](https://huggingface.co/datasets/mteb/HALClusteringS2S.v2) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Academic, Written | human-annotated | found |



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
    



#### HUMEArxivClusteringP2P

Human evaluation subset of Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/mteb-human-arxiv-clustering`](https://huggingface.co/datasets/mteb/mteb-human-arxiv-clustering) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | v_measure | eng | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{arxiv_org_submitters_2024,
      author = {arXiv.org submitters},
      doi = {10.34740/KAGGLE/DSV/7548853},
      publisher = {Kaggle},
      title = {arXiv Dataset},
      url = {https://www.kaggle.com/dsv/7548853},
      year = {2024},
    }
    
    ```
    



#### HUMERedditClusteringP2P

Human evaluation subset of Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.

**Dataset:** [`mteb/mteb-human-reddit-clustering`](https://huggingface.co/datasets/mteb/mteb-human-reddit-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | v_measure | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### HUMESIB200ClusteringS2S

Human evaluation subset of Clustering of news article headlines from SIB-200. Clustering of 10 sets, each with 8 categories and 10 texts per category.

**Dataset:** [`mteb/mteb-human-sib200-clustering`](https://huggingface.co/datasets/mteb/mteb-human-sib200-clustering) • **License:** cc-by-4.0 • [Learn more →](https://github.com/dadelani/sib-200)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | v_measure | ara, dan, eng, fra, rus | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{adelani-etal-2023-sib,
      address = {Toronto, Canada},
      author = {Adelani, David Ifeoluwa  and
    Hedderich, Michael A.  and
    Zhu, Dawei  and
    van den Berg, Esther  and
    Klakow, Dietrich},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/2023.acl-long.660},
      month = jul,
      pages = {11784--11801},
      publisher = {Association for Computational Linguistics},
      title = {{SIB}-200: A Large-Scale News Classification Dataset for Over 200 Languages},
      url = {https://aclanthology.org/2023.acl-long.660},
      year = {2023},
    }
    
    ```
    



#### HUMEWikiCitiesClustering

Human evaluation subset of Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.

**Dataset:** [`mteb/mteb-human-wikicities-clustering`](https://huggingface.co/datasets/mteb/mteb-human-wikicities-clustering) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/wikipedia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | v_measure | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @online{wikidump2024,
      author = {Wikimedia Foundation},
      title = {Wikimedia Downloads},
      url = {https://dumps.wikimedia.org},
    }
    
    ```
    



#### HamshahriClustring

These datasets have been extracted from the RSS feed of two Farsi news agency websites.

**Dataset:** [`community-datasets/farsi_news`](https://huggingface.co/datasets/community-datasets/farsi_news) • **License:** not specified • [Learn more →](https://github.com/mallahyari/Farsi-datasets)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | News | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### IconclassClusteringS2S

Iconclass is an iconographic thesaurus, which is widely used in the digital heritage domain to describe subjects depicted in artworks. The task is to classify the first layer of Iconclass

**Dataset:** [`clips/mteb-nl-iconclass-cls`](https://huggingface.co/datasets/clips/mteb-nl-iconclass-cls) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://dl.acm.org/doi/pdf/10.1145/3575865)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | Fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{banar2023transfer,
      author = {Banar, Nikolay and Daelemans, Walter and Kestemont, Mike},
      journal = {ACM Journal on Computing and Cultural Heritage},
      number = {2},
      pages = {1--16},
      publisher = {ACM New York, NY},
      title = {Transfer learning for the visual arts: The multi-modal retrieval of iconclass codes},
      volume = {16},
      year = {2023},
    }
    
    ```
    



#### IndicReviewsClusteringP2P

Clustering of reviews from IndicSentiment dataset. Clustering of 14 sets on the generic categories label.

**Dataset:** [`mteb/IndicReviewsClusteringP2P`](https://huggingface.co/datasets/mteb/IndicReviewsClusteringP2P) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | asm, ben, brx, guj, hin, ... (13) | Reviews, Written | human-annotated | machine-translated and verified |



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
    



#### KlueMrcDomainClustering

this dataset is a processed and redistributed version of the KLUE-MRC dataset. Domain: Game / Media / Automotive / Finance / Real Estate / Education

**Dataset:** [`mteb/KlueMrcDomainClustering`](https://huggingface.co/datasets/mteb/KlueMrcDomainClustering) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_context_domain)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | kor | News, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{park2021klue,
      archiveprefix = {arXiv},
      author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      eprint = {2105.09680},
      primaryclass = {cs.CL},
      title = {KLUE: Korean Language Understanding Evaluation},
      year = {2021},
    }
    
    ```
    



#### KlueYnatMrcCategoryClustering

this dataset is a processed and redistributed version of the KLUE-Ynat & KLUE-MRC  dataset. News_category: IT/Science, Sports, Media/Culture, Ecomomy/Finance, Real Estate

**Dataset:** [`mteb/KlueYnatMrcCategoryClustering`](https://huggingface.co/datasets/mteb/KlueYnatMrcCategoryClustering) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_ynat_title)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | v_measure | kor | News, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{park2021klue,
      archiveprefix = {arXiv},
      author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      eprint = {2105.09680},
      primaryclass = {cs.CL},
      title = {KLUE: Korean Language Understanding Evaluation},
      year = {2021},
    }
    
    ```
    



#### LivedoorNewsClustering

Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics).

**Dataset:** [`mteb/LivedoorNewsClustering`](https://huggingface.co/datasets/mteb/LivedoorNewsClustering) • **License:** cc-by-nd-2.1-jp • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | jpn | News, Written | derived | found |



#### LivedoorNewsClustering.v2

Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics). Version 2 updated on LivedoorNewsClustering by removing pairs where one of entries contain an empty sentences.

**Dataset:** [`mteb/LivedoorNewsClustering.v2`](https://huggingface.co/datasets/mteb/LivedoorNewsClustering.v2) • **License:** cc-by-nd-2.1-jp • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | jpn | News, Written | derived | found |



#### MLSUMClusteringP2P

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{scialom2020mlsum,
      author = {Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
      journal = {arXiv preprint arXiv:2004.14900},
      title = {MLSUM: The Multilingual Summarization Corpus},
      year = {2020},
    }
    
    ```
    



#### MLSUMClusteringP2P.v2

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{scialom2020mlsum,
      author = {Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
      journal = {arXiv preprint arXiv:2004.14900},
      title = {MLSUM: The Multilingual Summarization Corpus},
      year = {2020},
    }
    
    ```
    



#### MLSUMClusteringS2S

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{scialom2020mlsum,
      author = {Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
      journal = {arXiv preprint arXiv:2004.14900},
      title = {MLSUM: The Multilingual Summarization Corpus},
      year = {2020},
    }
    
    ```
    



#### MLSUMClusteringS2S.v2

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{scialom2020mlsum,
      author = {Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
      journal = {arXiv preprint arXiv:2004.14900},
      title = {MLSUM: The Multilingual Summarization Corpus},
      year = {2020},
    }
    
    ```
    



#### MasakhaNEWSClusteringP2P

Clustering of news article headlines and texts from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.

**Dataset:** [`mteb/MasakhaNEWSClusteringP2P`](https://huggingface.co/datasets/mteb/MasakhaNEWSClusteringP2P) • **License:** afl-3.0 • [Learn more →](https://huggingface.co/datasets/masakhane/masakhanews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | amh, eng, fra, hau, ibo, ... (16) | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{adelani2023masakhanews,
      author = {David Ifeoluwa Adelani and  Marek Masiak and  Israel Abebe Azime and  Jesujoba Oluwadara Alabi and  Atnafu Lambebo Tonja and  Christine Mwase and  Odunayo Ogundepo and  Bonaventure F. P. Dossou and  Akintunde Oladipo and  Doreen Nixdorf and  Chris Chinenye Emezue and  Sana Sabah al-azzawi and  Blessing K. Sibanda and  Davis David and  Lolwethu Ndolela and  Jonathan Mukiibi and  Tunde Oluwaseyi Ajayi and  Tatiana Moteu Ngoli and  Brian Odhiambo and  Abraham Toluwase Owodunni and  Nnaemeka C. Obiefuna and  Shamsuddeen Hassan Muhammad and  Saheed Salahudeen Abdullahi and  Mesay Gemeda Yigezu and  Tajuddeen Gwadabe and  Idris Abdulmumin and  Mahlet Taye Bame and  Oluwabusayo Olufunke Awoyomi and  Iyanuoluwa Shode and  Tolulope Anu Adelani and  Habiba Abdulganiy Kailani and  Abdul-Hakeem Omotayo and  Adetola Adeeko and  Afolabi Abeeb and  Anuoluwapo Aremu and  Olanrewaju Samuel and  Clemencia Siro and  Wangari Kimotho and  Onyekachi Raphael Ogbu and  Chinedu E. Mbonu and  Chiamaka I. Chukwuneke and  Samuel Fanijo and  Jessica Ojo and  Oyinkansola F. Awosan and  Tadesse Kebede Guge and  Sakayo Toadoum Sari and  Pamela Nyatsine and  Freedmore Sidume and  Oreen Yousuf and  Mardiyyah Oduwole and  Ussen Kimanuka and  Kanda Patrick Tshinu and  Thina Diko and  Siyanda Nxakama and   Abdulmejid Tuni Johar and  Sinodos Gebre and  Muhidin Mohamed and  Shafie Abdi Mohamed and  Fuad Mire Hassan and  Moges Ahmed Mehamed and  Evrard Ngabire and  and Pontus Stenetorp},
      journal = {ArXiv},
      title = {MasakhaNEWS: News Topic Classification for African languages},
      volume = {},
      year = {2023},
    }
    
    ```
    



#### MasakhaNEWSClusteringS2S

Clustering of news article headlines from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.

**Dataset:** [`mteb/MasakhaNEWSClusteringS2S`](https://huggingface.co/datasets/mteb/MasakhaNEWSClusteringS2S) • **License:** afl-3.0 • [Learn more →](https://huggingface.co/datasets/masakhane/masakhanews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | amh, eng, fra, hau, ibo, ... (16) | News, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{adelani2023masakhanews,
      author = {David Ifeoluwa Adelani and  Marek Masiak and  Israel Abebe Azime and  Jesujoba Oluwadara Alabi and  Atnafu Lambebo Tonja and  Christine Mwase and  Odunayo Ogundepo and  Bonaventure F. P. Dossou and  Akintunde Oladipo and  Doreen Nixdorf and  Chris Chinenye Emezue and  Sana Sabah al-azzawi and  Blessing K. Sibanda and  Davis David and  Lolwethu Ndolela and  Jonathan Mukiibi and  Tunde Oluwaseyi Ajayi and  Tatiana Moteu Ngoli and  Brian Odhiambo and  Abraham Toluwase Owodunni and  Nnaemeka C. Obiefuna and  Shamsuddeen Hassan Muhammad and  Saheed Salahudeen Abdullahi and  Mesay Gemeda Yigezu and  Tajuddeen Gwadabe and  Idris Abdulmumin and  Mahlet Taye Bame and  Oluwabusayo Olufunke Awoyomi and  Iyanuoluwa Shode and  Tolulope Anu Adelani and  Habiba Abdulganiy Kailani and  Abdul-Hakeem Omotayo and  Adetola Adeeko and  Afolabi Abeeb and  Anuoluwapo Aremu and  Olanrewaju Samuel and  Clemencia Siro and  Wangari Kimotho and  Onyekachi Raphael Ogbu and  Chinedu E. Mbonu and  Chiamaka I. Chukwuneke and  Samuel Fanijo and  Jessica Ojo and  Oyinkansola F. Awosan and  Tadesse Kebede Guge and  Sakayo Toadoum Sari and  Pamela Nyatsine and  Freedmore Sidume and  Oreen Yousuf and  Mardiyyah Oduwole and  Ussen Kimanuka and  Kanda Patrick Tshinu and  Thina Diko and  Siyanda Nxakama and   Abdulmejid Tuni Johar and  Sinodos Gebre and  Muhidin Mohamed and  Shafie Abdi Mohamed and  Fuad Mire Hassan and  Moges Ahmed Mehamed and  Evrard Ngabire and  and Pontus Stenetorp},
      journal = {ArXiv},
      title = {MasakhaNEWS: News Topic Classification for African languages},
      volume = {},
      year = {2023},
    }
    
    ```
    



#### MedrxivClusteringP2P

Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.

**Dataset:** [`mteb/medrxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/medrxiv-clustering-p2p) • **License:** https://www.medrxiv.org/content/about-medrxiv • [Learn more →](https://api.medrxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | created |



#### MedrxivClusteringP2P.v2

Clustering of titles+abstract from medrxiv across 51 categories.

**Dataset:** [`mteb/medrxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/medrxiv-clustering-p2p) • **License:** https://www.medrxiv.org/content/about-medrxiv • [Learn more →](https://api.medrxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Medical, Written | derived | created |



#### MedrxivClusteringS2S

Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.

**Dataset:** [`mteb/medrxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/medrxiv-clustering-s2s) • **License:** https://www.medrxiv.org/content/about-medrxiv • [Learn more →](https://api.medrxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Medical, Written | derived | created |



#### MedrxivClusteringS2S.v2

Clustering of titles from medrxiv across 51 categories.

**Dataset:** [`mteb/medrxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/medrxiv-clustering-s2s) • **License:** https://www.medrxiv.org/content/about-medrxiv • [Learn more →](https://api.medrxiv.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Medical, Written | derived | created |



#### MewsC16JaClustering

MewsC-16 (Multilingual Short Text Clustering Dataset for News in 16 languages) is constructed from Wikinews. This dataset is the Japanese split of MewsC-16, containing topic sentences from Wikinews articles in 12 categories. More detailed information is available in the Appendix E of the citation.

**Dataset:** [`mteb/MewsC16JaClustering`](https://huggingface.co/datasets/mteb/MewsC16JaClustering) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | jpn | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{nishikawa-etal-2022-ease,
      address = {Seattle, United States},
      author = {Nishikawa, Sosuke  and
    Ri, Ryokan  and
    Yamada, Ikuya  and
    Tsuruoka, Yoshimasa  and
    Echizen, Isao},
      booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      month = jul,
      pages = {3870--3885},
      publisher = {Association for Computational Linguistics},
      title = {{EASE}: Entity-Aware Contrastive Learning of Sentence Embedding},
      url = {https://aclanthology.org/2022.naacl-main.284},
      year = {2022},
    }
    
    ```
    



#### NLPTwitterAnalysisClustering

Clustering of tweets from twitter across 26 categories.

**Dataset:** [`hamedhf/nlp_twitter_analysis`](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/commits/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | Social | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### OpenTenderClusteringP2P

This dataset contains all the articles published by the NOS as of the 1st of January 2010. The data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news organizations in the Netherlands.

**Dataset:** [`clips/mteb-nl-opentender-cls-pr`](https://huggingface.co/datasets/clips/mteb-nl-opentender-cls-pr) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2509.12340)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | Government, Written | derived | found |



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
    



#### OpenTenderClusteringS2S

This dataset contains all the articles published by the NOS as of the 1st of January 2010. The data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news organizations in the Netherlands.

**Dataset:** [`clips/mteb-nl-opentender-clst-s2s-pr`](https://huggingface.co/datasets/clips/mteb-nl-opentender-clst-s2s-pr) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2509.12340)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | Government, Written | derived | found |



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
    



#### PlscClusteringP2P

Clustering of Polish article titles+abstracts from Library of Science (https://bibliotekanauki.pl/), either on the scientific field or discipline.

**Dataset:** [`PL-MTEB/plsc-clustering-p2p`](https://huggingface.co/datasets/PL-MTEB/plsc-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/rafalposwiata/plsc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Academic, Written | derived | found |



#### PlscClusteringP2P.v2

Clustering of Polish article titles+abstracts from Library of Science (https://bibliotekanauki.pl/), either on the scientific field or discipline.

**Dataset:** [`mteb/PlscClusteringP2P.v2`](https://huggingface.co/datasets/mteb/PlscClusteringP2P.v2) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/rafalposwiata/plsc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Academic, Written | derived | found |



#### PlscClusteringS2S

Clustering of Polish article titles from Library of Science (https://bibliotekanauki.pl/), either on the scientific field or discipline.

**Dataset:** [`PL-MTEB/plsc-clustering-s2s`](https://huggingface.co/datasets/PL-MTEB/plsc-clustering-s2s) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/rafalposwiata/plsc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Academic, Written | derived | found |



#### PlscClusteringS2S.v2

Clustering of Polish article titles from Library of Science (https://bibliotekanauki.pl/), either on the scientific field or discipline.

**Dataset:** [`PL-MTEB/plsc-clustering-s2s`](https://huggingface.co/datasets/PL-MTEB/plsc-clustering-s2s) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/rafalposwiata/plsc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Academic, Written | derived | found |



#### RedditClustering

Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/reddit-clustering`](https://huggingface.co/datasets/mteb/reddit-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### RedditClustering-VN

A translated dataset from Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/reddit-clustering-vn`](https://huggingface.co/datasets/GreenNode/reddit-clustering-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Social, Web, Written | derived | machine-translated and LM verified |



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
    



#### RedditClustering.v2

Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/reddit-clustering`](https://huggingface.co/datasets/mteb/reddit-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### RedditClusteringP2P

Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.

**Dataset:** [`mteb/reddit-clustering-p2p`](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### RedditClusteringP2P-VN

A translated dataset from Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/reddit-clustering-p2p-vn`](https://huggingface.co/datasets/GreenNode/reddit-clustering-p2p-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Social, Web, Written | derived | machine-translated and LM verified |



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
    



#### RedditClusteringP2P.v2

Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.

**Dataset:** [`mteb/reddit-clustering-p2p`](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### RomaniBibleClustering

Clustering verses from the Bible in Kalderash Romani by book.

**Dataset:** [`mteb/RomaniBibleClustering`](https://huggingface.co/datasets/mteb/RomaniBibleClustering) • **License:** mit • [Learn more →](https://romani.global.bible/info)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | rom | Religious, Written | derived | human-translated and localized |



#### RuSciBenchGRNTIClusteringP2P

Clustering of scientific papers (title+abstract) by rubric

**Dataset:** [`ai-forever/ru-scibench-grnti-classification`](https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | rus | Academic, Written | derived | found |



#### RuSciBenchOECDClusteringP2P

Clustering of scientific papers (title+abstract) by rubric

**Dataset:** [`ai-forever/ru-scibench-oecd-classification`](https://huggingface.co/datasets/ai-forever/ru-scibench-oecd-classification) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | rus | Academic, Written | derived | found |



#### SIB200ClusteringS2S

SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects annotated. The dataset is annotated in English for the topics, science/technology, travel, politics, sports, health, entertainment, and geography. The labels are then transferred to the other languages in Flores-200 which are human-translated.

**Dataset:** [`mteb/sib200`](https://huggingface.co/datasets/mteb/sib200) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2309.07445)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | ace, acm, acq, aeb, afr, ... (197) | News, Written | expert-annotated | human-translated and localized |



??? quote "Citation"

    
    ```bibtex
    
    @article{adelani2023sib,
      author = {Adelani, David Ifeoluwa and Liu, Hannah and Shen, Xiaoyu and Vassilyev, Nikita and Alabi, Jesujoba O and Mao, Yanke and Gao, Haonan and Lee, Annie En-Shiun},
      journal = {arXiv preprint arXiv:2309.07445},
      title = {SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects},
      year = {2023},
    }
    
    ```
    



#### SIDClustring

Clustering of summariesfrom SIDClustring across categories.

**Dataset:** [`MCINext/sid-clustering`](https://huggingface.co/datasets/MCINext/sid-clustering) • **License:** not specified • [Learn more →](https://www.sid.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | Academic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### SNLClustering

Webscraped articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.

**Dataset:** [`adrlau/navjordj-SNL_summarization_copy`](https://huggingface.co/datasets/adrlau/navjordj-SNL_summarization_copy) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLRetrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | Encyclopaedic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### SNLHierarchicalClusteringP2P

Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.

**Dataset:** [`mteb/SNLHierarchicalClusteringP2P`](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringP2P) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringP2P)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | Encyclopaedic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### SNLHierarchicalClusteringS2S

Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.

**Dataset:** [`mteb/SNLHierarchicalClusteringS2S`](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringS2S) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringS2S)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | Encyclopaedic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### SpanishNewsClusteringP2P

Clustering of news articles, 7 topics in total.

**Dataset:** [`jinaai/spanish_news_clustering`](https://huggingface.co/datasets/jinaai/spanish_news_clustering) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | spa | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{kevinmorgado2019spanish,
      author = {Kevin Morgado},
      howpublished = {Kaggle},
      title = {Spanish News Classification},
      url = {https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification},
      year = {2019},
    }
    
    ```
    



#### StackExchangeClustering

Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/stackexchange-clustering`](https://huggingface.co/datasets/mteb/stackexchange-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### StackExchangeClustering-VN

A translated dataset from Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stackexchange-clustering-vn`](https://huggingface.co/datasets/GreenNode/stackexchange-clustering-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Web, Written | derived | machine-translated and LM verified |



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
    



#### StackExchangeClustering.v2

Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/stackexchange-clustering`](https://huggingface.co/datasets/mteb/stackexchange-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### StackExchangeClusteringP2P

Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.

**Dataset:** [`mteb/stackexchange-clustering-p2p`](https://huggingface.co/datasets/mteb/stackexchange-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### StackExchangeClusteringP2P-VN

A translated Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stackexchange-clustering-p2p-vn`](https://huggingface.co/datasets/GreenNode/stackexchange-clustering-p2p-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Web, Written | derived | machine-translated and LM verified |



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
    



#### StackExchangeClusteringP2P.v2

Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.

**Dataset:** [`mteb/stackexchange-clustering-p2p`](https://huggingface.co/datasets/mteb/stackexchange-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{geigle:2021:arxiv,
      archiveprefix = {arXiv},
      author = {Gregor Geigle and
    Nils Reimers and
    Andreas R{\"u}ckl{\'e} and
    Iryna Gurevych},
      eprint = {2104.07081},
      journal = {arXiv preprint},
      title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
      url = {http://arxiv.org/abs/2104.07081},
      volume = {abs/2104.07081},
      year = {2021},
    }
    
    ```
    



#### SwednClustering

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.

**Dataset:** [`mteb/SwednClustering`](https://huggingface.co/datasets/mteb/SwednClustering) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | swe | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    
    ```
    



#### SwednClusteringP2P

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.

**Dataset:** [`mteb/SwednClusteringP2P`](https://huggingface.co/datasets/mteb/SwednClusteringP2P) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | swe | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    
    ```
    



#### SwednClusteringS2S

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.

**Dataset:** [`mteb/SwednClusteringS2S`](https://huggingface.co/datasets/mteb/SwednClusteringS2S) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | swe | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    
    ```
    



#### TenKGnadClusteringP2P

Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Web, Written | derived | found |



#### TenKGnadClusteringP2P.v2

Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category. v2 uses a faster evaluation method used in the MMTEB paper, which allow for notably faster evaluation.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p) • **License:** cc-by-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | News, Non-fiction, Written | derived | found |



#### TenKGnadClusteringS2S

Clustering of news article titles. Clustering of 10 splits on the news article category.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | News, Non-fiction, Written | derived | found |



#### TenKGnadClusteringS2S.v2

Clustering of news article titles. Clustering of 10 splits on the news article category. v2 uses a faster evaluation method used in the MMTEB paper, which allow for notably faster evaluation.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s) • **License:** cc-by-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | News, Non-fiction, Written | derived | found |



#### ThuNewsClusteringP2P

Clustering of titles + abstracts from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringP2P`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringP2P) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{eisner2007proceedings,
      author = {Eisner, Jason},
      booktitle = {Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)},
      title = {Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL)},
      year = {2007},
    }
    
    @inproceedings{li2006comparison,
      author = {Li, Jingyang and Sun, Maosong and Zhang, Xian},
      booktitle = {proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics},
      pages = {545--552},
      title = {A comparison and semi-quantitative analysis of words and character-bigrams as features in chinese text categorization},
      year = {2006},
    }
    
    ```
    



#### ThuNewsClusteringP2P.v2

Clustering of titles + abstracts from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringP2P`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringP2P) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @software{sun2016thuctc,
      author = {Sun, M. and Li, J. and Guo, Z. and Yu, Z. and Zheng, Y. and Si, X. and Liu, Z.},
      note = {THU Chinese Text Classification Toolkit},
      publisher = {THU Natural Language Processing Lab},
      title = {THUCTC: An Efficient Chinese Text Classifier},
      url = {https://github.com/thunlp/THUCTC},
      year = {2016},
    }
    
    ```
    



#### ThuNewsClusteringS2S

Clustering of titles from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringS2S`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringS2S) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{eisner2007proceedings,
      author = {Eisner, Jason},
      booktitle = {Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)},
      title = {Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL)},
      year = {2007},
    }
    
    @inproceedings{li2006comparison,
      author = {Li, Jingyang and Sun, Maosong and Zhang, Xian},
      booktitle = {proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics},
      pages = {545--552},
      title = {A comparison and semi-quantitative analysis of words and character-bigrams as features in chinese text categorization},
      year = {2006},
    }
    
    ```
    



#### ThuNewsClusteringS2S.v2

Clustering of titles from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringS2S`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringS2S) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @software{sun2016thuctc,
      author = {Sun, M. and Li, J. and Guo, Z. and Yu, Z. and Zheng, Y. and Si, X. and Liu, Z.},
      note = {THU Chinese Text Classification Toolkit},
      publisher = {THU Natural Language Processing Lab},
      title = {THUCTC: An Efficient Chinese Text Classifier},
      url = {https://github.com/thunlp/THUCTC},
      year = {2016},
    }
    
    ```
    



#### TwentyNewsgroupsClustering

Clustering of the 20 Newsgroups dataset (subject only).

**Dataset:** [`mteb/twentynewsgroups-clustering`](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) • **License:** not specified • [Learn more →](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @incollection{LANG1995331,
      address = {San Francisco (CA)},
      author = {Ken Lang},
      booktitle = {Machine Learning Proceedings 1995},
      doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
      editor = {Armand Prieditis and Stuart Russell},
      isbn = {978-1-55860-377-6},
      pages = {331-339},
      publisher = {Morgan Kaufmann},
      title = {NewsWeeder: Learning to Filter Netnews},
      url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
      year = {1995},
    }
    
    ```
    



#### TwentyNewsgroupsClustering-VN

A translated dataset from Clustering of the 20 Newsgroups dataset (subject only). The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/twentynewsgroups-clustering-vn`](https://huggingface.co/datasets/GreenNode/twentynewsgroups-clustering-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | News, Written | derived | machine-translated and LM verified |



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
    



#### TwentyNewsgroupsClustering.v2

Clustering of the 20 Newsgroups dataset (subject only).

**Dataset:** [`mteb/twentynewsgroups-clustering`](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) • **License:** not specified • [Learn more →](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @incollection{LANG1995331,
      address = {San Francisco (CA)},
      author = {Ken Lang},
      booktitle = {Machine Learning Proceedings 1995},
      doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
      editor = {Armand Prieditis and Stuart Russell},
      isbn = {978-1-55860-377-6},
      pages = {331-339},
      publisher = {Morgan Kaufmann},
      title = {NewsWeeder: Learning to Filter Netnews},
      url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
      year = {1995},
    }
    
    ```
    



#### VABBClusteringP2P

This dataset contains the fourteenth edition of the Flemish Academic Bibliography for the Social Sciences and Humanities (VABB-SHW), a database of academic publications from the social sciences and humanities authored by researchers affiliated to Flemish universities (more information). Publications in the database are used as one of the parameters of the Flemish performance-based research funding system

**Dataset:** [`clips/mteb-nl-vabb-cls`](https://huggingface.co/datasets/clips/mteb-nl-vabb-cls) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://zenodo.org/records/14214806)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | Academic, Written | derived | found |



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
    



#### VABBClusteringS2S

This dataset contains the fourteenth edition of the Flemish Academic Bibliography for the Social Sciences and Humanities (VABB-SHW), a database of academic publications from the social sciences and humanities authored by researchers affiliated to Flemish universities (more information). Publications in the database are used as one of the parameters of the Flemish performance-based research funding system

**Dataset:** [`clips/mteb-nl-vabb-cls`](https://huggingface.co/datasets/clips/mteb-nl-vabb-cls) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://zenodo.org/records/14214806)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nld | Academic, Written | derived | found |



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
    



#### VGClustering

Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.

**Dataset:** [`navjordj/VG_summarization`](https://huggingface.co/datasets/navjordj/VG_summarization) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/navjordj/VG_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### VGHierarchicalClusteringP2P

Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.

**Dataset:** [`navjordj/VG_summarization`](https://huggingface.co/datasets/navjordj/VG_summarization) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/navjordj/VG_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### VGHierarchicalClusteringS2S

Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.

**Dataset:** [`navjordj/VG_summarization`](https://huggingface.co/datasets/navjordj/VG_summarization) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/navjordj/VG_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### WikiCitiesClustering

Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.

**Dataset:** [`mteb/WikiCitiesClustering`](https://huggingface.co/datasets/mteb/WikiCitiesClustering) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/wikipedia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @online{wikidump2024,
      author = {Wikimedia Foundation},
      title = {Wikimedia Downloads},
      url = {https://dumps.wikimedia.org},
    }
    
    ```
    



#### WikiClusteringP2P

Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).

**Dataset:** [`ryzzlestrizzle/multi-wiki-clustering-p2p`](https://huggingface.co/datasets/ryzzlestrizzle/multi-wiki-clustering-p2p) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/Rysias/wiki-clustering)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | bos, cat, ces, dan, eus, ... (14) | Encyclopaedic, Written | derived | created |



#### WikiClusteringP2P.v2

Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).

**Dataset:** [`mteb/WikiClusteringP2P.v2`](https://huggingface.co/datasets/mteb/WikiClusteringP2P.v2) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/Rysias/wiki-clustering)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | bos, cat, ces, dan, eus, ... (14) | Encyclopaedic, Written | derived | created |



#### WikipediaChemistryTopicsClustering

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy10Clustering`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Clustering) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Chemistry | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    
    ```
    



#### WikipediaSpecialtiesInChemistryClustering

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium5Clustering`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Clustering) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Chemistry | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    
    ```



## ImageClustering

- **Number of tasks:** 5

#### CIFAR100Clustering

Clustering images from 100 classes.

**Dataset:** [`mteb/cifar100`](https://huggingface.co/datasets/mteb/cifar100) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar100)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | nmi | eng | Web | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @techreport{Krizhevsky09learningmultiple,
      author = {Alex Krizhevsky},
      institution = {},
      title = {Learning multiple layers of features from tiny images},
      year = {2009},
    }
    
    ```
    



#### CIFAR10Clustering

Clustering images from 10 classes.

**Dataset:** [`mteb/cifar10`](https://huggingface.co/datasets/mteb/cifar10) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | nmi | eng | Web | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @techreport{Krizhevsky09learningmultiple,
      author = {Alex Krizhevsky},
      institution = {},
      title = {Learning multiple layers of features from tiny images},
      year = {2009},
    }
    
    ```
    



#### ImageNet10Clustering

Clustering images from an 10-class subset of ImageNet which are generally easy to distinguish.

**Dataset:** [`mteb/imagenet-10`](https://huggingface.co/datasets/mteb/imagenet-10) • **License:** not specified • [Learn more →](https://www.kaggle.com/datasets/liusha249/imagenet10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | nmi | eng | Web | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{deng2009imagenet,
      author = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
      booktitle = {2009 IEEE Conference on Computer Vision and Pattern Recognition},
      doi = {10.1109/CVPR.2009.5206848},
      keywords = {Large-scale systems;Image databases;Explosions;Internet;Robustness;Information retrieval;Image retrieval;Multimedia databases;Ontologies;Spine},
      number = {},
      pages = {248-255},
      title = {ImageNet: A large-scale hierarchical image database},
      volume = {},
      year = {2009},
    }
    
    ```
    



#### ImageNetDog15Clustering

Clustering images from a 15-class dogs-only subset of the dog classes in ImageNet.

**Dataset:** [`mteb/imagenet-dog-15`](https://huggingface.co/datasets/mteb/imagenet-dog-15) • **License:** not specified • [Learn more →](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | nmi | eng | Web | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{deng2009imagenet,
      author = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
      booktitle = {2009 IEEE Conference on Computer Vision and Pattern Recognition},
      doi = {10.1109/CVPR.2009.5206848},
      keywords = {Large-scale systems;Image databases;Explosions;Internet;Robustness;Information retrieval;Image retrieval;Multimedia databases;Ontologies;Spine},
      number = {},
      pages = {248-255},
      title = {ImageNet: A large-scale hierarchical image database},
      volume = {},
      year = {2009},
    }
    
    ```
    



#### TinyImageNetClustering

Clustering over 200 classes.

**Dataset:** [`mteb/tiny-imagenet`](https://huggingface.co/datasets/mteb/tiny-imagenet) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/zh-plus/tiny-imagenet/viewer/default/valid)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | nmi | eng | Reviews | derived | found |

<!-- END-TASKS -->