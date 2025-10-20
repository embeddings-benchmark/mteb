
# Any2AnyRetrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 49

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
    



#### CUB200I2IRetrieval

Retrieve bird images from 200 classes.

**Dataset:** [`isaacchung/cub200_retrieval`](https://huggingface.co/datasets/isaacchung/cub200_retrieval) • **License:** not specified • [Learn more →](https://www.florian-schroff.de/publications/CUB-200.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{article,
      author = {Welinder, Peter and Branson, Steve and Mita, Takeshi and Wah, Catherine and Schroff, Florian and Belongie, Serge and Perona, Pietro},
      month = {09},
      pages = {},
      title = {Caltech-UCSD Birds 200},
      year = {2010},
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
| image, text to image (it2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



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
