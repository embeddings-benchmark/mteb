# Available Tasks

This section contains an overview of all available tasks in MTEB.

<!-- The following section is auto-generated. Changes will be overwritten. Please change the source dataset. -->
<!-- START TASK DESCRIPTION -->
## Any2AnyMultilingualRetrieval

- **Number of tasks of the given type:** 3 

#### WITT2IRetrieval

Retrieve images based on multilingual descriptions.

**Dataset:** [`mteb/wit`](https://huggingface.co/datasets/mteb/wit) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | ara, bul, dan, ell, eng, ... (11) | Encyclopaedic, Written | derived | found |



#### XFlickr30kCoT2IRetrieval

Retrieve images based on multilingual descriptions.

**Dataset:** [`floschne/xflickrco`](https://huggingface.co/datasets/floschne/xflickrco) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.mlr.press/v162/bugliarello22a/bugliarello22a.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, ind, jpn, rus, ... (8) | Encyclopaedic, Written | derived | found |



#### XM3600T2IRetrieval

Retrieve images based on multilingual descriptions.

**Dataset:** [`floschne/xm3600`](https://huggingface.co/datasets/floschne/xm3600) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.emnlp-main.45/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | ara, ben, ces, dan, deu, ... (36) | Encyclopaedic, Written | derived | found |



## Any2AnyRetrieval

- **Number of tasks of the given type:** 49 

#### BLINKIT2IRetrieval

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`JamieSJS/blink-it2i`](https://huggingface.co/datasets/JamieSJS/blink-it2i) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.12390)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | found |



#### BLINKIT2TRetrieval

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`JamieSJS/blink-it2t`](https://huggingface.co/datasets/JamieSJS/blink-it2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.12390)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_1 | eng | Encyclopaedic | derived | found |



#### CIRRIT2IRetrieval

Retrieve images based on texts and images.

**Dataset:** [`MRBench/mbeir_cirr_task7`](https://huggingface.co/datasets/MRBench/mbeir_cirr_task7) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### CUB200I2IRetrieval

Retrieve bird images from 200 classes.

**Dataset:** [`isaacchung/cub200_retrieval`](https://huggingface.co/datasets/isaacchung/cub200_retrieval) • **License:** not specified • [Learn more →](https://www.florian-schroff.de/publications/CUB-200.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



#### EDIST2ITRetrieval

Retrieve news images and titles based on news content.

**Dataset:** [`MRBench/mbeir_edis_task2`](https://huggingface.co/datasets/MRBench/mbeir_edis_task2) • **License:** apache-2.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.297/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image, text (t2it) | ndcg_at_10 | eng | News | derived | created |



#### EncyclopediaVQAIT2ITRetrieval

Retrieval Wiki passage and image and passage to answer query about an image.

**Dataset:** [`izhx/UMRB-EncyclopediaVQA`](https://huggingface.co/datasets/izhx/UMRB-EncyclopediaVQA) • **License:** cc-by-4.0 • [Learn more →](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | cv_recall_at_5 | eng | Encyclopaedic | derived | created |



#### FORBI2IRetrieval

Retrieve flat object images from 8 classes.

**Dataset:** [`isaacchung/forb_retrieval`](https://huggingface.co/datasets/isaacchung/forb_retrieval) • **License:** not specified • [Learn more →](https://github.com/pxiangwu/FORB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



#### Fashion200kI2TRetrieval

Retrieve clothes based on descriptions.

**Dataset:** [`MRBench/mbeir_fashion200k_task3`](https://huggingface.co/datasets/MRBench/mbeir_fashion200k_task3) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### Fashion200kT2IRetrieval

Retrieve clothes based on descriptions.

**Dataset:** [`MRBench/mbeir_fashion200k_task0`](https://huggingface.co/datasets/MRBench/mbeir_fashion200k_task0) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### FashionIQIT2IRetrieval

Retrieve clothes based on descriptions.

**Dataset:** [`MRBench/mbeir_fashioniq_task7`](https://huggingface.co/datasets/MRBench/mbeir_fashioniq_task7) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### Flickr30kI2TRetrieval

Retrieve captions based on images.

**Dataset:** [`isaacchung/flickr30ki2t`](https://huggingface.co/datasets/isaacchung/flickr30ki2t) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Web, Written | derived | found |



#### Flickr30kT2IRetrieval

Retrieve images based on captions.

**Dataset:** [`isaacchung/flickr30kt2i`](https://huggingface.co/datasets/isaacchung/flickr30kt2i) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Web, Written | derived | found |



#### GLDv2I2IRetrieval

Retrieve names of landmarks based on their image.

**Dataset:** [`gowitheflow/gld-v2`](https://huggingface.co/datasets/gowitheflow/gld-v2) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### GLDv2I2TRetrieval

Retrieve names of landmarks based on their image.

**Dataset:** [`JamieSJS/gld-v2-i2t`](https://huggingface.co/datasets/JamieSJS/gld-v2-i2t) • **License:** apache-2.0 • [Learn more →](https://openaccess.thecvf.com/content_CVPR_2020/html/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### HatefulMemesI2TRetrieval

Retrieve captions based on memes to assess OCR abilities.

**Dataset:** [`Ahren09/MMSoc_HatefulMemes`](https://huggingface.co/datasets/Ahren09/MMSoc_HatefulMemes) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2005.04790)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### HatefulMemesT2IRetrieval

Retrieve captions based on memes to assess OCR abilities.

**Dataset:** [`Ahren09/MMSoc_HatefulMemes`](https://huggingface.co/datasets/Ahren09/MMSoc_HatefulMemes) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2005.04790)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### ImageCoDeT2IRetrieval

Retrieve a specific video frame based on a precise caption.

**Dataset:** [`JamieSJS/imagecode`](https://huggingface.co/datasets/JamieSJS/imagecode) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.acl-long.241.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | cv_recall_at_3 | eng | Web, Written | derived | found |



#### InfoSeekIT2ITRetrieval

Retrieve source text and image information to answer questions about images.

**Dataset:** [`MRBench/mbeir_infoseek_task8`](https://huggingface.co/datasets/MRBench/mbeir_infoseek_task8) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.925)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### InfoSeekIT2TRetrieval

Retrieve source information to answer questions about images.

**Dataset:** [`MRBench/mbeir_infoseek_task6`](https://huggingface.co/datasets/MRBench/mbeir_infoseek_task6) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.emnlp-main.925)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### LLaVAIT2TRetrieval

Retrieve responses to answer questions about images.

**Dataset:** [`izhx/UMRB-LLaVA`](https://huggingface.co/datasets/izhx/UMRB-LLaVA) • **License:** cc-by-4.0 • [Learn more →](https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_5 | eng | Encyclopaedic | derived | found |



#### METI2IRetrieval

Retrieve photos of more than 224k artworks.

**Dataset:** [`JamieSJS/met`](https://huggingface.co/datasets/JamieSJS/met) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2202.01747)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



#### MSCOCOI2TRetrieval

Retrieve captions based on images.

**Dataset:** [`MRBench/mbeir_mscoco_task3`](https://huggingface.co/datasets/MRBench/mbeir_mscoco_task3) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### MSCOCOT2IRetrieval

Retrieve images based on captions.

**Dataset:** [`MRBench/mbeir_mscoco_task0`](https://huggingface.co/datasets/MRBench/mbeir_mscoco_task0) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### MemotionI2TRetrieval

Retrieve captions based on memes.

**Dataset:** [`Ahren09/MMSoc_Memotion`](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion) • **License:** mit • [Learn more →](https://aclanthology.org/2020.semeval-1.99/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### MemotionT2IRetrieval

Retrieve memes based on captions.

**Dataset:** [`Ahren09/MMSoc_Memotion`](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion) • **License:** mit • [Learn more →](https://aclanthology.org/2020.semeval-1.99/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### NIGHTSI2IRetrieval

Retrieval identical image to the given image.

**Dataset:** [`MRBench/mbeir_nights_task4`](https://huggingface.co/datasets/MRBench/mbeir_nights_task4) • **License:** cc-by-sa-4.0 • [Learn more →](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### OKVQAIT2TRetrieval

Retrieval a Wiki passage to answer query about an image.

**Dataset:** [`izhx/UMRB-OKVQA`](https://huggingface.co/datasets/izhx/UMRB-OKVQA) • **License:** cc-by-4.0 • [Learn more →](https://okvqa.allenai.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_10 | eng | Encyclopaedic | derived | created |



#### OVENIT2ITRetrieval

Retrieval a Wiki image and passage to answer query about an image.

**Dataset:** [`MRBench/mbeir_oven_task8`](https://huggingface.co/datasets/MRBench/mbeir_oven_task8) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### OVENIT2TRetrieval

Retrieval a Wiki passage to answer query about an image.

**Dataset:** [`MRBench/mbeir_oven_task6`](https://huggingface.co/datasets/MRBench/mbeir_oven_task6) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### ROxfordEasyI2IRetrieval

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`JamieSJS/r-oxford-easy-multi`](https://huggingface.co/datasets/JamieSJS/r-oxford-easy-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



#### ROxfordHardI2IRetrieval

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`JamieSJS/r-oxford-hard-multi`](https://huggingface.co/datasets/JamieSJS/r-oxford-hard-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



#### ROxfordMediumI2IRetrieval

Retrieve photos of landmarks in Oxford, UK.

**Dataset:** [`JamieSJS/r-oxford-medium-multi`](https://huggingface.co/datasets/JamieSJS/r-oxford-medium-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Oxford_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



#### RP2kI2IRetrieval

Retrieve photos of 39457 products.

**Dataset:** [`JamieSJS/rp2k`](https://huggingface.co/datasets/JamieSJS/rp2k) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2006.12634)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Web | derived | created |



#### RParisEasyI2IRetrieval

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`JamieSJS/r-paris-easy-multi`](https://huggingface.co/datasets/JamieSJS/r-paris-easy-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



#### RParisHardI2IRetrieval

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`JamieSJS/r-paris-hard-multi`](https://huggingface.co/datasets/JamieSJS/r-paris-hard-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



#### RParisMediumI2IRetrieval

Retrieve photos of landmarks in Paris, UK.

**Dataset:** [`JamieSJS/r-paris-medium-multi`](https://huggingface.co/datasets/JamieSJS/r-paris-medium-multi) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/html/Radenovic_Revisiting_Paris_and_CVPR_2018_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | map_at_5 | eng | Web | derived | created |



#### ReMuQIT2TRetrieval

Retrieval of a Wiki passage to answer a query about an image.

**Dataset:** [`izhx/UMRB-ReMuQ`](https://huggingface.co/datasets/izhx/UMRB-ReMuQ) • **License:** cc0-1.0 • [Learn more →](https://github.com/luomancs/ReMuQ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | cv_recall_at_5 | eng | Encyclopaedic | derived | created |



#### SOPI2IRetrieval

Retrieve product photos of 22634 online products.

**Dataset:** [`JamieSJS/stanford-online-products`](https://huggingface.co/datasets/JamieSJS/stanford-online-products) • **License:** not specified • [Learn more →](https://paperswithcode.com/dataset/stanford-online-products)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



#### SciMMIRI2TRetrieval

Retrieve captions based on figures and tables.

**Dataset:** [`m-a-p/SciMMIR`](https://huggingface.co/datasets/m-a-p/SciMMIR) • **License:** mit • [Learn more →](https://aclanthology.org/2024.findings-acl.746/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Academic | derived | found |



#### SciMMIRT2IRetrieval

Retrieve figures and tables based on captions.

**Dataset:** [`m-a-p/SciMMIR`](https://huggingface.co/datasets/m-a-p/SciMMIR) • **License:** mit • [Learn more →](https://aclanthology.org/2024.findings-acl.746/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Academic | derived | found |



#### SketchyI2IRetrieval

Retrieve photos from sketches.

**Dataset:** [`JamieSJS/sketchy`](https://huggingface.co/datasets/JamieSJS/sketchy) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2202.01747)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



#### StanfordCarsI2IRetrieval

Retrieve car images from 196 makes.

**Dataset:** [`isaacchung/stanford_cars_retrieval`](https://huggingface.co/datasets/isaacchung/stanford_cars_retrieval) • **License:** not specified • [Learn more →](https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cv_recall_at_1 | eng | Encyclopaedic | derived | created |



#### TUBerlinT2IRetrieval

Retrieve sketch images based on text descriptions.

**Dataset:** [`gowitheflow/tu-berlin`](https://huggingface.co/datasets/gowitheflow/tu-berlin) • **License:** cc-by-sa-4.0 • [Learn more →](https://dl.acm.org/doi/pdf/10.1145/2185520.2185540)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | found |



#### VQA2IT2TRetrieval

Retrieve the correct answer for a question about an image.

**Dataset:** [`JamieSJS/vqa-2`](https://huggingface.co/datasets/JamieSJS/vqa-2) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Web | derived | found |



#### VisualNewsI2TRetrieval

Retrieval entity-rich captions for news images.

**Dataset:** [`MRBench/mbeir_visualnews_task3`](https://huggingface.co/datasets/MRBench/mbeir_visualnews_task3) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.emnlp-main.542/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### VisualNewsT2IRetrieval

Retrieve news images with captions.

**Dataset:** [`MRBench/mbeir_visualnews_task0`](https://huggingface.co/datasets/MRBench/mbeir_visualnews_task0) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.emnlp-main.542/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### VizWizIT2TRetrieval

Retrieve the correct answer for a question about an image.

**Dataset:** [`JamieSJS/vizwiz`](https://huggingface.co/datasets/JamieSJS/vizwiz) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | ndcg_at_10 | eng | Web | derived | found |



#### WebQAT2ITRetrieval

Retrieve sources of information based on questions.

**Dataset:** [`MRBench/mbeir_webqa_task2`](https://huggingface.co/datasets/MRBench/mbeir_webqa_task2) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image, text (t2it) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



#### WebQAT2TRetrieval

Retrieve sources of information based on questions.

**Dataset:** [`MRBench/mbeir_webqa_task1`](https://huggingface.co/datasets/MRBench/mbeir_webqa_task1) • **License:** cc-by-sa-4.0 • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic | derived | created |



## BitextMining

- **Number of tasks of the given type:** 30 

#### BUCC

BUCC bitext mining dataset

**Dataset:** [`mteb/bucc-bitext-mining`](https://huggingface.co/datasets/mteb/bucc-bitext-mining) • **License:** not specified • [Learn more →](https://comparable.limsi.fr/bucc2018/bucc2018-task.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | cmn, deu, eng, fra, rus | Written | human-annotated | human-translated |



#### BUCC.v2

BUCC bitext mining dataset

**Dataset:** [`mteb/bucc-bitext-mining`](https://huggingface.co/datasets/mteb/bucc-bitext-mining) • **License:** not specified • [Learn more →](https://comparable.limsi.fr/bucc2018/bucc2018-task.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | cmn, deu, eng, fra, rus | Written | human-annotated | human-translated |



#### BibleNLPBitextMining

Partial Bible translations in 829 languages, aligned by verse.

**Dataset:** [`davidstap/biblenlp-corpus-mmteb`](https://huggingface.co/datasets/davidstap/biblenlp-corpus-mmteb) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.09919)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | aai, aak, aau, aaz, abt, ... (829) | Religious, Written | expert-annotated | created |



#### BornholmBitextMining

Danish Bornholmsk Parallel Corpus. Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.

**Dataset:** [`strombergnlp/bornholmsk_parallel`](https://huggingface.co/datasets/strombergnlp/bornholmsk_parallel) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/W19-6138/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | dan | Fiction, Social, Web, Written | expert-annotated | created |



#### DanishMedicinesAgencyBitextMining

A Bilingual English-Danish parallel corpus from The Danish Medicines Agency.

**Dataset:** [`mteb/english-danish-parallel-corpus`](https://huggingface.co/datasets/mteb/english-danish-parallel-corpus) • **License:** https://opendefinition.org/od/2.1/en/ • [Learn more →](https://sprogteknologi.dk/dataset/bilingual-english-danish-parallel-corpus-from-the-danish-medicines-agency)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | dan, eng | Medical, Written | human-annotated | found |



#### DiaBlaBitextMining

English-French Parallel Corpus. DiaBLa is an English-French dataset for the evaluation of Machine Translation (MT) for informal, written bilingual dialogue.

**Dataset:** [`rbawden/DiaBLa`](https://huggingface.co/datasets/rbawden/DiaBLa) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://inria.hal.science/hal-03021633)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng, fra | Social, Written | human-annotated | created |



#### FloresBitextMining

FLORES is a benchmark dataset for machine translation between English and low-resource languages.

**Dataset:** [`mteb/flores`](https://huggingface.co/datasets/mteb/flores) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/facebook/flores)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | ace, acm, acq, aeb, afr, ... (196) | Encyclopaedic, Non-fiction, Written | human-annotated | created |



#### IN22ConvBitextMining

IN22-Conv is a n-way parallel conversation domain benchmark dataset for machine translation spanning English and 22 Indic languages.

**Dataset:** [`mteb/IN22-Conv`](https://huggingface.co/datasets/mteb/IN22-Conv) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/ai4bharat/IN22-Conv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | asm, ben, brx, doi, eng, ... (23) | Fiction, Social, Spoken, Spoken | expert-annotated | created |



#### IN22GenBitextMining

IN22-Gen is a n-way parallel general-purpose multi-domain benchmark dataset for machine translation spanning English and 22 Indic languages.

**Dataset:** [`mteb/IN22-Gen`](https://huggingface.co/datasets/mteb/IN22-Gen) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/ai4bharat/IN22-Gen)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | asm, ben, brx, doi, eng, ... (23) | Government, Legal, News, Non-fiction, Religious, ... (7) | expert-annotated | created |



#### IWSLT2017BitextMining

The IWSLT 2017 Multilingual Task addresses text translation, including zero-shot translation, with a single MT system across all directions including English, German, Dutch, Italian and Romanian.

**Dataset:** [`mteb/IWSLT2017BitextMining`](https://huggingface.co/datasets/mteb/IWSLT2017BitextMining) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://aclanthology.org/2017.iwslt-1.1/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | ara, cmn, deu, eng, fra, ... (10) | Fiction, Non-fiction, Written | expert-annotated | found |



#### IndicGenBenchFloresBitextMining

Flores-IN dataset is an extension of Flores dataset released as a part of the IndicGenBench by Google

**Dataset:** [`google/IndicGenBench_flores_in`](https://huggingface.co/datasets/google/IndicGenBench_flores_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/google-research-datasets/indic-gen-bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | asm, awa, ben, bgc, bho, ... (30) | News, Web, Written | expert-annotated | human-translated and localized |



#### LinceMTBitextMining

LinceMT is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.

**Dataset:** [`gentaiscool/bitext_lincemt_miners`](https://huggingface.co/datasets/gentaiscool/bitext_lincemt_miners) • **License:** not specified • [Learn more →](https://ritual.uh.edu/lince/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng, hin | Social, Written | human-annotated | found |



#### NTREXBitextMining

NTREX is a News Test References dataset for Machine Translation Evaluation, covering translation from English into 128 languages. We select language pairs according to the M2M-100 language grouping strategy, resulting in 1916 directions.

**Dataset:** [`mteb/NTREX`](https://huggingface.co/datasets/mteb/NTREX) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/davidstap/NTREX)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | afr, amh, arb, aze, bak, ... (119) | News, Written | expert-annotated | human-translated and localized |



#### NollySentiBitextMining

NollySenti is Nollywood movie reviews for five languages widely spoken in Nigeria (English, Hausa, Igbo, Nigerian-Pidgin, and Yoruba.

**Dataset:** [`gentaiscool/bitext_nollysenti_miners`](https://huggingface.co/datasets/gentaiscool/bitext_nollysenti_miners) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/IyanuSh/NollySenti)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng, hau, ibo, pcm, yor | Reviews, Social, Written | human-annotated | found |



#### NorwegianCourtsBitextMining

Nynorsk and Bokmål parallel corpus from Norwegian courts. Norwegian courts have two standardised written languages. Bokmål is a variant closer to Danish, while Nynorsk was created to resemble regional dialects of Norwegian.

**Dataset:** [`kardosdrur/norwegian-courts`](https://huggingface.co/datasets/kardosdrur/norwegian-courts) • **License:** cc-by-4.0 • [Learn more →](https://opus.nlpl.eu/index.php)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | nno, nob | Legal, Written | human-annotated | found |



#### NusaTranslationBitextMining

NusaTranslation is a parallel dataset for machine translation on 11 Indonesia languages and English.

**Dataset:** [`gentaiscool/bitext_nusatranslation_miners`](https://huggingface.co/datasets/gentaiscool/bitext_nusatranslation_miners) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/indonlp/nusatranslation_mt)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | abs, bbc, bew, bhp, ind, ... (12) | Social, Written | human-annotated | created |



#### NusaXBitextMining

NusaX is a parallel dataset for machine translation and sentiment analysis on 11 Indonesia languages and English.

**Dataset:** [`gentaiscool/bitext_nusax_miners`](https://huggingface.co/datasets/gentaiscool/bitext_nusax_miners) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/indonlp/NusaX-senti/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | ace, ban, bbc, bjn, bug, ... (12) | Reviews, Written | human-annotated | created |



#### PhincBitextMining

Phinc is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.

**Dataset:** [`gentaiscool/bitext_phinc_miners`](https://huggingface.co/datasets/gentaiscool/bitext_phinc_miners) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/veezbo/phinc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng, hin | Social, Written | human-annotated | found |



#### PubChemSMILESBitextMining

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemSMILESBitextMining`](https://huggingface.co/datasets/BASF-AI/PubChemSMILESBitextMining) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng | Chemistry | derived | created |



#### RomaTalesBitextMining

Parallel corpus of Roma Tales in Lovari with Hungarian translations.

**Dataset:** [`kardosdrur/roma-tales`](https://huggingface.co/datasets/kardosdrur/roma-tales) • **License:** not specified • [Learn more →](https://idoc.pub/documents/idocpub-zpnxm9g35ylv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | hun, rom | Fiction, Written | expert-annotated | created |



#### RuSciBenchBitextMining

This task focuses on finding translations of scientific articles.
        The dataset is sourced from eLibrary, Russia's largest electronic library of scientific publications.
        Russian authors often provide English translations for their abstracts and titles,
        and the data consists of these paired titles and abstracts. The task evaluates a model's ability
        to match an article's Russian title and abstract to its English counterpart, or vice versa.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_bitext_mining`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_bitext_mining) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | eng, rus | Academic, Non-fiction, Written | derived | found |



#### SAMSumFa

Translated Version of SAMSum Dataset for summary retrieval.

**Dataset:** [`MCINext/samsum-fa`](https://huggingface.co/datasets/MCINext/samsum-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/samsum-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | fas | Spoken | LM-generated | machine-translated |



#### SRNCorpusBitextMining

SRNCorpus is a machine translation corpus for creole language Sranantongo and Dutch.

**Dataset:** [`davidstap/sranantongo`](https://huggingface.co/datasets/davidstap/sranantongo) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2212.06383)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | nld, srn | Social, Web, Written | human-annotated | found |



#### SynPerChatbotRAGSumSRetrieval

Synthetic Persian Chatbot RAG Summary Dataset for summary retrieval.

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-summary-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotSumSRetrieval

Synthetic Persian Chatbot Summary Dataset for summary retrieval.

**Dataset:** [`MCINext/synthetic-persian-chatbot-summary-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | fas | Spoken | LM-generated | LM-generated and verified |



#### Tatoeba

1,000 English-aligned sentence pairs for each language based on the Tatoeba corpus

**Dataset:** [`mteb/tatoeba-bitext-mining`](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining) • **License:** cc-by-2.0 • [Learn more →](https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | afr, amh, ang, ara, arq, ... (113) | Written | human-annotated | found |



#### TbilisiCityHallBitextMining

Parallel news titles from the Tbilisi City Hall website (https://tbilisi.gov.ge/).

**Dataset:** [`jupyterjazz/tbilisi-city-hall-titles`](https://huggingface.co/datasets/jupyterjazz/tbilisi-city-hall-titles) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jupyterjazz/tbilisi-city-hall-titles)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng, kat | News, Written | derived | created |



#### VieMedEVBitextMining

A high-quality Vietnamese-English parallel data from the medical domain for machine translation

**Dataset:** [`nhuvo/MedEV`](https://huggingface.co/datasets/nhuvo/MedEV) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/2015.iwslt-evaluation.11/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | eng, vie | Medical, Written | expert-annotated | human-translated and localized |



#### WebFAQBitextMiningQAs

The WebFAQ Bitext Dataset consists of natural FAQ-style Question-Answer pairs that align across languages.
A sentence in the "WebFAQBitextMiningQAs" task is a concatenation of a question and its corresponding answer.
The dataset is sourced from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-bitexts`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-bitexts) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | ara, aze, ben, bul, cat, ... (47) | Web, Written | human-annotated | human-translated |



#### WebFAQBitextMiningQuestions

The WebFAQ Bitext Dataset consists of natural FAQ-style Question-Answer pairs that align across languages.
A sentence in the "WebFAQBitextMiningQuestions" task is the question originating from an aligned QA.
The dataset is sourced from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-bitexts`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-bitexts) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | f1 | ara, aze, ben, bul, cat, ... (47) | Web, Written | human-annotated | human-translated |



## Classification

- **Number of tasks of the given type:** 456 

#### AJGT

Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.

**Dataset:** [`komari6/ajgt_twitter_ar`](https://huggingface.co/datasets/komari6/ajgt_twitter_ar) • **License:** afl-3.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### AJGT.v2

Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets (900 for training and 900 for testing) annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/ajgt`](https://huggingface.co/datasets/mteb/ajgt) • **License:** afl-3.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### AfriSentiClassification

AfriSenti is the largest sentiment analysis dataset for under-represented African languages.

**Dataset:** [`shmuhammad/AfriSenti-twitter-sentiment`](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2302.08956)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, arq, ary, hau, ibo, ... (12) | Social, Written | derived | found |



#### AfriSentiLangClassification

AfriSentiLID is the largest LID classification dataset for African Languages.

**Dataset:** [`HausaNLP/afrisenti-lid-data`](https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, arq, ary, hau, ibo, ... (12) | Social, Written | derived | found |



#### AllegroReviews

A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.

**Dataset:** [`PL-MTEB/allegro-reviews`](https://huggingface.co/datasets/PL-MTEB/allegro-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.acl-main.111.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Reviews | derived | found |



#### AllegroReviews.v2

A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/allegro_reviews`](https://huggingface.co/datasets/mteb/allegro_reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.acl-main.111.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Reviews | derived | found |



#### AmazonCounterfactualClassification

A collection of Amazon customer reviews annotated for counterfactual detection pair classification.

**Dataset:** [`mteb/amazon_counterfactual`](https://huggingface.co/datasets/mteb/amazon_counterfactual) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2104.06893)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, jpn | Reviews, Written | human-annotated | found |



#### AmazonCounterfactualVNClassification

A collection of translated Amazon customer reviews annotated for counterfactual detection pair classification.
        The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
        - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
        - Applies advanced embedding models to filter the translations.
        - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.
        

**Dataset:** [`GreenNode/amazon-counterfactual-vn`](https://huggingface.co/datasets/GreenNode/amazon-counterfactual-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.06893)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### AmazonPolarityClassification

Amazon Polarity Classification Dataset.

**Dataset:** [`mteb/amazon_polarity`](https://huggingface.co/datasets/mteb/amazon_polarity) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### AmazonPolarityClassification.v2

Amazon Polarity Classification Dataset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/amazon_polarity`](https://huggingface.co/datasets/mteb/amazon_polarity) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### AmazonPolarityVNClassification

A collection of translated Amazon customer reviews annotated for polarity classification.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.
            

**Dataset:** [`GreenNode/amazon-polarity-vn`](https://huggingface.co/datasets/GreenNode/amazon-polarity-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### AmazonReviewsClassification

A collection of Amazon reviews specifically designed to aid research in multilingual text classification.

**Dataset:** [`mteb/AmazonReviewsClassification`](https://huggingface.co/datasets/mteb/AmazonReviewsClassification) • **License:** https://docs.opendata.aws/amazon-reviews-ml/license.txt • [Learn more →](https://arxiv.org/abs/2010.02573)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn, deu, eng, fra, jpn, ... (6) | Reviews, Written | human-annotated | found |



#### AmazonReviewsVNClassification

A collection of translated Amazon reviews specifically designed to aid research in multilingual text classification.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/amazon-reviews-multi-vn`](https://huggingface.co/datasets/GreenNode/amazon-reviews-multi-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2010.02573)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### AngryTweetsClassification

A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets

**Dataset:** [`DDSC/angry-tweets`](https://huggingface.co/datasets/DDSC/angry-tweets) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.53/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | human-annotated | found |



#### AngryTweetsClassification.v2

A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/angry_tweets`](https://huggingface.co/datasets/mteb/angry_tweets) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.53/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | human-annotated | found |



#### ArxivClassification

Classification Dataset of Arxiv Papers

**Dataset:** [`mteb/ArxivClassification`](https://huggingface.co/datasets/mteb/ArxivClassification) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8675939)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Academic, Written | derived | found |



#### ArxivClassification.v2

Classification Dataset of Arxiv Papers
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/arxiv`](https://huggingface.co/datasets/mteb/arxiv) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8675939)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Academic, Written | derived | found |



#### Banking77Classification

Dataset composed of online banking queries annotated with their corresponding intents.

**Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) • **License:** mit • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Written | human-annotated | found |



#### Banking77Classification.v2

Dataset composed of online banking queries annotated with their corresponding intents.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) • **License:** mit • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Written | human-annotated | found |



#### Banking77VNClassification

A translated dataset composed of online banking queries annotated with their corresponding intents.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/banking77-vn`](https://huggingface.co/datasets/GreenNode/banking77-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Written | derived | machine-translated and LM verified |



#### BengaliDocumentClassification

Dataset for News Classification, categorized with 13 domains.

**Dataset:** [`dialect-ai/shironaam`](https://huggingface.co/datasets/dialect-ai/shironaam) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2023.eacl-main.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ben | News, Written | derived | found |



#### BengaliDocumentClassification.v2

Dataset for News Classification, categorized with 13 domains.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_document`](https://huggingface.co/datasets/mteb/bengali_document) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2023.eacl-main.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ben | News, Written | derived | found |



#### BengaliHateSpeechClassification

The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.

**Dataset:** [`rezacsedu/bn_hate_speech`](https://huggingface.co/datasets/rezacsedu/bn_hate_speech) • **License:** mit • [Learn more →](https://huggingface.co/datasets/bn_hate_speech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | News, Written | expert-annotated | found |



#### BengaliHateSpeechClassification.v2

The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_hate_speech`](https://huggingface.co/datasets/mteb/bengali_hate_speech) • **License:** mit • [Learn more →](https://huggingface.co/datasets/bn_hate_speech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | News, Written | expert-annotated | found |



#### BengaliSentimentAnalysis

dataset contains 3307 Negative reviews and 8500 Positive reviews collected and manually annotated from Youtube Bengali drama.

**Dataset:** [`Akash190104/bengali_sentiment_analysis`](https://huggingface.co/datasets/Akash190104/bengali_sentiment_analysis) • **License:** cc-by-4.0 • [Learn more →](https://data.mendeley.com/datasets/p6zc7krs37/4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | Reviews, Written | human-annotated | found |



#### BengaliSentimentAnalysis.v2

dataset contains 2854 Negative reviews and 7238 Positive reviews collected and manually annotated from Youtube Bengali drama.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_sentiment_analysis`](https://huggingface.co/datasets/mteb/bengali_sentiment_analysis) • **License:** cc-by-4.0 • [Learn more →](https://data.mendeley.com/datasets/p6zc7krs37/4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | Reviews, Written | human-annotated | found |



#### BulgarianStoreReviewSentimentClassfication

Bulgarian online store review dataset for sentiment classification.

**Dataset:** [`artist/Bulgarian-Online-Store-Feedback-Text-Analysis`](https://huggingface.co/datasets/artist/Bulgarian-Online-Store-Feedback-Text-Analysis) • **License:** cc-by-4.0 • [Learn more →](https://doi.org/10.7910/DVN/TXIK9P)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bul | Reviews, Written | human-annotated | found |



#### CBD

Polish Tweets annotated for cyberbullying detection.

**Dataset:** [`PL-MTEB/cbd`](https://huggingface.co/datasets/PL-MTEB/cbd) • **License:** bsd-3-clause • [Learn more →](http://2019.poleval.pl/files/poleval2019.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | human-annotated | found |



#### CBD.v2

Polish Tweets annotated for cyberbullying detection.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/cbd`](https://huggingface.co/datasets/mteb/cbd) • **License:** bsd-3-clause • [Learn more →](http://2019.poleval.pl/files/poleval2019.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | human-annotated | found |



#### CSFDCZMovieReviewSentimentClassification

The dataset contains 30k user reviews from csfd.cz in Czech.

**Dataset:** [`fewshot-goes-multilingual/cs_csfd-movie-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_csfd-movie-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CSFDCZMovieReviewSentimentClassification.v2

The dataset contains 30k user reviews from csfd.cz in Czech.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/csfdcz_movie_review_sentiment`](https://huggingface.co/datasets/mteb/csfdcz_movie_review_sentiment) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CSFDSKMovieReviewSentimentClassification

The dataset contains 30k user reviews from csfd.cz in Slovak.

**Dataset:** [`fewshot-goes-multilingual/sk_csfd-movie-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/sk_csfd-movie-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Reviews, Written | derived | found |



#### CSFDSKMovieReviewSentimentClassification.v2

The dataset contains 30k user reviews from csfd.cz in Slovak.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/csfdsk_movie_review_sentiment`](https://huggingface.co/datasets/mteb/csfdsk_movie_review_sentiment) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Reviews, Written | derived | found |



#### CUADAffiliateLicenseLicenseeLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if a clause describes a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADAffiliateLicenseLicensorLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause describes a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADAntiAssignmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires consent or notice of a party if the contract is assigned to a third party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADAuditRightsLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause gives a party the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADCapOnLiabilityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a cap on liability upon the breach of a party's obligation. This includes time limitation for the counterparty to bring claims or maximum amount for recovery.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADChangeOfControlLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause gives one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADCompetitiveRestrictionExceptionLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause mentions exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADCovenantNotToSueLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party is restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADEffectiveDateLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the agreement becomes effective.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADExclusivityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies exclusive dealing commitment with the counterparty. This includes a commitment to procure all 'requirements' from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADExpirationDateLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the initial term expires.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADGoverningLawLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies which state/country’s law governs the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADIPOwnershipAssignmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADInsuranceLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if clause creates a requirement for insurance that must be maintained by one party for the benefit of the counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADIrrevocableOrPerpetualLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a license grant that is irrevocable or perpetual.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADJointIPOwnershipLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause provides for joint or shared ownership of intellectual property between the parties to the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADLicenseGrantLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause contains a license granted by one party to its counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADLiquidatedDamagesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause awards either party liquidated damages for breach or a fee upon the termination of a contract (termination fee).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADMinimumCommitmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a minimum order size or minimum amount or units per time period that one party must buy from the counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADMostFavoredNationLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNoSolicitOfCustomersLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNoSolicitOfEmployeesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNonCompeteLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNonDisparagementLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires a party not to disparage the counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNonTransferableLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause limits the ability of a party to transfer the license being granted to a third party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNoticePeriodToTerminateRenewalLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a notice period required to terminate renewal.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADPostTerminationServicesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause subjects a party to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADPriceRestrictionsLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause places a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADRenewalTermLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a renewal term.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADRevenueProfitSharingLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause require a party to share revenue or profit with the counterparty for any technology, goods, or services.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADRofrRofoRofnLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause grant one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADSourceCodeEscrowLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires one party to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADTerminationForConvenienceLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that one party can terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADThirdPartyBeneficiaryLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that that there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADUncappedLiabilityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party's liability is uncapped upon the breach of its obligation in the contract. This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause grants one party an “enterprise,” “all you can eat” or unlimited usage license.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADVolumeRestrictionLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADWarrantyDurationLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a duration of any warranty against defects or errors in technology, products, or services provided under the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CanadaTaxCourtOutcomesLegalBenchClassification

The input is an excerpt of text from Tax Court of Canada decisions involving appeals of tax related matters. The task is to classify whether the excerpt includes the outcome of the appeal, and if so, to specify whether the appeal was allowed or dismissed. Partial success (e.g. appeal granted on one tax year but dismissed on another) counts as allowed (with the exception of costs orders which are disregarded). Where the excerpt does not clearly articulate an outcome, the system should indicate other as the outcome. Categorizing case outcomes is a common task that legal researchers complete in order to gather datasets involving outcomes in legal processes for the purposes of quantitative empirical legal research.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CataloniaTweetClassification

This dataset contains two corpora in Spanish and Catalan that consist of annotated Twitter
        messages for automatic stance detection. The data was collected over 12 days during February and March
        of 2019 from tweets posted in Barcelona, and during September of 2018 from tweets posted in the town of Terrassa, Catalonia.
        Each corpus is annotated with three classes: AGAINST, FAVOR and NEUTRAL, which express the stance
        towards the target - independence of Catalonia.
        

**Dataset:** [`community-datasets/catalonia_independence`](https://huggingface.co/datasets/community-datasets/catalonia_independence) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.171/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cat, spa | Government, Social, Written | expert-annotated | created |



#### ContractNLIConfidentialityOfAgreementLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA provides that the Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIExplicitIdentificationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that all Confidential Information shall be expressly identified by the Disclosing Party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that Confidential Information may include verbally conveyed information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLILimitedUseLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLINoLicensingLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Agreement shall not grant Receiving Party any right to Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLINoticeOnCompelledDisclosureLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may acquire information similar to Confidential Information from a third party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissibleCopyLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may create a copy of some Confidential Information in some circumstances.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may independently develop information similar to Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIReturnOfConfidentialInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLISharingWithEmployeesLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some of Receiving Party's employees.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLISharingWithThirdPartiesLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLISurvivalOfObligationsLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that some obligations of Agreement may survive termination of Agreement.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CorporateLobbyingLegalBenchClassification

The Corporate Lobbying task consists of determining whether a proposed Congressional bill may be relevant to a company based on a company's self-description in its SEC 10K filing.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CyrillicTurkicLangClassification

Cyrillic dataset of 8 Turkic languages spoken in Russia and former USSR

**Dataset:** [`tatiana-merz/cyrillic_turkic_langs`](https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bak, chv, kaz, kir, krc, ... (9) | Web, Written | derived | found |



#### CzechProductReviewSentimentClassification

User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)

**Dataset:** [`fewshot-goes-multilingual/cs_mall-product-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_mall-product-reviews) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechProductReviewSentimentClassification.v2

User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/czech_product_review_sentiment`](https://huggingface.co/datasets/mteb/czech_product_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechSoMeSentimentClassification

User comments on Facebook

**Dataset:** [`fewshot-goes-multilingual/cs_facebook-comments`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_facebook-comments) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechSoMeSentimentClassification.v2

User comments on Facebook
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/czech_so_me_sentiment`](https://huggingface.co/datasets/mteb/czech_so_me_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechSubjectivityClassification

An Czech dataset for subjectivity classification.

**Dataset:** [`pauli31/czech-subjectivity-dataset`](https://huggingface.co/datasets/pauli31/czech-subjectivity-dataset) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | human-annotated | found |



#### DBpediaClassification

DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.

**Dataset:** [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Encyclopaedic, Written | derived | found |



#### DBpediaClassification.v2

DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/d_bpedia`](https://huggingface.co/datasets/mteb/d_bpedia) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Encyclopaedic, Written | derived | found |



#### DKHateClassification

Danish Tweets annotated for Hate Speech either being Offensive or not

**Dataset:** [`DDSC/dkhate`](https://huggingface.co/datasets/DDSC/dkhate) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.430/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | expert-annotated | found |



#### DKHateClassification.v2

Danish Tweets annotated for Hate Speech either being Offensive or not
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dk_hate`](https://huggingface.co/datasets/mteb/dk_hate) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.430/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | expert-annotated | found |



#### DadoEvalCoarseClassification

The DaDoEval dataset is a curated collection of 2,759 documents authored by Alcide De Gasperi, spanning the period from 1901 to 1954. Each document in the dataset is manually tagged with its date of issue.

**Dataset:** [`MattiaSangermano/DaDoEval`](https://huggingface.co/datasets/MattiaSangermano/DaDoEval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/dhfbk/DaDoEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Written | derived | found |



#### DalajClassification

A Swedish dataset for linguistic acceptability. Available as a part of Superlim.

**Dataset:** [`AI-Sweden/SuperLim`](https://huggingface.co/datasets/AI-Sweden/SuperLim) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Non-fiction, Written | expert-annotated | created |



#### DalajClassification.v2

A Swedish dataset for linguistic acceptability. Available as a part of Superlim.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dalaj`](https://huggingface.co/datasets/mteb/dalaj) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Non-fiction, Written | expert-annotated | created |



#### DanishPoliticalCommentsClassification

A dataset of Danish political comments rated for sentiment

**Dataset:** [`community-datasets/danish_political_comments`](https://huggingface.co/datasets/community-datasets/danish_political_comments) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/danish_political_comments)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | derived | found |



#### DanishPoliticalCommentsClassification.v2

A dataset of Danish political comments rated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/danish_political_comments`](https://huggingface.co/datasets/mteb/danish_political_comments) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/danish_political_comments)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | derived | found |



#### Ddisco

A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)

**Dataset:** [`DDSC/ddisco`](https://huggingface.co/datasets/DDSC/ddisco) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.260/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Non-fiction, Social, Written | expert-annotated | found |



#### Ddisco.v2

A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ddisco_cohesion`](https://huggingface.co/datasets/mteb/ddisco_cohesion) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.260/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Non-fiction, Social, Written | expert-annotated | found |



#### DeepSentiPers

Persian Sentiment Analysis Dataset

**Dataset:** [`PartAI/DeepSentiPers`](https://huggingface.co/datasets/PartAI/DeepSentiPers) • **License:** not specified • [Learn more →](https://github.com/JoyeBright/DeepSentiPers)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### DeepSentiPers.v2

Persian Sentiment Analysis Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/deep_senti_pers`](https://huggingface.co/datasets/mteb/deep_senti_pers) • **License:** not specified • [Learn more →](https://github.com/JoyeBright/DeepSentiPers)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### DefinitionClassificationLegalBenchClassification

This task consists of determining whether or not a sentence from a Supreme Court opinion offers a definition of a term.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### DigikalamagClassification

A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.

**Dataset:** [`PNLPhub/DigiMag`](https://huggingface.co/datasets/PNLPhub/DigiMag) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/tc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Web | derived | found |



#### Diversity1LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 1).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity2LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 2).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity3LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 3).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity4LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 4).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity5LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 5).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity6LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 6).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### DutchBookReviewSentimentClassification

A Dutch book review for sentiment classification.

**Dataset:** [`mteb/DutchBookReviewSentimentClassification`](https://huggingface.co/datasets/mteb/DutchBookReviewSentimentClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/benjaminvdb/DBRD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nld | Reviews, Written | derived | found |



#### DutchBookReviewSentimentClassification.v2

A Dutch book review for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dutch_book_review_sentiment`](https://huggingface.co/datasets/mteb/dutch_book_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/benjaminvdb/DBRD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nld | Reviews, Written | derived | found |



#### EmotionClassification

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### EmotionClassification.v2

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### EmotionVNClassification

Emotion is a translated dataset of Vietnamese from English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/emotion-vn`](https://huggingface.co/datasets/GreenNode/emotion-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Social, Written | derived | machine-translated and LM verified |



#### EstonianValenceClassification

Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.

**Dataset:** [`kardosdrur/estonian-valence`](https://huggingface.co/datasets/kardosdrur/estonian-valence) • **License:** cc-by-4.0 • [Learn more →](https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | est | News, Written | human-annotated | found |



#### EstonianValenceClassification.v2

Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/estonian_valence`](https://huggingface.co/datasets/mteb/estonian_valence) • **License:** cc-by-4.0 • [Learn more →](https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | est | News, Written | human-annotated | found |



#### FilipinoHateSpeechClassification

Filipino Twitter dataset for sentiment classification.

**Dataset:** [`mteb/FilipinoHateSpeechClassification`](https://huggingface.co/datasets/mteb/FilipinoHateSpeechClassification) • **License:** not specified • [Learn more →](https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



#### FilipinoHateSpeechClassification.v2

Filipino Twitter dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/filipino_hate_speech`](https://huggingface.co/datasets/mteb/filipino_hate_speech) • **License:** not specified • [Learn more →](https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



#### FilipinoShopeeReviewsClassification

The Shopee reviews tl 15 dataset is constructed by randomly taking 2100 training samples and 450 samples for testing and validation for each review star from 1 to 5. In total, there are 10500 training samples and 2250 each in validation and testing samples.

**Dataset:** [`scaredmeow/shopee-reviews-tl-stars`](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars) • **License:** mpl-2.0 • [Learn more →](https://uijrt.com/articles/v4/i8/UIJRTV4I80009.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



#### FinToxicityClassification


        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.
        

**Dataset:** [`TurkuNLP/jigsaw_toxicity_pred_fi`](https://huggingface.co/datasets/TurkuNLP/jigsaw_toxicity_pred_fi) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.68)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | fin | News, Written | derived | machine-translated |



#### FinToxicityClassification.v2


        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/fin_toxicity`](https://huggingface.co/datasets/mteb/fin_toxicity) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.68)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | fin | News, Written | derived | machine-translated |



#### FinancialPhrasebankClassification

Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.

**Dataset:** [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://arxiv.org/abs/1307.5336)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Financial, News, Written | expert-annotated | found |



#### FinancialPhrasebankClassification.v2

Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/financial_phrasebank`](https://huggingface.co/datasets/mteb/financial_phrasebank) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://arxiv.org/abs/1307.5336)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Financial, News, Written | expert-annotated | found |



#### FrenchBookReviews

It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).

**Dataset:** [`Abirate/french_book_reviews`](https://huggingface.co/datasets/Abirate/french_book_reviews) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/Abirate/french_book_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### FrenchBookReviews.v2

It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/french_book_reviews`](https://huggingface.co/datasets/mteb/french_book_reviews) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/Abirate/french_book_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### FrenkEnClassification

English subset of the FRENK dataset

**Dataset:** [`classla/FRENK-hate-en`](https://huggingface.co/datasets/classla/FRENK-hate-en) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | derived | found |



#### FrenkEnClassification.v2

English subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_en`](https://huggingface.co/datasets/mteb/frenk_en) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | derived | found |



#### FrenkHrClassification

Croatian subset of the FRENK dataset

**Dataset:** [`classla/FRENK-hate-hr`](https://huggingface.co/datasets/classla/FRENK-hate-hr) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hrv | Social, Written | derived | found |



#### FrenkHrClassification.v2

Croatian subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_hr`](https://huggingface.co/datasets/mteb/frenk_hr) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hrv | Social, Written | derived | found |



#### FrenkSlClassification

Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.

**Dataset:** [`classla/FRENK-hate-sl`](https://huggingface.co/datasets/classla/FRENK-hate-sl) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slv | Social, Written | derived | found |



#### FrenkSlClassification.v2

Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_sl`](https://huggingface.co/datasets/mteb/frenk_sl) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slv | Social, Written | derived | found |



#### FunctionOfDecisionSectionLegalBenchClassification

The task is to classify a paragraph extracted from a written court decision into one of seven possible categories:
            1. Facts - The paragraph describes the faction background that led up to the present lawsuit.
            2. Procedural History - The paragraph describes the course of litigation that led to the current proceeding before the court.
            3. Issue - The paragraph describes the legal or factual issue that must be resolved by the court.
            4. Rule - The paragraph describes a rule of law relevant to resolving the issue.
            5. Analysis - The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.
            6. Conclusion - The paragraph presents a conclusion of the court.
            7. Decree - The paragraph constitutes a decree resolving the dispute.
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### GeoreviewClassification

Review classification (5-point scale) based on Yandex Georeview dataset

**Dataset:** [`ai-forever/georeview-classification`](https://huggingface.co/datasets/ai-forever/georeview-classification) • **License:** mit • [Learn more →](https://github.com/yandex/geo-reviews-dataset-2023)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### GeoreviewClassification.v2

Review classification (5-point scale) based on Yandex Georeview dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/georeview`](https://huggingface.co/datasets/mteb/georeview) • **License:** mit • [Learn more →](https://github.com/yandex/geo-reviews-dataset-2023)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### GeorgianSentimentClassification

Goergian Sentiment Dataset

**Dataset:** [`asparius/Georgian-Sentiment`](https://huggingface.co/datasets/asparius/Georgian-Sentiment) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.173)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kat | Reviews, Written | derived | found |



#### GermanPoliticiansTwitterSentimentClassification

GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).

**Dataset:** [`Alienmaster/german_politicians_twitter_sentiment`](https://huggingface.co/datasets/Alienmaster/german_politicians_twitter_sentiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.konvens-1.9)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | Government, Social, Written | human-annotated | found |



#### GermanPoliticiansTwitterSentimentClassification.v2

GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/german_politicians_twitter_sentiment`](https://huggingface.co/datasets/mteb/german_politicians_twitter_sentiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.konvens-1.9)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | Government, Social, Written | human-annotated | found |



#### GreekLegalCodeClassification

Greek Legal Code Dataset for Classification. (subset = chapter)

**Dataset:** [`AI-team-UoA/greek_legal_code`](https://huggingface.co/datasets/AI-team-UoA/greek_legal_code) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2109.15298)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ell | Legal, Written | human-annotated | found |



#### GujaratiNewsClassification

A Gujarati dataset for 3-class classification of Gujarati news articles

**Dataset:** [`mlexplorer008/gujarati_news_classification`](https://huggingface.co/datasets/mlexplorer008/gujarati_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-gujarati)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj | News, Written | derived | found |



#### GujaratiNewsClassification.v2

A Gujarati dataset for 3-class classification of Gujarati news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/gujarati_news`](https://huggingface.co/datasets/mteb/gujarati_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-gujarati)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj | News, Written | derived | found |



#### HateSpeechPortugueseClassification

HateSpeechPortugueseClassification is a dataset of Portuguese tweets categorized with their sentiment (2 classes).

**Dataset:** [`hate-speech-portuguese/hate_speech_portuguese`](https://huggingface.co/datasets/hate-speech-portuguese/hate_speech_portuguese) • **License:** not specified • [Learn more →](https://aclanthology.org/W19-3510)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | por | Social, Written | expert-annotated | found |



#### HeadlineClassification

Headline rubric classification based on the paraphraser plus dataset.

**Dataset:** [`ai-forever/headline-classification`](https://huggingface.co/datasets/ai-forever/headline-classification) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | News, Written | derived | found |



#### HeadlineClassification.v2

Headline rubric classification based on the paraphraser plus dataset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/headline`](https://huggingface.co/datasets/mteb/headline) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | News, Written | derived | found |



#### HebrewSentimentAnalysis

HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.

**Dataset:** [`omilab/hebrew_sentiment`](https://huggingface.co/datasets/omilab/hebrew_sentiment) • **License:** mit • [Learn more →](https://huggingface.co/datasets/hebrew_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | heb | Reviews, Written | expert-annotated | found |



#### HebrewSentimentAnalysis.v2

HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/hebrew_sentiment_analysis`](https://huggingface.co/datasets/mteb/hebrew_sentiment_analysis) • **License:** mit • [Learn more →](https://huggingface.co/datasets/hebrew_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | heb | Reviews, Written | expert-annotated | found |



#### HinDialectClassification

HinDialect: 26 Hindi-related languages and dialects of the Indic Continuum in North India

**Dataset:** [`mlexplorer008/hin_dialect_classification`](https://huggingface.co/datasets/mlexplorer008/hin_dialect_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4839)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | anp, awa, ben, bgc, bhb, ... (21) | Social, Spoken, Written | expert-annotated | found |



#### HindiDiscourseClassification

A Hindi Discourse dataset in Hindi with values for coherence.

**Dataset:** [`midas/hindi_discourse`](https://huggingface.co/datasets/midas/hindi_discourse) • **License:** mit • [Learn more →](https://aclanthology.org/2020.lrec-1.149/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hin | Fiction, Social, Written | expert-annotated | found |



#### HindiDiscourseClassification.v2

A Hindi Discourse dataset in Hindi with values for coherence.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/hindi_discourse`](https://huggingface.co/datasets/mteb/hindi_discourse) • **License:** mit • [Learn more →](https://aclanthology.org/2020.lrec-1.149/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hin | Fiction, Social, Written | expert-annotated | found |



#### HotelReviewSentimentClassification

HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.

**Dataset:** [`mteb/HotelReviewSentimentClassification`](https://huggingface.co/datasets/mteb/HotelReviewSentimentClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### HotelReviewSentimentClassification.v2

HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/HotelReviewSentimentClassification`](https://huggingface.co/datasets/mteb/HotelReviewSentimentClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### IFlyTek

Long Text classification for the description of Apps

**Dataset:** [`C-MTEB/IFlyTek-classification`](https://huggingface.co/datasets/C-MTEB/IFlyTek-classification) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### IFlyTek.v2

Long Text classification for the description of Apps
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/i_fly_tek`](https://huggingface.co/datasets/mteb/i_fly_tek) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### ImdbClassification

Large Movie Review Dataset

**Dataset:** [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) • **License:** not specified • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### ImdbClassification.v2

Large Movie Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) • **License:** not specified • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### ImdbVNClassification

A translated dataset of large movie reviews annotated for sentiment classification.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/imdb-vn`](https://huggingface.co/datasets/GreenNode/imdb-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### InappropriatenessClassification

Inappropriateness identification in the form of binary classification

**Dataset:** [`ai-forever/inappropriateness-classification`](https://huggingface.co/datasets/ai-forever/inappropriateness-classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |



#### InappropriatenessClassification.v2

Inappropriateness identification in the form of binary classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/inappropriateness`](https://huggingface.co/datasets/mteb/inappropriateness) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |



#### InappropriatenessClassificationv2

Inappropriateness identification in the form of binary classification

**Dataset:** [`mteb/InappropriatenessClassificationv2`](https://huggingface.co/datasets/mteb/InappropriatenessClassificationv2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | Social, Web, Written | human-annotated | found |



#### IndicLangClassification

A language identification test set for native-script as well as Romanized text which spans 22 Indic languages.

**Dataset:** [`ai4bharat/Bhasha-Abhijnaanam`](https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2305.15814)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | asm, ben, brx, doi, gom, ... (22) | Non-fiction, Web, Written | expert-annotated | created |



#### IndicNLPNewsClassification

A News classification dataset in multiple Indian regional languages.

**Dataset:** [`Sakshamrzt/IndicNLP-Multilingual`](https://huggingface.co/datasets/Sakshamrzt/IndicNLP-Multilingual) • **License:** cc-by-nc-4.0 • [Learn more →](https://github.com/AI4Bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj, kan, mal, mar, ori, ... (8) | News, Written | expert-annotated | found |



#### IndicSentimentClassification

A new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages.

**Dataset:** [`mteb/IndicSentiment`](https://huggingface.co/datasets/mteb/IndicSentiment) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | asm, ben, brx, guj, hin, ... (13) | Reviews, Written | human-annotated | machine-translated and verified |



#### IndonesianIdClickbaitClassification

The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.

**Dataset:** [`manandey/id_clickbait`](https://huggingface.co/datasets/manandey/id_clickbait) • **License:** cc-by-4.0 • [Learn more →](http://www.sciencedirect.com/science/article/pii/S2352340920311252)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | News, Written | expert-annotated | found |



#### IndonesianIdClickbaitClassification.v2

The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/indonesian_id_clickbait`](https://huggingface.co/datasets/mteb/indonesian_id_clickbait) • **License:** cc-by-4.0 • [Learn more →](http://www.sciencedirect.com/science/article/pii/S2352340920311252)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | News, Written | expert-annotated | found |



#### IndonesianMongabayConservationClassification

Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)

**Dataset:** [`Datasaur/mongabay-experiment`](https://huggingface.co/datasets/Datasaur/mongabay-experiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.sealp-1.4/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | Web, Written | derived | found |



#### IndonesianMongabayConservationClassification.v2

Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/indonesian_mongabay_conservation`](https://huggingface.co/datasets/mteb/indonesian_mongabay_conservation) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.sealp-1.4/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | Web, Written | derived | found |



#### InsurancePolicyInterpretationLegalBenchClassification

Given an insurance claim and policy, determine whether the claim is covered by the policy.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### InternationalCitizenshipQuestionsLegalBenchClassification

Answer questions about citizenship law from across the world. Dataset was made using the GLOBALCIT citizenship law dataset, by constructing questions about citizenship law as Yes or No questions.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### IsiZuluNewsClassification

isiZulu News Classification Dataset

**Dataset:** [`isaacchung/isizulu-news`](https://huggingface.co/datasets/isaacchung/isizulu-news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | zul | News, Written | human-annotated | found |



#### IsiZuluNewsClassification.v2

isiZulu News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/isi_zulu_news`](https://huggingface.co/datasets/mteb/isi_zulu_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | zul | News, Written | human-annotated | found |



#### ItaCaseholdClassification

An Italian Dataset consisting of 1101 pairs of judgments and their official holdings between the years 2019 and 2022 from the archives of Italian Administrative Justice categorized with 64 subjects.

**Dataset:** [`itacasehold/itacasehold`](https://huggingface.co/datasets/itacasehold/itacasehold) • **License:** apache-2.0 • [Learn more →](https://doi.org/10.1145/3594536.3595177)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Government, Legal, Written | expert-annotated | found |



#### Itacola

An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.

**Dataset:** [`gsarti/itacola`](https://huggingface.co/datasets/gsarti/itacola) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.findings-emnlp.250/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Non-fiction, Spoken, Written | expert-annotated | found |



#### Itacola.v2

An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/italian_linguistic_acceptability`](https://huggingface.co/datasets/mteb/italian_linguistic_acceptability) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.findings-emnlp.250/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Non-fiction, Spoken, Written | expert-annotated | found |



#### JCrewBlockerLegalBenchClassification

The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### JCrewBlockerLegalBenchClassification.v2

The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/j_crew_blocker_legal_bench`](https://huggingface.co/datasets/mteb/j_crew_blocker_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### JDReview

review for iphone

**Dataset:** [`C-MTEB/JDReview-classification`](https://huggingface.co/datasets/C-MTEB/JDReview-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### JDReview.v2

review for iphone
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/jd_review`](https://huggingface.co/datasets/mteb/jd_review) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### JapaneseSentimentClassification

Japanese sentiment classification dataset with binary
                       (positive vs negative sentiment) labels. This version reverts
                       the morphological analysis from the original multilingual dataset
                       to restore natural Japanese text without artificial spaces.
                     

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Reviews, Written | derived | found |



#### JavaneseIMDBClassification

Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.

**Dataset:** [`w11wo/imdb-javanese`](https://huggingface.co/datasets/w11wo/imdb-javanese) • **License:** mit • [Learn more →](https://github.com/w11wo/nlp-datasets#javanese-imdb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jav | Reviews, Written | human-annotated | found |



#### JavaneseIMDBClassification.v2

Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/javanese_imdb`](https://huggingface.co/datasets/mteb/javanese_imdb) • **License:** mit • [Learn more →](https://github.com/w11wo/nlp-datasets#javanese-imdb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jav | Reviews, Written | human-annotated | found |



#### KLUE-TC

Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | News, Written | human-annotated | found |



#### KLUE-TC.v2

Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/klue_tc`](https://huggingface.co/datasets/mteb/klue_tc) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | News, Written | human-annotated | found |



#### KannadaNewsClassification

The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.

**Dataset:** [`Akash190104/kannada_news_classification`](https://huggingface.co/datasets/Akash190104/kannada_news_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-kannada)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kan | News, Written | derived | found |



#### KannadaNewsClassification.v2

The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kannada_news`](https://huggingface.co/datasets/mteb/kannada_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-kannada)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kan | News, Written | derived | found |



#### KinopoiskClassification

Kinopoisk review sentiment classification

**Dataset:** [`ai-forever/kinopoisk-sentiment-classification`](https://huggingface.co/datasets/ai-forever/kinopoisk-sentiment-classification) • **License:** not specified • [Learn more →](https://www.dialog-21.ru/media/1226/blinovpd.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### KorFin

The KorFin-ASC is an extension of KorFin-ABSA, which is a financial sentiment analysis dataset including 8818 samples with (aspect, polarity) pairs annotated. The samples were collected from KLUE-TC and analyst reports from Naver Finance.

**Dataset:** [`amphora/korfin-asc`](https://huggingface.co/datasets/amphora/korfin-asc) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/amphora/korfin-asc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Financial, News, Written | expert-annotated | found |



#### KorHateClassification

The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.
        

**Dataset:** [`inmoonlight/kor_hate`](https://huggingface.co/datasets/inmoonlight/kor_hate) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-hatespeech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KorHateClassification.v2

The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kor_hate`](https://huggingface.co/datasets/mteb/kor_hate) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-hatespeech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KorSarcasmClassification


        The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original
        meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These
        tweets were gathered by querying for: irony sarcastic, and
        sarcasm.
        The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm
        and variants of it were used to return tweets. It was preprocessed by removing the keyword
        hashtag, urls and mentions of the user to preserve anonymity.
        

**Dataset:** [`SpellOnYou/kor_sarcasm`](https://huggingface.co/datasets/SpellOnYou/kor_sarcasm) • **License:** mit • [Learn more →](https://github.com/SpellOnYou/korean-sarcasm)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KorSarcasmClassification.v2


        The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original
        meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These
        tweets were gathered by querying for: irony sarcastic, and
        sarcasm.
        The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm
        and variants of it were used to return tweets. It was preprocessed by removing the keyword
        hashtag, urls and mentions of the user to preserve anonymity.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kor_sarcasm`](https://huggingface.co/datasets/mteb/kor_sarcasm) • **License:** mit • [Learn more →](https://github.com/SpellOnYou/korean-sarcasm)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KurdishSentimentClassification

Kurdish Sentiment Dataset

**Dataset:** [`asparius/Kurdish-Sentiment`](https://huggingface.co/datasets/asparius/Kurdish-Sentiment) • **License:** cc-by-4.0 • [Learn more →](https://link.springer.com/article/10.1007/s10579-023-09716-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kur | Web, Written | derived | found |



#### KurdishSentimentClassification.v2

Kurdish Sentiment Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kurdish_sentiment`](https://huggingface.co/datasets/mteb/kurdish_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://link.springer.com/article/10.1007/s10579-023-09716-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kur | Web, Written | derived | found |



#### LanguageClassification

A language identification dataset for 20 languages.

**Dataset:** [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/papluca/language-identification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, bul, cmn, deu, ell, ... (20) | Fiction, Government, Non-fiction, Reviews, Web, ... (6) | derived | found |



#### LccSentimentClassification

The leipzig corpora collection, annotated for sentiment

**Dataset:** [`DDSC/lcc`](https://huggingface.co/datasets/DDSC/lcc) • **License:** cc-by-4.0 • [Learn more →](https://github.com/fnielsen/lcc-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | News, Web, Written | expert-annotated | found |



#### LearnedHandsBenefitsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal post discusses public benefits and social services that people can get from the government, like for food, disability, old age, housing, medical help, unemployment, child care, or other social needs.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsBusinessLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal question discusses issues faced by people who run small businesses or nonprofits, including around incorporation, licenses, taxes, regulations, and other concerns. It also includes options when there are disasters, bankruptcies, or other problems.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsConsumerLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsCourtsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses the logistics of how a person can interact with a lawyer or the court system. It applies to situations about procedure, rules, how to file lawsuits, how to hire lawyers, how to represent oneself, and other practical matters about dealing with these systems.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsCrimeLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues in the criminal system including when people are charged with crimes, go to a criminal trial, go to prison, or are a victim of a crime.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsDivorceLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues around filing for divorce, separation, or annulment, getting spousal support, splitting money and property, and following the court processes.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsDomesticViolenceLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses dealing with domestic violence and abuse, including getting protective orders, enforcing them, understanding abuse, reporting abuse, and getting resources and status if there is abuse.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsEducationLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues around school, including accommodations for special needs, discrimination, student debt, discipline, and other issues in education.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsEmploymentLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsEstatesLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses planning for end-of-life, possible incapacitation, and other special circumstances that would prevent a person from making decisions about their own well-being, finances, and property. This includes issues around wills, powers of attorney, advance directives, trusts, guardianships, conservatorships, and other estate issues that people and families deal with.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsFamilyLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsHealthLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues with accessing health services, paying for medical care, getting public benefits for health care, protecting one's rights in medical settings, and other issues related to health.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsHousingLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsImmigrationLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses visas, asylum, green cards, citizenship, migrant work and benefits, and other issues faced by people who are not full citizens in the US.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsTortsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal question discusses problems that one person has with another person (or animal), like when there is a car accident, a dog bite, bullying or possible harassment, or neighbors treating each other badly.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsTrafficLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal post discusses problems with traffic and parking tickets, fees, driver's licenses, and other issues experienced with the traffic system. It also concerns issues with car accidents and injuries, cars' quality, repairs, purchases, and other contracts.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LegalReasoningCausalityLegalBenchClassification

Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LegalReasoningCausalityLegalBenchClassification.v2

Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/legal_reasoning_causality_legal_bench`](https://huggingface.co/datasets/mteb/legal_reasoning_causality_legal_bench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### MAUDLegalBenchClassification

This task was constructed from the MAUD dataset, which consists of over 47,000 labels across 152 merger agreements annotated to identify 92 questions in each agreement used by the 2021 American Bar Association (ABA) Public Target Deal Points Study. Each dataset is formatted as a series of multiple-choice questions, where given a segment of the merger agreement and a Deal Point question, the model is to choose the answer that best characterizes the agreement as response.

        This is a combination of all 34 of the MAUD Legal Bench datasets:
        1. MAUD Ability To Consummate Concept Is Subject To MAE Carveouts: Given an excerpt from a merger agreement and the task is to answer: is the “ability to consummate” concept subject to Material Adverse Effect (MAE) carveouts? amongst the multiple choice options.
        2. MAUD Accuracy Of Fundamental Target RWS Bringdown Standard: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        3. MAUD Accuracy Of Target Capitalization RW Outstanding Shares Bringdown Standard Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        4. MAUD Accuracy Of Target General RW Bringdown Timing Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        5. MAUD Additional Matching Rights Period For Modifications Cor: Given an excerpt from a merger agreement and the task is to answer: how long is the additional matching rights period for modifications in case the board changes its recommendation, amongst the multiple choice options.
        6. MAUD Application Of Buyer Consent Requirement Negative Interim Covenant: Given an excerpt from a merger agreement and the task is to answer: what negative covenants does the requirement of Buyer consent apply to, amongst the multiple choice options.
        7. MAUD Buyer Consent Requirement Ordinary Course: Given an excerpt from a merger agreement and the task is to answer: in case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent, amongst the multiple choice options.
        8. MAUD Change In Law Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        9. MAUD Changes In GAAP Or Other Accounting Principles Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        10. MAUD COR Permitted In Response To Intervening Event: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted in response to an intervening event, amongst the multiple choice options.
        11. MAUD COR Permitted With Board Fiduciary Determination Only: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations, amongst the multiple choice options.
        12. MAUD COR Standard Intervening Event: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in response to an intervening event, amongst the multiple choice options.
        13. MAUD COR Standard Superior Offer: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in connection with a superior offer, amongst the multiple choice options.
        14. MAUD Definition Contains Knowledge Requirement Answer: Given an excerpt from a merger agreement and the task is to answer: what is the knowledge requirement in the definition of “Intervening Event”, amongst the multiple choice options.
        15. MAUD Definition Includes Asset Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of asset deals, amongst the multiple choice options.
        16. MAUD Definition Includes Stock Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of stock deals, amongst the multiple choice options.
        17. MAUD Fiduciary Exception Board Determination Standard: Given an excerpt from a merger agreement and the task is to answer: under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision, amongst the multiple choice options.
        18. MAUD Fiduciary Exception Board Determination Trigger No Shop: Given an excerpt from a merger agreement and the task is to answer: what type of offer could the Board take actions on notwithstanding the no-shop provision, amongst the multiple choice options.
        19. MAUD Financial Point Of View Is The Sole Consideration: Given an excerpt from a merger agreement and the task is to answer: is “financial point of view” the sole consideration when determining whether an offer is superior, amongst the multiple choice options.
        20. MAUD FLS MAE Standard: Given an excerpt from a merger agreement and the task is to answer: what is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE), amongst the multiple choice options.
        21. MAUD General Economic and Financial Conditions Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        22. MAUD Includes Consistent With Past Practice: Given an excerpt from a merger agreement and the task is to answer: does the wording of the Efforts Covenant clause include “consistent with past practice”, amongst the multiple choice options.
        23. MAUD Initial Matching Rights Period COR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in case the board changes its recommendation, amongst the multiple choice options.
        24. MAUD Initial Matching Rights Period FTR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR), amongst the multiple choice options.
        25. MAUDInterveningEventRequiredToOccurAfterSigningAnswer: Given an excerpt from a merger agreement and the task is to answer: is an “Intervening Event” required to occur after signing, amongst the multiple choice options.
        26. MAUD Knowledge Definition: Given an excerpt from a merger agreement and the task is to answer: what counts as Knowledge, amongst the multiple choice options.
        27. MAUDLiabilityStandardForNoShopBreachByTargetNonDORepresentatives: Given an excerpt from a merger agreement and the task is to answer:  what is the liability standard for no-shop breach by Target Non-D&O Representatives, amongst the multiple choice options.
        28. MAUD Ordinary Course Efforts Standard: Given an excerpt from a merger agreement and the task is to answer: what is the efforts standard, amongst the multiple choice options.
        29. MAUD Pandemic Or Other Public Health Event Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        30. MAUD Pandemic Or Other Public Health Event Specific Reference To Pandemic Related Governmental Responses Or Measures: Given an excerpt from a merger agreement and the task is to answer: is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE), amongst the multiple choice options.
        31. MAUD Relational Language MAE Applies To: Given an excerpt from a merger agreement and the task is to answer: what carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?, amongst the multiple choice options.
        32. MAUD Specific Performance: Given an excerpt from a merger agreement and the task is to answer: what is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach, amongst the multiple choice options.
        33. MAUD Tail Period Length: Given an excerpt from a merger agreement and the task is to answer: how long is the Tail Period, amongst the multiple choice options.
        34. MAUD Type Of Consideration: Given an excerpt from a merger agreement and the task is to answer: what type of consideration is specified in this agreement, amongst the multiple choice options.
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### MAUDLegalBenchClassification.v2

This task was constructed from the MAUD dataset, which consists of over 47,000 labels across 152 merger agreements annotated to identify 92 questions in each agreement used by the 2021 American Bar Association (ABA) Public Target Deal Points Study. Each dataset is formatted as a series of multiple-choice questions, where given a segment of the merger agreement and a Deal Point question, the model is to choose the answer that best characterizes the agreement as response.

        This is a combination of all 34 of the MAUD Legal Bench datasets:
        1. MAUD Ability To Consummate Concept Is Subject To MAE Carveouts: Given an excerpt from a merger agreement and the task is to answer: is the “ability to consummate” concept subject to Material Adverse Effect (MAE) carveouts? amongst the multiple choice options.
        2. MAUD Accuracy Of Fundamental Target RWS Bringdown Standard: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        3. MAUD Accuracy Of Target Capitalization RW Outstanding Shares Bringdown Standard Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        4. MAUD Accuracy Of Target General RW Bringdown Timing Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        5. MAUD Additional Matching Rights Period For Modifications Cor: Given an excerpt from a merger agreement and the task is to answer: how long is the additional matching rights period for modifications in case the board changes its recommendation, amongst the multiple choice options.
        6. MAUD Application Of Buyer Consent Requirement Negative Interim Covenant: Given an excerpt from a merger agreement and the task is to answer: what negative covenants does the requirement of Buyer consent apply to, amongst the multiple choice options.
        7. MAUD Buyer Consent Requirement Ordinary Course: Given an excerpt from a merger agreement and the task is to answer: in case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent, amongst the multiple choice options.
        8. MAUD Change In Law Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        9. MAUD Changes In GAAP Or Other Accounting Principles Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        10. MAUD COR Permitted In Response To Intervening Event: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted in response to an intervening event, amongst the multiple choice options.
        11. MAUD COR Permitted With Board Fiduciary Determination Only: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations, amongst the multiple choice options.
        12. MAUD COR Standard Intervening Event: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in response to an intervening event, amongst the multiple choice options.
        13. MAUD COR Standard Superior Offer: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in connection with a superior offer, amongst the multiple choice options.
        14. MAUD Definition Contains Knowledge Requirement Answer: Given an excerpt from a merger agreement and the task is to answer: what is the knowledge requirement in the definition of “Intervening Event”, amongst the multiple choice options.
        15. MAUD Definition Includes Asset Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of asset deals, amongst the multiple choice options.
        16. MAUD Definition Includes Stock Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of stock deals, amongst the multiple choice options.
        17. MAUD Fiduciary Exception Board Determination Standard: Given an excerpt from a merger agreement and the task is to answer: under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision, amongst the multiple choice options.
        18. MAUD Fiduciary Exception Board Determination Trigger No Shop: Given an excerpt from a merger agreement and the task is to answer: what type of offer could the Board take actions on notwithstanding the no-shop provision, amongst the multiple choice options.
        19. MAUD Financial Point Of View Is The Sole Consideration: Given an excerpt from a merger agreement and the task is to answer: is “financial point of view” the sole consideration when determining whether an offer is superior, amongst the multiple choice options.
        20. MAUD FLS MAE Standard: Given an excerpt from a merger agreement and the task is to answer: what is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE), amongst the multiple choice options.
        21. MAUD General Economic and Financial Conditions Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        22. MAUD Includes Consistent With Past Practice: Given an excerpt from a merger agreement and the task is to answer: does the wording of the Efforts Covenant clause include “consistent with past practice”, amongst the multiple choice options.
        23. MAUD Initial Matching Rights Period COR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in case the board changes its recommendation, amongst the multiple choice options.
        24. MAUD Initial Matching Rights Period FTR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR), amongst the multiple choice options.
        25. MAUDInterveningEventRequiredToOccurAfterSigningAnswer: Given an excerpt from a merger agreement and the task is to answer: is an “Intervening Event” required to occur after signing, amongst the multiple choice options.
        26. MAUD Knowledge Definition: Given an excerpt from a merger agreement and the task is to answer: what counts as Knowledge, amongst the multiple choice options.
        27. MAUDLiabilityStandardForNoShopBreachByTargetNonDORepresentatives: Given an excerpt from a merger agreement and the task is to answer:  what is the liability standard for no-shop breach by Target Non-D&O Representatives, amongst the multiple choice options.
        28. MAUD Ordinary Course Efforts Standard: Given an excerpt from a merger agreement and the task is to answer: what is the efforts standard, amongst the multiple choice options.
        29. MAUD Pandemic Or Other Public Health Event Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        30. MAUD Pandemic Or Other Public Health Event Specific Reference To Pandemic Related Governmental Responses Or Measures: Given an excerpt from a merger agreement and the task is to answer: is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE), amongst the multiple choice options.
        31. MAUD Relational Language MAE Applies To: Given an excerpt from a merger agreement and the task is to answer: what carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?, amongst the multiple choice options.
        32. MAUD Specific Performance: Given an excerpt from a merger agreement and the task is to answer: what is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach, amongst the multiple choice options.
        33. MAUD Tail Period Length: Given an excerpt from a merger agreement and the task is to answer: how long is the Tail Period, amongst the multiple choice options.
        34. MAUD Type Of Consideration: Given an excerpt from a merger agreement and the task is to answer: what type of consideration is specified in this agreement, amongst the multiple choice options.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/maud_legal_bench`](https://huggingface.co/datasets/mteb/maud_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### MTOPDomainClassification

MTOP: Multilingual Task-Oriented Semantic Parsing

**Dataset:** [`mteb/mtop_domain`](https://huggingface.co/datasets/mteb/mtop_domain) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, fra, hin, spa, ... (6) | Spoken, Spoken | human-annotated | created |



#### MTOPDomainVNClassification

A translated dataset from MTOP: Multilingual Task-Oriented Semantic Parsing
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/mtop-domain-vn`](https://huggingface.co/datasets/GreenNode/mtop-domain-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken, Spoken | derived | machine-translated and LM verified |



#### MTOPIntentClassification

MTOP: Multilingual Task-Oriented Semantic Parsing

**Dataset:** [`mteb/mtop_intent`](https://huggingface.co/datasets/mteb/mtop_intent) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, fra, hin, spa, ... (6) | Spoken, Spoken | human-annotated | created |



#### MTOPIntentVNClassification

A translated dataset from MTOP: Multilingual Task-Oriented Semantic Parsing
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/mtop-intent-vn`](https://huggingface.co/datasets/GreenNode/mtop-intent-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken, Spoken | derived | machine-translated and LM verified |



#### MacedonianTweetSentimentClassification

An Macedonian dataset for tweet sentiment classification.

**Dataset:** [`isaacchung/macedonian-tweet-sentiment-classification`](https://huggingface.co/datasets/isaacchung/macedonian-tweet-sentiment-classification) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/R15-1034/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mkd | Social, Written | human-annotated | found |



#### MacedonianTweetSentimentClassification.v2

An Macedonian dataset for tweet sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/macedonian_tweet_sentiment`](https://huggingface.co/datasets/mteb/macedonian_tweet_sentiment) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/R15-1034/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mkd | Social, Written | human-annotated | found |



#### MalayalamNewsClassification

A Malayalam dataset for 3-class classification of Malayalam news articles

**Dataset:** [`mlexplorer008/malayalam_news_classification`](https://huggingface.co/datasets/mlexplorer008/malayalam_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-malyalam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mal | News, Written | derived | found |



#### MalayalamNewsClassification.v2

A Malayalam dataset for 3-class classification of Malayalam news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/malayalam_news`](https://huggingface.co/datasets/mteb/malayalam_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-malyalam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mal | News, Written | derived | found |



#### MarathiNewsClassification

A Marathi dataset for 3-class classification of Marathi news articles

**Dataset:** [`mlexplorer008/marathi_news_classification`](https://huggingface.co/datasets/mlexplorer008/marathi_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-marathi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | mar | News, Written | derived | found |



#### MarathiNewsClassification.v2

A Marathi dataset for 3-class classification of Marathi news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/marathi_news`](https://huggingface.co/datasets/mteb/marathi_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-marathi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | mar | News, Written | derived | found |



#### MasakhaNEWSClassification

MasakhaNEWS is the largest publicly available dataset for news topic classification in 16 languages widely spoken in Africa. The train/validation/test sets are available for all the 16 languages.

**Dataset:** [`mteb/masakhanews`](https://huggingface.co/datasets/mteb/masakhanews) • **License:** cc-by-nc-4.0 • [Learn more →](https://arxiv.org/abs/2304.09972)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, eng, fra, hau, ibo, ... (16) | News, Written | expert-annotated | found |



#### MassiveIntentClassification

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

**Dataset:** [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.08582)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, amh, ara, aze, ben, ... (50) | Spoken | human-annotated | human-translated and localized |



#### MassiveIntentVNClassification

A translated dataset from MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/amazon-massive-intent-vn`](https://huggingface.co/datasets/GreenNode/amazon-massive-intent-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken | derived | machine-translated and LM verified |



#### MassiveScenarioClassification

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

**Dataset:** [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.08582)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, amh, ara, aze, ben, ... (50) | Spoken | human-annotated | human-translated and localized |



#### MassiveScenarioVNClassification

A translated dataset from MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/amazon-massive-scenario-vn`](https://huggingface.co/datasets/GreenNode/amazon-massive-scenario-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken | derived | machine-translated and LM verified |



#### Moroco

The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech

**Dataset:** [`universityofbucharest/moroco`](https://huggingface.co/datasets/universityofbucharest/moroco) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/moroco)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | News, Written | derived | found |



#### Moroco.v2

The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/moroco`](https://huggingface.co/datasets/mteb/moroco) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/moroco)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | News, Written | derived | found |



#### MovieReviewSentimentClassification

The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.

**Dataset:** [`tblard/allocine`](https://huggingface.co/datasets/tblard/allocine) • **License:** mit • [Learn more →](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### MovieReviewSentimentClassification.v2

The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/movie_review_sentiment`](https://huggingface.co/datasets/mteb/movie_review_sentiment) • **License:** mit • [Learn more →](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### MultiHateClassification

Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate, and 11 languages.
                     

**Dataset:** [`mteb/multi-hatecheck`](https://huggingface.co/datasets/mteb/multi-hatecheck) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2022.woah-1.15/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, cmn, deu, eng, fra, ... (11) | Constructed, Written | expert-annotated | created |



#### MultilingualSentiment

A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative

**Dataset:** [`C-MTEB/MultilingualSentiment-classification`](https://huggingface.co/datasets/C-MTEB/MultilingualSentiment-classification) • **License:** not specified • [Learn more →](https://github.com/tyqiangz/multilingual-sentiment-datasets)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### MultilingualSentiment.v2

A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/multilingual_sentiment`](https://huggingface.co/datasets/mteb/multilingual_sentiment) • **License:** not specified • [Learn more →](https://github.com/tyqiangz/multilingual-sentiment-datasets)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### MultilingualSentimentClassification

Sentiment classification dataset with binary
                       (positive vs negative sentiment) labels. Includes 30 languages and dialects.
                     

**Dataset:** [`mteb/multilingual-sentiment-classification`](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, bam, bul, cmn, cym, ... (29) | Reviews, Written | derived | found |



#### MyanmarNews

The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.

**Dataset:** [`mteb/MyanmarNews`](https://huggingface.co/datasets/mteb/MyanmarNews) • **License:** gpl-3.0 • [Learn more →](https://huggingface.co/datasets/myanmar_news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mya | News, Written | derived | found |



#### MyanmarNews.v2

The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/myanmar_news`](https://huggingface.co/datasets/mteb/myanmar_news) • **License:** gpl-3.0 • [Learn more →](https://huggingface.co/datasets/myanmar_news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mya | News, Written | derived | found |



#### NLPTwitterAnalysisClassification

Twitter Analysis Classification

**Dataset:** [`hamedhf/nlp_twitter_analysis`](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Social | derived | found |



#### NLPTwitterAnalysisClassification.v2

Twitter Analysis Classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/nlp_twitter_analysis`](https://huggingface.co/datasets/mteb/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Social | derived | found |



#### NYSJudicialEthicsLegalBenchClassification

Answer questions on judicial ethics from the New York State Unified Court System Advisory Committee.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** mit • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### NaijaSenti

NaijaSenti is the first large-scale human-annotated Twitter sentiment dataset for the four most widely spoken languages in Nigeria — Hausa, Igbo, Nigerian-Pidgin, and Yorùbá — consisting of around 30,000 annotated tweets per language, including a significant fraction of code-mixed tweets.

**Dataset:** [`HausaNLP/NaijaSenti-Twitter`](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter) • **License:** cc-by-4.0 • [Learn more →](https://github.com/hausanlp/NaijaSenti)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hau, ibo, pcm, yor | Social, Written | expert-annotated | found |



#### NepaliNewsClassification

A Nepali dataset for 7500 news articles 

**Dataset:** [`bpHigh/iNLTK_Nepali_News_Dataset`](https://huggingface.co/datasets/bpHigh/iNLTK_Nepali_News_Dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-nepali)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nep | News, Written | derived | found |



#### NepaliNewsClassification.v2

A Nepali dataset for 7500 news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/nepali_news`](https://huggingface.co/datasets/mteb/nepali_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-nepali)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nep | News, Written | derived | found |



#### NewsClassification

Large News Classification Dataset

**Dataset:** [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Written | expert-annotated | found |



#### NewsClassification.v2

Large News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/news`](https://huggingface.co/datasets/mteb/news) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Written | expert-annotated | found |



#### NoRecClassification

A Norwegian dataset for sentiment classification on review

**Dataset:** [`mteb/norec_classification`](https://huggingface.co/datasets/mteb/norec_classification) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/L18-1661/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Reviews, Written | derived | found |



#### NoRecClassification.v2

A Norwegian dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/no_rec`](https://huggingface.co/datasets/mteb/no_rec) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/L18-1661/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Reviews, Written | derived | found |



#### NordicLangClassification

A dataset for Nordic language identification.

**Dataset:** [`strombergnlp/nordic_langid`](https://huggingface.co/datasets/strombergnlp/nordic_langid) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2021.vardial-1.8/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, fao, isl, nno, nob, ... (6) | Encyclopaedic | derived | found |



#### NorwegianParliamentClassification

Norwegian parliament speeches annotated for sentiment

**Dataset:** [`NbAiLab/norwegian_parliament`](https://huggingface.co/datasets/NbAiLab/norwegian_parliament) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NbAiLab/norwegian_parliament)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Government, Spoken | derived | found |



#### NorwegianParliamentClassification.v2

Norwegian parliament speeches annotated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/norwegian_parliament`](https://huggingface.co/datasets/mteb/norwegian_parliament) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NbAiLab/norwegian_parliament)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Government, Spoken | derived | found |



#### NusaParagraphEmotionClassification

NusaParagraphEmotionClassification is a multi-class emotion classification on 10 Indonesian languages from the NusaParagraph dataset.

**Dataset:** [`gentaiscool/nusaparagraph_emot`](https://huggingface.co/datasets/gentaiscool/nusaparagraph_emot) • **License:** apache-2.0 • [Learn more →](https://github.com/IndoNLP/nusa-writes)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | bbc, bew, bug, jav, mad, ... (10) | Fiction, Non-fiction, Written | human-annotated | found |



#### NusaParagraphTopicClassification

NusaParagraphTopicClassification is a multi-class topic classification on 10 Indonesian languages.

**Dataset:** [`gentaiscool/nusaparagraph_topic`](https://huggingface.co/datasets/gentaiscool/nusaparagraph_topic) • **License:** apache-2.0 • [Learn more →](https://github.com/IndoNLP/nusa-writes)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | bbc, bew, bug, jav, mad, ... (10) | Fiction, Non-fiction, Written | human-annotated | found |



#### NusaX-senti

NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak. NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English.

**Dataset:** [`indonlp/NusaX-senti`](https://huggingface.co/datasets/indonlp/NusaX-senti) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2205.15960)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ace, ban, bbc, bjn, bug, ... (12) | Constructed, Reviews, Social, Web, Written | expert-annotated | found |



#### OPP115DataRetentionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how long user information is stored.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DataSecurityLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how user information is protected.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DataSecurityLegalBenchClassification.v2

Given a clause from a privacy policy, classify if the clause describes how user information is protected.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_data_security_legal_bench`](https://huggingface.co/datasets/mteb/opp115_data_security_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DoNotTrackLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DoNotTrackLegalBenchClassification.v2

Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_do_not_track_legal_bench`](https://huggingface.co/datasets/mteb/opp115_do_not_track_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115FirstPartyCollectionUseLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how and why a service provider collects user information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115InternationalAndSpecificAudiencesLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describe practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115PolicyChangeLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how users will be informed about changes to the privacy policy.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115ThirdPartySharingCollectionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describe how user information may be shared with or collected by third parties.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115UserAccessEditAndDeletionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how users may access, edit, or delete their information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115UserChoiceControlLegalBenchClassification

Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115UserChoiceControlLegalBenchClassification.v2

Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_user_choice_control_legal_bench`](https://huggingface.co/datasets/mteb/opp115_user_choice_control_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OdiaNewsClassification

A Odia dataset for 3-class classification of Odia news articles

**Dataset:** [`mlexplorer008/odia_news_classification`](https://huggingface.co/datasets/mlexplorer008/odia_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-odia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ory | News, Written | derived | found |



#### OdiaNewsClassification.v2

A Odia dataset for 3-class classification of Odia news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/odia_news`](https://huggingface.co/datasets/mteb/odia_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-odia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ory | News, Written | derived | found |



#### OnlineShopping

Sentiment Analysis of User Reviews on Online Shopping Websites

**Dataset:** [`C-MTEB/OnlineShopping-classification`](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### OnlineStoreReviewSentimentClassification

This dataset contains Arabic reviews of products from the SHEIN online store.

**Dataset:** [`Ruqiya/Arabic_Reviews_of_SHEIN`](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### OnlineStoreReviewSentimentClassification.v2

This dataset contains Arabic reviews of products from the SHEIN online store.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/online_store_review_sentiment`](https://huggingface.co/datasets/mteb/online_store_review_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### OralArgumentQuestionPurposeLegalBenchClassification

This task classifies questions asked by Supreme Court justices at oral argument into seven categories:
        1. Background - questions seeking factual or procedural information that is missing or not clear in the briefing
        2. Clarification - questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for
        3. Implications - questions about the limits of a rule or its implications for future cases
        4. Support - questions offering support for the advocate’s position
        5. Criticism - questions criticizing an advocate’s position
        6. Communicate - question designed primarily to communicate with other justices
        7. Humor - questions designed to interject humor into the argument and relieve tension
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OralArgumentQuestionPurposeLegalBenchClassification.v2

This task classifies questions asked by Supreme Court justices at oral argument into seven categories:
        1. Background - questions seeking factual or procedural information that is missing or not clear in the briefing
        2. Clarification - questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for
        3. Implications - questions about the limits of a rule or its implications for future cases
        4. Support - questions offering support for the advocate’s position
        5. Criticism - questions criticizing an advocate’s position
        6. Communicate - question designed primarily to communicate with other justices
        7. Humor - questions designed to interject humor into the argument and relieve tension

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/oral_argument_question_purpose_legal_bench`](https://huggingface.co/datasets/mteb/oral_argument_question_purpose_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OverrulingLegalBenchClassification

This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OverrulingLegalBenchClassification.v2

This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/overruling_legal_bench`](https://huggingface.co/datasets/mteb/overruling_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### PAC

Polish Paraphrase Corpus

**Dataset:** [`laugustyniak/abusive-clauses-pl`](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2211.13112.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Legal, Written | not specified | not specified |



#### PAC.v2

Polish Paraphrase Corpus
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/pac`](https://huggingface.co/datasets/mteb/pac) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2211.13112.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Legal, Written | not specified | not specified |



#### PROALegalBenchClassification

Given a statute, determine if the text contains an explicit private right of action. Given a privacy policy clause and a description of the clause, determine if the description is correct. A private right of action (PROA) exists when a statute empowers an ordinary individual (i.e., a private person) to legally enforce their rights by bringing an action in court. In short, a PROA creates the ability for an individual to sue someone in order to recover damages or halt some offending conduct. PROAs are ubiquitous in antitrust law (in which individuals harmed by anti-competitive behavior can sue offending firms for compensation) and environmental law (in which individuals can sue entities which release hazardous substances for damages).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### PatentClassification

Classification Dataset of Patents and Abstract

**Dataset:** [`mteb/PatentClassification`](https://huggingface.co/datasets/mteb/PatentClassification) • **License:** not specified • [Learn more →](https://aclanthology.org/P19-1212.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | derived | found |



#### PatentClassification.v2

Classification Dataset of Patents and Abstract
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/patent`](https://huggingface.co/datasets/mteb/patent) • **License:** not specified • [Learn more →](https://aclanthology.org/P19-1212.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | derived | found |



#### PersianFoodSentimentClassification

Persian Food Review Dataset

**Dataset:** [`asparius/Persian-Food-Sentiment`](https://huggingface.co/datasets/asparius/Persian-Food-Sentiment) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/sa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews, Written | derived | found |



#### PersianTextEmotion

Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`SeyedAli/Persian-Text-Emotion`](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | derived | found |



#### PersianTextEmotion.v2

Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/persian_text_emotion`](https://huggingface.co/datasets/mteb/persian_text_emotion) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | derived | found |



#### PersonalJurisdictionLegalBenchClassification

Given a fact pattern describing the set of contacts between a plaintiff, defendant, and forum, determine if a court in that forum could excercise personal jurisdiction over the defendant.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### PoemSentimentClassification

Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.

**Dataset:** [`google-research-datasets/poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2011.02686)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | human-annotated | found |



#### PoemSentimentClassification.v2

Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/poem_sentiment`](https://huggingface.co/datasets/mteb/poem_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2011.02686)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | human-annotated | found |



#### PolEmo2.0-IN

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.

**Dataset:** [`PL-MTEB/polemo2_in`](https://huggingface.co/datasets/PL-MTEB/polemo2_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | derived | found |



#### PolEmo2.0-IN.v2

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.

**Dataset:** [`mteb/pol_emo2_in`](https://huggingface.co/datasets/mteb/pol_emo2_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | derived | found |



#### PolEmo2.0-OUT

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and school) reviews using models train on reviews from medicine and hotels domains.

**Dataset:** [`PL-MTEB/polemo2_out`](https://huggingface.co/datasets/PL-MTEB/polemo2_out) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | not specified | not specified |



#### PolEmo2.0-OUT.v2

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and school) reviews using models train on reviews from medicine and hotels domains.

**Dataset:** [`mteb/pol_emo2_out`](https://huggingface.co/datasets/mteb/pol_emo2_out) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | not specified | not specified |



#### PunjabiNewsClassification

A Punjabi dataset for 2-class classification of Punjabi news articles

**Dataset:** [`mlexplorer008/punjabi_news_classification`](https://huggingface.co/datasets/mlexplorer008/punjabi_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-punjabi/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pan | News, Written | derived | found |



#### RestaurantReviewSentimentClassification

Dataset of 8364 restaurant reviews from qaym.com in Arabic for sentiment analysis

**Dataset:** [`hadyelsahar/ar_res_reviews`](https://huggingface.co/datasets/hadyelsahar/ar_res_reviews) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### RestaurantReviewSentimentClassification.v2

Dataset of 8156 restaurant reviews from qaym.com in Arabic for sentiment analysis
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/restaurant_review_sentiment`](https://huggingface.co/datasets/mteb/restaurant_review_sentiment) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### RomanianReviewsSentiment

LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian

**Dataset:** [`universityofbucharest/laroseda`](https://huggingface.co/datasets/universityofbucharest/laroseda) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.04197)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | derived | found |



#### RomanianReviewsSentiment.v2

LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/romanian_reviews_sentiment`](https://huggingface.co/datasets/mteb/romanian_reviews_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.04197)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | derived | found |



#### RomanianSentimentClassification

An Romanian dataset for sentiment classification.

**Dataset:** [`dumitrescustefan/ro_sent`](https://huggingface.co/datasets/dumitrescustefan/ro_sent) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | human-annotated | found |



#### RomanianSentimentClassification.v2

An Romanian dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/romanian_sentiment`](https://huggingface.co/datasets/mteb/romanian_sentiment) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | human-annotated | found |



#### RuNLUIntentClassification

Contains natural language data for human-robot interaction in home domain which we collected and annotated for evaluating NLU Services/platforms.

**Dataset:** [`mteb/RuNLUIntentClassification`](https://huggingface.co/datasets/mteb/RuNLUIntentClassification) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/1903.05566)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | human-annotated | found |



#### RuReviewsClassification

Product review classification (3-point scale) based on RuRevies dataset

**Dataset:** [`ai-forever/ru-reviews-classification`](https://huggingface.co/datasets/ai-forever/ru-reviews-classification) • **License:** apache-2.0 • [Learn more →](https://github.com/sismetanin/rureviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### RuReviewsClassification.v2

Product review classification (3-point scale) based on RuRevies dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ru_reviews`](https://huggingface.co/datasets/mteb/ru_reviews) • **License:** apache-2.0 • [Learn more →](https://github.com/sismetanin/rureviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### RuSciBenchCoreRiscClassification

This binary classification task aims to determine whether a scientific paper
        (based on its title and abstract) belongs to the Core of the Russian Science Citation Index (RISC).
        The RISC includes a wide range of publications, but the Core RISC comprises the most cited and prestigious
        journals, dissertations, theses, monographs, and studies. The task is provided for both Russian and English
        versions of the paper's title and abstract.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchGRNTIClassification

Classification of scientific papers (title+abstract) by rubric

**Dataset:** [`ai-forever/ru-scibench-grnti-classification`](https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Academic, Written | derived | found |



#### RuSciBenchGRNTIClassification.v2

Classification of scientific papers based on the GRNTI (State Rubricator of Scientific and
        Technical Information) rubricator. GRNTI is a universal hierarchical classification of knowledge domains
        adopted in Russia and CIS countries to systematize the entire flow of scientific and technical information.
        This task uses the first level of the GRNTI hierarchy and top 28 classes by frequency.

        In this version, English language support has been added and data partitioning has been slightly modified.
        

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchOECDClassification

Classification of scientific papers (title+abstract) by rubric

**Dataset:** [`ai-forever/ru-scibench-oecd-classification`](https://huggingface.co/datasets/ai-forever/ru-scibench-oecd-classification) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Academic, Written | derived | found |



#### RuSciBenchOECDClassification.v2

Classification of scientific papers based on the OECD
        (Organization for Economic Co-operation and Development) rubricator. OECD provides
        a hierarchical 3-level system of classes for labeling scientific articles.
        This task uses the first two levels of the OECD hierarchy, top 29 classes.

        In this version, English language support has been added and data partitioning has been slightly modified.
        

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchPubTypeClassification

This task involves classifying scientific papers (based on their title and abstract)
        into different publication types. The dataset identifies the following types:
        'Article', 'Conference proceedings', 'Survey', 'Miscellanea', 'Short message', 'Review', and 'Personalia'.
        This task is available for both Russian and English versions of the paper's title and abstract.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuToxicOKMLCUPClassification

On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.

**Dataset:** [`mteb/RuToxicOKMLCUPClassification`](https://huggingface.co/datasets/mteb/RuToxicOKMLCUPClassification) • **License:** not specified • [Learn more →](https://cups.online/ru/contests/okmlcup2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### RuToxicOKMLCUPClassification.v2

On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ru_toxic_okmlcup`](https://huggingface.co/datasets/mteb/ru_toxic_okmlcup) • **License:** not specified • [Learn more →](https://cups.online/ru/contests/okmlcup2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### RuToxicOKMLCUPMultilabelClassification

On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.

**Dataset:** [`mteb/RuToxicOKMLCUPClassification`](https://huggingface.co/datasets/mteb/RuToxicOKMLCUPClassification) • **License:** not specified • [Learn more →](https://cups.online/ru/contests/okmlcup2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### SCDBPAccountabilityLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer maintains internal compliance procedures on company standards regarding human trafficking and slavery? This includes any type of internal accountability mechanism. Requiring independently of the supply to comply with laws does not qualify or asking for documentary evidence of compliance does not count either.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPAuditsLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPCertificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPTrainingLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  provides training to employees on human trafficking and slavery? Broad policies such as ongoing dialogue on mitigating risks of human trafficking and slavery or increasing managers and purchasers knowledge about health, safety and labor practices qualify as training. Providing training to contractors who failed to comply with human trafficking laws counts as training.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPVerificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer engages in verification and auditing as one practice, expresses that it may conduct an audit, or expressess that it is assessing supplier risks through a review of the US Dept. of Labor's List?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDAccountabilityLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer maintains internal accountability standards and procedures for employees or contractors failing to meet company standards regarding slavery and trafficking?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDAuditsLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer conducts audits of suppliers to evaluate supplier compliance with company standards for trafficking and slavery in supply chains? The disclosure shall specify if the verification was not an independent, unannounced audit.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDCertificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer requires direct suppliers to certify that materials incorporated into the product comply with the laws regarding slavery and human trafficking of the country or countries in which they are doing business?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDTrainingLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer provides company employees and management, who have direct responsibility for supply chain management, training on human trafficking and slavery, particularly with respect to mitigating risks within the supply chains of products?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDVerificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer engages in verification of product supply chains to evaluate and address risks of human trafficking and slavery? If the company conducts verification], the disclosure shall specify if the verification was not conducted by a third party.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SDSEyeProtectionClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/SDSEyeProtectionClassification`](https://huggingface.co/datasets/BASF-AI/SDSEyeProtectionClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SDSEyeProtectionClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sds_eye_protection`](https://huggingface.co/datasets/mteb/sds_eye_protection) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SDSGlovesClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/SDSGlovesClassification`](https://huggingface.co/datasets/BASF-AI/SDSGlovesClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SDSGlovesClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sds_gloves`](https://huggingface.co/datasets/mteb/sds_gloves) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SIB200Classification

SIB-200 is the largest publicly available topic classification
        dataset based on Flores-200 covering 205 languages and dialects annotated. The dataset is
        annotated in English for the topics,  science/technology, travel, politics, sports,
        health, entertainment, and geography. The labels are then transferred to the other languages
        in Flores-200 which are human-translated.
        

**Dataset:** [`mteb/sib200`](https://huggingface.co/datasets/mteb/sib200) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2309.07445)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ace, acm, acq, aeb, afr, ... (197) | News, Written | expert-annotated | human-translated and localized |



#### SIDClassification

SID Classification

**Dataset:** [`MCINext/sid-classification`](https://huggingface.co/datasets/MCINext/sid-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Academic | derived | found |



#### SIDClassification.v2

SID Classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sid`](https://huggingface.co/datasets/mteb/sid) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Academic | derived | found |



#### SanskritShlokasClassification

This data set contains ~500 Shlokas  

**Dataset:** [`bpHigh/iNLTK_Sanskrit_Shlokas_Dataset`](https://huggingface.co/datasets/bpHigh/iNLTK_Sanskrit_Shlokas_Dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-sanskrit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | san | Religious, Written | derived | found |



#### SardiStanceClassification

SardiStance is a unique dataset designed for the task of stance detection in Italian tweets. It consists of tweets related to the Sardines movement, providing a valuable resource for researchers and practitioners in the field of NLP.

**Dataset:** [`MattiaSangermano/SardiStance`](https://huggingface.co/datasets/MattiaSangermano/SardiStance) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/mirkolai/evalita-sardistance)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Social | derived | found |



#### ScalaClassification

ScaLa a linguistic acceptability dataset for the mainland Scandinavian languages automatically constructed from dependency annotations in Universal Dependencies Treebanks.
        Published as part of 'ScandEval: A Benchmark for Scandinavian Natural Language Processing'

**Dataset:** [`mteb/multilingual-scala-classification`](https://huggingface.co/datasets/mteb/multilingual-scala-classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, nno, nob, swe | Blog, Fiction, News, Non-fiction, Spoken, ... (7) | human-annotated | created |



#### ScandiSentClassification

The corpus is crawled from se.trustpilot.com, no.trustpilot.com, dk.trustpilot.com, fi.trustpilot.com and trustpilot.com.

**Dataset:** [`mteb/scandisent`](https://huggingface.co/datasets/mteb/scandisent) • **License:** openrail • [Learn more →](https://github.com/timpal0l/ScandiSent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, eng, fin, nob, swe | Reviews, Written | expert-annotated | found |



#### SentiRuEval2016

Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results.

**Dataset:** [`mteb/SentiRuEval2016`](https://huggingface.co/datasets/mteb/SentiRuEval2016) • **License:** not specified • [Learn more →](https://github.com/mokoron/sentirueval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### SentiRuEval2016.v2

Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/senti_ru_eval2016`](https://huggingface.co/datasets/mteb/senti_ru_eval2016) • **License:** not specified • [Learn more →](https://github.com/mokoron/sentirueval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### SentimentAnalysisHindi

Hindi Sentiment Analysis Dataset

**Dataset:** [`OdiaGenAI/sentiment_analysis_hindi`](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | hin | Reviews, Written | derived | found |



#### SentimentAnalysisHindi.v2

Hindi Sentiment Analysis Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sentiment_analysis_hindi`](https://huggingface.co/datasets/mteb/sentiment_analysis_hindi) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | hin | Reviews, Written | derived | found |



#### SentimentDKSF

The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.

**Dataset:** [`hezarai/sentiment-dksf`](https://huggingface.co/datasets/hezarai/sentiment-dksf) • **License:** not specified • [Learn more →](https://github.com/hezarai/hezar)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### SentimentDKSF.v2

The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sentiment_dksf`](https://huggingface.co/datasets/mteb/sentiment_dksf) • **License:** not specified • [Learn more →](https://github.com/hezarai/hezar)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### SinhalaNewsClassification

This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).

**Dataset:** [`NLPC-UOM/Sinhala-News-Category-classification`](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SinhalaNewsClassification.v2

This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sinhala_news`](https://huggingface.co/datasets/mteb/sinhala_news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SinhalaNewsSourceClassification

This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).

**Dataset:** [`NLPC-UOM/Sinhala-News-Source-classification`](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SinhalaNewsSourceClassification.v2

This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sinhala_news_source`](https://huggingface.co/datasets/mteb/sinhala_news_source) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SiswatiNewsClassification

Siswati News Classification Dataset

**Dataset:** [`isaacchung/siswati-news`](https://huggingface.co/datasets/isaacchung/siswati-news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ssw | News, Written | human-annotated | found |



#### SiswatiNewsClassification.v2

Siswati News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/siswati_news`](https://huggingface.co/datasets/mteb/siswati_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ssw | News, Written | human-annotated | found |



#### SlovakHateSpeechClassification

The dataset contains posts from a social network with human annotations for hateful or offensive language in Slovak.

**Dataset:** [`TUKE-KEMT/hate_speech_slovak`](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Social, Written | human-annotated | found |



#### SlovakHateSpeechClassification.v2

The dataset contains posts from a social network with human annotations for hateful or offensive language in Slovak.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/slovak_hate_speech`](https://huggingface.co/datasets/mteb/slovak_hate_speech) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Social, Written | human-annotated | found |



#### SlovakMovieReviewSentimentClassification

User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)

**Dataset:** [`janko/sk_csfd-movie-reviews`](https://huggingface.co/datasets/janko/sk_csfd-movie-reviews) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | svk | Reviews, Written | derived | found |



#### SlovakMovieReviewSentimentClassification.v2

User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/slovak_movie_review_sentiment`](https://huggingface.co/datasets/mteb/slovak_movie_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | svk | Reviews, Written | derived | found |



#### SouthAfricanLangClassification

A language identification test set for 11 South African Languages.

**Dataset:** [`mlexplorer008/south_african_language_identification`](https://huggingface.co/datasets/mlexplorer008/south_african_language_identification) • **License:** mit • [Learn more →](https://www.kaggle.com/competitions/south-african-language-identification/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, eng, nbl, nso, sot, ... (11) | Non-fiction, Web, Written | expert-annotated | found |



#### SpanishNewsClassification

A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.

**Dataset:** [`MarcOrfilaCarreras/spanish-news`](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | News, Written | derived | found |



#### SpanishNewsClassification.v2

A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/spanish_news`](https://huggingface.co/datasets/mteb/spanish_news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | News, Written | derived | found |



#### SpanishSentimentClassification

A Spanish dataset for sentiment classification.

**Dataset:** [`sepidmnorozy/Spanish_sentiment`](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | Reviews, Written | derived | found |



#### SpanishSentimentClassification.v2

A Spanish dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/spanish_sentiment`](https://huggingface.co/datasets/mteb/spanish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | Reviews, Written | derived | found |



#### SwahiliNewsClassification

Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets

**Dataset:** [`Mollel/SwahiliNewsClassification`](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swa | News, Written | derived | found |



#### SwahiliNewsClassification.v2

Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swahili_news`](https://huggingface.co/datasets/mteb/swahili_news) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swa | News, Written | derived | found |



#### SweRecClassification

A Swedish dataset for sentiment classification on review

**Dataset:** [`mteb/swerec_classification`](https://huggingface.co/datasets/mteb/swerec_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SweRecClassification.v2

A Swedish dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swe_rec`](https://huggingface.co/datasets/mteb/swe_rec) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SwedishSentimentClassification

Dataset of Swedish reviews scarped from various public available websites

**Dataset:** [`mteb/SwedishSentimentClassification`](https://huggingface.co/datasets/mteb/SwedishSentimentClassification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/swedish_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SwedishSentimentClassification.v2

Dataset of Swedish reviews scarped from various public available websites
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swedish_sentiment`](https://huggingface.co/datasets/mteb/swedish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/swedish_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SwissJudgementClassification

Multilingual, diachronic dataset of Swiss Federal Supreme Court cases annotated with the respective binarized judgment outcome (approval/dismissal)

**Dataset:** [`rcds/swiss_judgment_prediction`](https://huggingface.co/datasets/rcds/swiss_judgment_prediction) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nllp-1.3/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, fra, ita | Legal, Written | expert-annotated | found |



#### SynPerChatbotConvSAAnger

Synthetic Persian Chatbot Conversational Sentiment Analysis Anger

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAFear

Synthetic Persian Chatbot Conversational Sentiment Analysis Fear

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAFriendship

Synthetic Persian Chatbot Conversational Sentiment Analysis Friendship

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAHappiness

Synthetic Persian Chatbot Conversational Sentiment Analysis Happiness

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAJealousy

Synthetic Persian Chatbot Conversational Sentiment Analysis Jealousy

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSALove

Synthetic Persian Chatbot Conversational Sentiment Analysis Love

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSASadness

Synthetic Persian Chatbot Conversational Sentiment Analysis Sadness

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSASatisfaction

Synthetic Persian Chatbot Conversational Sentiment Analysis Satisfaction

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSASurprise

Synthetic Persian Chatbot Conversational Sentiment Analysis Surprise

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAToneChatbotClassification

Synthetic Persian Chatbot Conversational Sentiment Analysis Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAToneUserClassification

Synthetic Persian Chatbot Conversational Sentiment Analysis Tone User

**Dataset:** [`MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification`](https://huggingface.co/datasets/MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotRAGToneChatbotClassification

Synthetic Persian Chatbot RAG Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotRAGToneUserClassification

Synthetic Persian Chatbot RAG Tone User Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-tone-user-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotSatisfactionLevelClassification

Synthetic Persian Chatbot Satisfaction Level Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-satisfaction-level-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-satisfaction-level-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotToneChatbotClassification

Synthetic Persian Chatbot Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotToneUserClassification

Synthetic Persian Chatbot Tone User Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-tone-user-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerTextToneClassification

Persian Text Tone

**Dataset:** [`MCINext/synthetic-persian-text-tone-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-text-tone-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



#### SynPerTextToneClassification.v2

Persian Text Tone
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/syn_per_text_tone`](https://huggingface.co/datasets/mteb/syn_per_text_tone) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



#### TNews

Short Text Classification for News

**Dataset:** [`C-MTEB/TNews-classification`](https://huggingface.co/datasets/C-MTEB/TNews-classification) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### TNews.v2

Short Text Classification for News
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/t_news`](https://huggingface.co/datasets/mteb/t_news) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### TamilNewsClassification

A Tamil dataset for 6-class classification of Tamil news articles

**Dataset:** [`mlexplorer008/tamil_news_classification`](https://huggingface.co/datasets/mlexplorer008/tamil_news_classification) • **License:** mit • [Learn more →](https://github.com/vanangamudi/tamil-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tam | News, Written | derived | found |



#### TamilNewsClassification.v2

A Tamil dataset for 6-class classification of Tamil news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tamil_news`](https://huggingface.co/datasets/mteb/tamil_news) • **License:** mit • [Learn more →](https://github.com/vanangamudi/tamil-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tam | News, Written | derived | found |



#### TelemarketingSalesRuleLegalBenchClassification

Determine how 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) (governing deceptive practices) apply to different fact patterns. This dataset is designed to test a model’s ability to apply 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) of the Telemarketing Sales Rule to a simple fact pattern with a clear outcome. Each fact pattern ends with the question: “Is this a violation of the Telemarketing Sales Rule?” Each fact pattern is paired with the answer “Yes” or the answer “No.” Fact patterns are listed in the column “text,” and answers are listed in the column “label.”

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### TeluguAndhraJyotiNewsClassification

A Telugu dataset for 5-class classification of Telugu news articles

**Dataset:** [`mlexplorer008/telugu_news_classification`](https://huggingface.co/datasets/mlexplorer008/telugu_news_classification) • **License:** mit • [Learn more →](https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tel | News, Written | derived | found |



#### TeluguAndhraJyotiNewsClassification.v2

A Telugu dataset for 5-class classification of Telugu news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/telugu_andhra_jyoti_news`](https://huggingface.co/datasets/mteb/telugu_andhra_jyoti_news) • **License:** mit • [Learn more →](https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tel | News, Written | derived | found |



#### TenKGnadClassification

10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).

**Dataset:** [`mteb/TenKGnadClassification`](https://huggingface.co/datasets/mteb/TenKGnadClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | News, Written | expert-annotated | found |



#### TenKGnadClassification.v2

10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ten_k_gnad`](https://huggingface.co/datasets/mteb/ten_k_gnad) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | News, Written | expert-annotated | found |



#### TextualismToolDictionariesLegalBenchClassification

Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the dictionary meaning of terms.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### TextualismToolPlainLegalBenchClassification

Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the ordinary (“plain”) meaning of terms.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ToxicChatClassification

This dataset contains toxicity annotations on 10K user
            prompts collected from the Vicuna online demo. We utilize a human-AI
            collaborative annotation framework to guarantee the quality of annotation
            while maintaining a feasible annotation workload. The details of data
            collection, pre-processing, and annotation can be found in our paper.
            We believe that ToxicChat can be a valuable resource to drive further
            advancements toward building a safe and healthy environment for user-AI
            interactions.
            Only human annotated samples are selected here.

**Dataset:** [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.findings-emnlp.311/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Constructed, Written | expert-annotated | found |



#### ToxicChatClassification.v2

This dataset contains toxicity annotations on 10K user
            prompts collected from the Vicuna online demo. We utilize a human-AI
            collaborative annotation framework to guarantee the quality of annotation
            while maintaining a feasible annotation workload. The details of data
            collection, pre-processing, and annotation can be found in our paper.
            We believe that ToxicChat can be a valuable resource to drive further
            advancements toward building a safe and healthy environment for user-AI
            interactions.
            Only human annotated samples are selected here.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/toxic_chat`](https://huggingface.co/datasets/mteb/toxic_chat) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.findings-emnlp.311/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Constructed, Written | expert-annotated | found |



#### ToxicConversationsClassification

Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.

**Dataset:** [`mteb/toxic_conversations_50k`](https://huggingface.co/datasets/mteb/toxic_conversations_50k) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### ToxicConversationsClassification.v2

Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/toxic_conversations`](https://huggingface.co/datasets/mteb/toxic_conversations) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### ToxicConversationsVNClassification

A translated dataset from Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/toxic-conversations-50k-vn`](https://huggingface.co/datasets/GreenNode/toxic-conversations-50k-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Social, Written | derived | machine-translated and LM verified |



#### TswanaNewsClassification

Tswana News Classification Dataset

**Dataset:** [`dsfsi/daily-news-dikgang`](https://huggingface.co/datasets/dsfsi/daily-news-dikgang) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tsn | News, Written | derived | found |



#### TswanaNewsClassification.v2

Tswana News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tswana_news`](https://huggingface.co/datasets/mteb/tswana_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tsn | News, Written | derived | found |



#### TurkicClassification

A dataset of news classification in three Turkic languages.

**Dataset:** [`Electrotubbie/classification_Turkic_languages`](https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bak, kaz, kir | News, Written | derived | found |



#### TurkishMovieSentimentClassification

Turkish Movie Review Dataset

**Dataset:** [`asparius/Turkish-Movie-Review`](https://huggingface.co/datasets/asparius/Turkish-Movie-Review) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TurkishMovieSentimentClassification.v2

Turkish Movie Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/turkish_movie_sentiment`](https://huggingface.co/datasets/mteb/turkish_movie_sentiment) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TurkishProductSentimentClassification

Turkish Product Review Dataset

**Dataset:** [`asparius/Turkish-Product-Review`](https://huggingface.co/datasets/asparius/Turkish-Product-Review) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TurkishProductSentimentClassification.v2

Turkish Product Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/turkish_product_sentiment`](https://huggingface.co/datasets/mteb/turkish_product_sentiment) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TweetEmotionClassification

A dataset of 10,000 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.

**Dataset:** [`mteb/TweetEmotionClassification`](https://huggingface.co/datasets/mteb/TweetEmotionClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetEmotionClassification.v2

A dataset of 10,012 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/TweetEmotionClassification`](https://huggingface.co/datasets/mteb/TweetEmotionClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetSarcasmClassification

Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.

**Dataset:** [`iabufarha/ar_sarcasm`](https://huggingface.co/datasets/iabufarha/ar_sarcasm) • **License:** mit • [Learn more →](https://aclanthology.org/2020.osact-1.5/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetSarcasmClassification.v2

Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/tweet_sarcasm`](https://huggingface.co/datasets/mteb/tweet_sarcasm) • **License:** mit • [Learn more →](https://aclanthology.org/2020.osact-1.5/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetSentimentClassification

A multilingual Sentiment Analysis dataset consisting of tweets in 8 different languages.

**Dataset:** [`mteb/tweet_sentiment_multilingual`](https://huggingface.co/datasets/mteb/tweet_sentiment_multilingual) • **License:** cc-by-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.27)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, deu, eng, fra, hin, ... (8) | Social, Written | human-annotated | found |



#### TweetSentimentExtractionClassification



**Dataset:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### TweetSentimentExtractionClassification.v2


        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### TweetSentimentExtractionVNClassification

A collection of translated tweets annotated for sentiment extraction.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/tweet-sentiment-extraction-vn`](https://huggingface.co/datasets/GreenNode/tweet-sentiment-extraction-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Social, Written | derived | machine-translated and LM verified |



#### TweetTopicSingleClassification

Topic classification dataset on Twitter with 6 labels. Each instance of
        TweetTopic comes with a timestamp which distributes from September 2019 to August 2021.
        Tweets were preprocessed before the annotation to normalize some artifacts, converting
        URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified
        usernames, we replace its display name (or account name) with symbols {@}.
        

**Dataset:** [`cardiffnlp/tweet_topic_single`](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.09824)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Social, Written | expert-annotated | found |



#### TweetTopicSingleClassification.v2

Topic classification dataset on Twitter with 6 labels. Each instance of
        TweetTopic comes with a timestamp which distributes from September 2019 to August 2021.
        Tweets were preprocessed before the annotation to normalize some artifacts, converting
        URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified
        usernames, we replace its display name (or account name) with symbols {@}.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tweet_topic_single`](https://huggingface.co/datasets/mteb/tweet_topic_single) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.09824)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Social, Written | expert-annotated | found |



#### UCCVCommonLawLegalBenchClassification

Determine if a contract is governed by the Uniform Commercial Code (UCC) or the common law of contracts.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### UkrFormalityClassification


        This dataset contains Ukrainian Formality Classification dataset obtained by
        trainslating English GYAFC data.
        English data source: https://aclanthology.org/N18-1012/
        Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M
        Additionally, the dataset was balanced, witha labels: 0 - informal, 1 - formal.
        

**Dataset:** [`ukr-detect/ukr-formality-dataset-translated-gyafc`](https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc) • **License:** openrail++ • [Learn more →](https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ukr | News, Written | derived | machine-translated |



#### UkrFormalityClassification.v2


        This dataset contains Ukrainian Formality Classification dataset obtained by
        trainslating English GYAFC data.
        English data source: https://aclanthology.org/N18-1012/
        Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M
        Additionally, the dataset was balanced, witha labels: 0 - informal, 1 - formal.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ukr_formality`](https://huggingface.co/datasets/mteb/ukr_formality) • **License:** openrail++ • [Learn more →](https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ukr | News, Written | derived | machine-translated |



#### UnfairTOSLegalBenchClassification

Given a clause from a terms-of-service contract, determine the category the clause belongs to. The purpose of this task is classifying clauses in Terms of Service agreements. Clauses have been annotated by into nine categories: ['Arbitration', 'Unilateral change', 'Content removal', 'Jurisdiction', 'Choice of law', 'Limitation of liability', 'Unilateral termination', 'Contract by using', 'Other']. The first eight categories correspond to clauses that would potentially be deemed potentially unfair. The last category (Other) corresponds to clauses in agreements which don’t fit into these categories.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### UrduRomanSentimentClassification

The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)

**Dataset:** [`mteb/UrduRomanSentimentClassification`](https://huggingface.co/datasets/mteb/UrduRomanSentimentClassification) • **License:** mit • [Learn more →](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | urd | Social, Written | derived | found |



#### UrduRomanSentimentClassification.v2

The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/urdu_roman_sentiment`](https://huggingface.co/datasets/mteb/urdu_roman_sentiment) • **License:** mit • [Learn more →](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | urd | Social, Written | derived | found |



#### VieStudentFeedbackClassification

A Vietnamese dataset for classification of student feedback

**Dataset:** [`uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) • **License:** mit • [Learn more →](https://ieeexplore.ieee.org/document/8573337)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | human-annotated | created |



#### VieStudentFeedbackClassification.v2

A Vietnamese dataset for classification of student feedback
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/vie_student_feedback`](https://huggingface.co/datasets/mteb/vie_student_feedback) • **License:** mit • [Learn more →](https://ieeexplore.ieee.org/document/8573337)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | human-annotated | created |



#### WRIMEClassification

A dataset of Japanese social network rated for sentiment

**Dataset:** [`shunk031/wrime`](https://huggingface.co/datasets/shunk031/wrime) • **License:** https://huggingface.co/datasets/shunk031/wrime#licensing-information • [Learn more →](https://aclanthology.org/2021.naacl-main.169/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Social, Written | human-annotated | found |



#### WRIMEClassification.v2

A dataset of Japanese social network rated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wrime`](https://huggingface.co/datasets/mteb/wrime) • **License:** https://huggingface.co/datasets/shunk031/wrime#licensing-information • [Learn more →](https://aclanthology.org/2021.naacl-main.169/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Social, Written | human-annotated | found |



#### Waimai

Sentiment Analysis of user reviews on takeaway platforms

**Dataset:** [`C-MTEB/waimai-classification`](https://huggingface.co/datasets/C-MTEB/waimai-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### Waimai.v2

Sentiment Analysis of user reviews on takeaway platforms
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/waimai`](https://huggingface.co/datasets/mteb/waimai) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### WikipediaBioMetChemClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaBioMetChemClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_bio_met_chem`](https://huggingface.co/datasets/mteb/wikipedia_bio_met_chem) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaBiolumNeurochemClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemEngSpecialtiesClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium5Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemFieldsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEZ10Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEZ10Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemFieldsClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_chem_fields`](https://huggingface.co/datasets/mteb/wikipedia_chem_fields) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemistryTopicsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy10Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCompChemSpectroscopyClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCompChemSpectroscopyClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_comp_chem_spectroscopy`](https://huggingface.co/datasets/mteb/wikipedia_comp_chem_spectroscopy) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCryobiologySeparationClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy5Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy5Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCrystallographyAnalyticalClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCrystallographyAnalyticalClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_crystallography_analytical`](https://huggingface.co/datasets/mteb/wikipedia_crystallography_analytical) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaGreenhouseEnantiopureClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaIsotopesFissionClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaLuminescenceClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaOrganicInorganicClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2SpecialClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SpecialClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaSaltsSemiconductorsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaSolidStateColloidalClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaTheoreticalAppliedClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEZ2Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEZ2Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaTheoreticalAppliedClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_theoretical_applied`](https://huggingface.co/datasets/mteb/wikipedia_theoretical_applied) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WisesightSentimentClassification

Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)

**Dataset:** [`mteb/WisesightSentimentClassification`](https://huggingface.co/datasets/mteb/WisesightSentimentClassification) • **License:** cc0-1.0 • [Learn more →](https://github.com/PyThaiNLP/wisesight-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tha | News, Social, Written | expert-annotated | found |



#### WisesightSentimentClassification.v2

Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wisesight_sentiment`](https://huggingface.co/datasets/mteb/wisesight_sentiment) • **License:** cc0-1.0 • [Learn more →](https://github.com/PyThaiNLP/wisesight-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tha | News, Social, Written | expert-annotated | found |



#### WongnaiReviewsClassification

Wongnai features over 200,000 restaurants, beauty salons, and spas across Thailand on its platform, with detailed information about each merchant and user reviews. In this dataset there are 5 classes corressponding each star rating

**Dataset:** [`Wongnai/wongnai_reviews`](https://huggingface.co/datasets/Wongnai/wongnai_reviews) • **License:** lgpl-3.0 • [Learn more →](https://github.com/wongnai/wongnai-corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tha | Reviews, Written | derived | found |



#### YahooAnswersTopicsClassification

Dataset composed of questions and answers from Yahoo Answers, categorized into topics.

**Dataset:** [`community-datasets/yahoo_answers_topics`](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/yahoo_answers_topics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Web, Written | human-annotated | found |



#### YahooAnswersTopicsClassification.v2

Dataset composed of questions and answers from Yahoo Answers, categorized into topics.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yahoo_answers_topics`](https://huggingface.co/datasets/mteb/yahoo_answers_topics) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/yahoo_answers_topics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Web, Written | human-annotated | found |



#### YelpReviewFullClassification

Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.

**Dataset:** [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) • **License:** https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### YelpReviewFullClassification.v2

Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yelp_review_full`](https://huggingface.co/datasets/mteb/yelp_review_full) • **License:** https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### YueOpenriceReviewClassification

A Cantonese dataset for review classification

**Dataset:** [`izhx/yue-openrice-review`](https://huggingface.co/datasets/izhx/yue-openrice-review) • **License:** not specified • [Learn more →](https://github.com/Christainx/Dataset_Cantonese_Openrice)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | yue | Reviews, Spoken | human-annotated | found |



#### YueOpenriceReviewClassification.v2

A Cantonese dataset for review classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yue_openrice_review`](https://huggingface.co/datasets/mteb/yue_openrice_review) • **License:** not specified • [Learn more →](https://github.com/Christainx/Dataset_Cantonese_Openrice)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | yue | Reviews, Spoken | human-annotated | found |



## Clustering

- **Number of tasks of the given type:** 98 

#### AlloProfClusteringP2P

Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



#### AlloProfClusteringP2P.v2

Clustering of document titles and descriptions from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



#### AlloProfClusteringS2S

Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



#### AlloProfClusteringS2S.v2

Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.

**Dataset:** [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) • **License:** mit • [Learn more →](https://huggingface.co/datasets/lyon-nlp/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Encyclopaedic, Written | human-annotated | found |



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



#### ArxivClusteringP2P.v2

Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-p2p`](https://huggingface.co/datasets/mteb/arxiv-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



#### ArxivClusteringS2S

Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category

**Dataset:** [`mteb/arxiv-clustering-s2s`](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s) • **License:** cc0-1.0 • [Learn more →](https://www.kaggle.com/Cornell-University/arxiv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Written | derived | found |



#### BeytooteClustering

Beytoote Website Articles Clustering

**Dataset:** [`MCINext/beytoote-clustering`](https://huggingface.co/datasets/MCINext/beytoote-clustering) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | News | derived | found |



#### BigPatentClustering

Clustering of documents from the Big Patent dataset. Test set only includes documents belonging to a single category, with a total of 9 categories.

**Dataset:** [`jinaai/big-patent-clustering`](https://huggingface.co/datasets/jinaai/big-patent-clustering) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NortheasternUniversity/big_patent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Legal, Written | derived | found |



#### BigPatentClustering.v2

Clustering of documents from the Big Patent dataset. Test set only includes documents belonging to a single category, with a total of 9 categories.

**Dataset:** [`mteb/big-patent`](https://huggingface.co/datasets/mteb/big-patent) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NortheasternUniversity/big_patent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Legal, Written | derived | found |



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

**Dataset:** [`slvnwhrl/blurbs-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p) • **License:** not specified • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Written | not specified | not specified |



#### BlurbsClusteringP2P.v2

Clustering of book titles+blurbs. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-p2p) • **License:** cc-by-nc-4.0 • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Fiction, Written | derived | found |



#### BlurbsClusteringS2S

Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s) • **License:** not specified • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Written | not specified | not specified |



#### BlurbsClusteringS2S.v2

Clustering of book titles. Clustering of 28 sets, either on the main or secondary genre.

**Dataset:** [`slvnwhrl/blurbs-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/blurbs-clustering-s2s) • **License:** cc-by-nc-4.0 • [Learn more →](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Fiction, Written | derived | found |



#### BuiltBenchClusteringP2P

Clustering of built asset item descriptions based on categories identified within industry classification systems such as IFC, Uniclass, etc.

**Dataset:** [`mehrzad-shahin/BuiltBench-clustering-p2p`](https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-clustering-p2p) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Engineering, Written | derived | created |



#### BuiltBenchClusteringS2S

Clustering of built asset names/titles based on categories identified within industry classification systems such as IFC, Uniclass, etc.

**Dataset:** [`mehrzad-shahin/BuiltBench-clustering-s2s`](https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-clustering-s2s) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Engineering, Written | derived | created |



#### CLSClusteringP2P

Clustering of titles + abstract from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringP2P`](https://huggingface.co/datasets/C-MTEB/CLSClusteringP2P) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | not specified | not specified | not specified |



#### CLSClusteringP2P.v2

Clustering of titles + abstract from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringP2P`](https://huggingface.co/datasets/C-MTEB/CLSClusteringP2P) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | Academic, Written | derived | found |



#### CLSClusteringS2S

Clustering of titles from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringS2S`](https://huggingface.co/datasets/C-MTEB/CLSClusteringS2S) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | not specified | not specified | not specified |



#### CLSClusteringS2S.v2

Clustering of titles from CLS dataset. Clustering of 13 sets on the main category.

**Dataset:** [`C-MTEB/CLSClusteringS2S`](https://huggingface.co/datasets/C-MTEB/CLSClusteringS2S) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2209.05034)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | Academic, Written | derived | found |



#### ClusTREC-Covid

A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.

**Dataset:** [`Uri-ka/ClusTREC-Covid`](https://huggingface.co/datasets/Uri-ka/ClusTREC-Covid) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Academic, Medical, Written | expert-annotated | created |



#### DigikalamagClustering

A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.

**Dataset:** [`PNLPhub/DigiMag`](https://huggingface.co/datasets/PNLPhub/DigiMag) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/tc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | Web | derived | found |



#### EightTagsClustering

Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, food, medicine, motorization, work, sport and technology.

**Dataset:** [`PL-MTEB/8tags-clustering`](https://huggingface.co/datasets/PL-MTEB/8tags-clustering) • **License:** gpl-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Social, Written | derived | found |



#### EightTagsClustering.v2

Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, food, medicine, motorization, work, sport and technology.

**Dataset:** [`PL-MTEB/8tags-clustering`](https://huggingface.co/datasets/PL-MTEB/8tags-clustering) • **License:** gpl-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Social, Written | derived | found |



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



#### HALClusteringS2S.v2

Clustering of titles from HAL (https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)

**Dataset:** [`lyon-nlp/clustering-hal-s2s`](https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fra | Academic, Written | human-annotated | found |



#### HamshahriClustring

These datasets have been extracted from the RSS feed of two Farsi news agency websites.

**Dataset:** [`community-datasets/farsi_news`](https://huggingface.co/datasets/community-datasets/farsi_news) • **License:** not specified • [Learn more →](https://github.com/mallahyari/Farsi-datasets)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | News | derived | found |



#### IndicReviewsClusteringP2P

Clustering of reviews from IndicSentiment dataset. Clustering of 14 sets on the generic categories label.

**Dataset:** [`mteb/IndicReviewsClusteringP2P`](https://huggingface.co/datasets/mteb/IndicReviewsClusteringP2P) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | asm, ben, brx, guj, hin, ... (13) | Reviews, Written | human-annotated | machine-translated and verified |



#### KlueMrcDomainClustering

this dataset is a processed and redistributed version of the KLUE-MRC dataset. Domain: Game / Media / Automotive / Finance / Real Estate / Education

**Dataset:** [`on-and-on/clustering_klue_mrc_context_domain`](https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_context_domain) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_context_domain)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | kor | News, Written | human-annotated | found |



#### KlueYnatMrcCategoryClustering

this dataset is a processed and redistributed version of the KLUE-Ynat & KLUE-MRC  dataset. News_category: IT/Science, Sports, Media/Culture, Ecomomy/Finance, Real Estate

**Dataset:** [`on-and-on/clustering_klue_mrc_ynat_title`](https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_ynat_title) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_ynat_title)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | v_measure | kor | News, Written | human-annotated | found |



#### LivedoorNewsClustering

Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-nd-2.1-jp • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | jpn | News, Written | derived | found |



#### LivedoorNewsClustering.v2

Clustering of the news reports of a Japanese news site, Livedoor News by RONDHUIT Co, Ltd. in 2012. It contains over 7,000 news report texts across 9 categories (topics). Version 2 updated on LivedoorNewsClustering by removing pairs where one of entries contain an empty sentences.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-nd-2.1-jp • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | jpn | News, Written | derived | found |



#### MLSUMClusteringP2P

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



#### MLSUMClusteringP2P.v2

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



#### MLSUMClusteringS2S

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



#### MLSUMClusteringS2S.v2

Clustering of newspaper article contents and titles from MLSUM dataset. Clustering of 10 sets on the newpaper article topics.

**Dataset:** [`mteb/mlsum`](https://huggingface.co/datasets/mteb/mlsum) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/mlsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu, fra, rus, spa | News, Written | derived | found |



#### MasakhaNEWSClusteringP2P

Clustering of news article headlines and texts from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.

**Dataset:** [`masakhane/masakhanews`](https://huggingface.co/datasets/masakhane/masakhanews) • **License:** afl-3.0 • [Learn more →](https://huggingface.co/datasets/masakhane/masakhanews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | amh, eng, fra, hau, ibo, ... (16) | News, Non-fiction, Written | derived | found |



#### MasakhaNEWSClusteringS2S

Clustering of news article headlines from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.

**Dataset:** [`masakhane/masakhanews`](https://huggingface.co/datasets/masakhane/masakhanews) • **License:** afl-3.0 • [Learn more →](https://huggingface.co/datasets/masakhane/masakhanews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | amh, eng, fra, hau, ibo, ... (16) | News, Written | human-annotated | not specified |



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

MewsC-16 (Multilingual Short Text Clustering Dataset for News in 16 languages) is constructed from Wikinews.
        This dataset is the Japanese split of MewsC-16, containing topic sentences from Wikinews articles in 12 categories.
        More detailed information is available in the Appendix E of the citation.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | jpn | News, Written | derived | found |



#### NLPTwitterAnalysisClustering

Clustering of tweets from twitter across 26 categories.

**Dataset:** [`hamedhf/nlp_twitter_analysis`](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/commits/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | Social | derived | found |



#### PlscClusteringP2P

Clustering of Polish article titles+abstracts from Library of Science (https://bibliotekanauki.pl/), either on the scientific field or discipline.

**Dataset:** [`PL-MTEB/plsc-clustering-p2p`](https://huggingface.co/datasets/PL-MTEB/plsc-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/rafalposwiata/plsc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | pol | Academic, Written | derived | found |



#### PlscClusteringP2P.v2

Clustering of Polish article titles+abstracts from Library of Science (https://bibliotekanauki.pl/), either on the scientific field or discipline.

**Dataset:** [`PL-MTEB/plsc-clustering-p2p`](https://huggingface.co/datasets/PL-MTEB/plsc-clustering-p2p) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/rafalposwiata/plsc)

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



#### RedditClustering-VN

A translated dataset from Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/reddit-clustering-vn`](https://huggingface.co/datasets/GreenNode/reddit-clustering-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Social, Web, Written | derived | machine-translated and LM verified |



#### RedditClustering.v2

Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/reddit-clustering`](https://huggingface.co/datasets/mteb/reddit-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



#### RedditClusteringP2P

Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.

**Dataset:** [`mteb/reddit-clustering-p2p`](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



#### RedditClusteringP2P-VN

A translated dataset from Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/reddit-clustering-p2p-vn`](https://huggingface.co/datasets/GreenNode/reddit-clustering-p2p-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Social, Web, Written | derived | machine-translated and LM verified |



#### RedditClusteringP2P.v2

Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.

**Dataset:** [`mteb/reddit-clustering-p2p`](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Social, Web, Written | derived | found |



#### RomaniBibleClustering

Clustering verses from the Bible in Kalderash Romani by book.

**Dataset:** [`kardosdrur/romani-bible`](https://huggingface.co/datasets/kardosdrur/romani-bible) • **License:** mit • [Learn more →](https://romani.global.bible/info)

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

SIB-200 is the largest publicly available topic classification
        dataset based on Flores-200 covering 205 languages and dialects annotated. The dataset is
        annotated in English for the topics,  science/technology, travel, politics, sports,
        health, entertainment, and geography. The labels are then transferred to the other languages
        in Flores-200 which are human-translated.
        

**Dataset:** [`mteb/sib200`](https://huggingface.co/datasets/mteb/sib200) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2309.07445)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | ace, acm, acq, aeb, afr, ... (197) | News, Written | expert-annotated | human-translated and localized |



#### SIDClustring

Clustering of summariesfrom SIDClustring across categories.

**Dataset:** [`MCINext/sid-clustering`](https://huggingface.co/datasets/MCINext/sid-clustering) • **License:** not specified • [Learn more →](https://www.sid.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | fas | Academic | derived | found |



#### SNLClustering

Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.

**Dataset:** [`navjordj/SNL_summarization`](https://huggingface.co/datasets/navjordj/SNL_summarization) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/navjordj/SNL_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### SNLHierarchicalClusteringP2P

Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.

**Dataset:** [`mteb/SNLHierarchicalClusteringP2P`](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringP2P) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringP2P)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### SNLHierarchicalClusteringS2S

Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.

**Dataset:** [`mteb/SNLHierarchicalClusteringS2S`](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringS2S) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringS2S)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### SpanishNewsClusteringP2P

Clustering of news articles, 7 topics in total.

**Dataset:** [`jinaai/spanish_news_clustering`](https://huggingface.co/datasets/jinaai/spanish_news_clustering) • **License:** not specified • [Learn more →](https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | spa | not specified | not specified | not specified |



#### StackExchangeClustering

Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/stackexchange-clustering`](https://huggingface.co/datasets/mteb/stackexchange-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



#### StackExchangeClustering-VN

A translated dataset from Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stackexchange-clustering-vn`](https://huggingface.co/datasets/GreenNode/stackexchange-clustering-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Web, Written | derived | machine-translated and LM verified |



#### StackExchangeClustering.v2

Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.

**Dataset:** [`mteb/stackexchange-clustering`](https://huggingface.co/datasets/mteb/stackexchange-clustering) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



#### StackExchangeClusteringP2P

Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.

**Dataset:** [`mteb/stackexchange-clustering-p2p`](https://huggingface.co/datasets/mteb/stackexchange-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



#### StackExchangeClusteringP2P-VN

A translated Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stackexchange-clustering-p2p-vn`](https://huggingface.co/datasets/GreenNode/stackexchange-clustering-p2p-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | Web, Written | derived | machine-translated and LM verified |



#### StackExchangeClusteringP2P.v2

Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.

**Dataset:** [`mteb/stackexchange-clustering-p2p`](https://huggingface.co/datasets/mteb/stackexchange-clustering-p2p) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2104.07081)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Web, Written | derived | found |



#### SwednClustering

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.

**Dataset:** [`sbx/superlim-2`](https://huggingface.co/datasets/sbx/superlim-2) • **License:** not specified • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | swe | News, Non-fiction, Written | derived | found |



#### SwednClusteringP2P

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.

**Dataset:** [`sbx/superlim-2`](https://huggingface.co/datasets/sbx/superlim-2) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | swe | News, Non-fiction, Written | derived | found |



#### SwednClusteringS2S

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.

**Dataset:** [`sbx/superlim-2`](https://huggingface.co/datasets/sbx/superlim-2) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | swe | News, Non-fiction, Written | derived | found |



#### TenKGnadClusteringP2P

Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | Web, Written | not specified | found |



#### TenKGnadClusteringP2P.v2

Clustering of news article titles+subheadings+texts. Clustering of 10 splits on the news article category.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-p2p`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-p2p) • **License:** cc-by-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | News, Non-fiction, Written | derived | found |



#### TenKGnadClusteringS2S

Clustering of news article titles. Clustering of 10 splits on the news article category.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s) • **License:** not specified • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | News, Non-fiction, Written | not specified | not specified |



#### TenKGnadClusteringS2S.v2

Clustering of news article titles. Clustering of 10 splits on the news article category.

**Dataset:** [`slvnwhrl/tenkgnad-clustering-s2s`](https://huggingface.co/datasets/slvnwhrl/tenkgnad-clustering-s2s) • **License:** cc-by-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | deu | News, Non-fiction, Written | derived | found |



#### ThuNewsClusteringP2P

Clustering of titles + abstracts from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringP2P`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringP2P) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | not specified | not specified | not specified |



#### ThuNewsClusteringP2P.v2

Clustering of titles + abstracts from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringP2P`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringP2P) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | News, Written | derived | found |



#### ThuNewsClusteringS2S

Clustering of titles from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringS2S`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringS2S) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | not specified | not specified | not specified |



#### ThuNewsClusteringS2S.v2

Clustering of titles from the THUCNews dataset

**Dataset:** [`C-MTEB/ThuNewsClusteringS2S`](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringS2S) • **License:** not specified • [Learn more →](http://thuctc.thunlp.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | cmn | News, Written | derived | found |



#### TwentyNewsgroupsClustering

Clustering of the 20 Newsgroups dataset (subject only).

**Dataset:** [`mteb/twentynewsgroups-clustering`](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) • **License:** not specified • [Learn more →](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | News, Written | derived | found |



#### TwentyNewsgroupsClustering-VN

A translated dataset from Clustering of the 20 Newsgroups dataset (subject only).
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/twentynewsgroups-clustering-vn`](https://huggingface.co/datasets/GreenNode/twentynewsgroups-clustering-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | vie | News, Written | derived | machine-translated and LM verified |



#### TwentyNewsgroupsClustering.v2

Clustering of the 20 Newsgroups dataset (subject only).

**Dataset:** [`mteb/twentynewsgroups-clustering`](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) • **License:** not specified • [Learn more →](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | News, Written | derived | found |



#### VGClustering

Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.

**Dataset:** [`navjordj/VG_summarization`](https://huggingface.co/datasets/navjordj/VG_summarization) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/navjordj/VG_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | News, Non-fiction, Written | derived | found |



#### VGHierarchicalClusteringP2P

Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.

**Dataset:** [`navjordj/VG_summarization`](https://huggingface.co/datasets/navjordj/VG_summarization) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/navjordj/VG_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | News, Non-fiction, Written | derived | found |



#### VGHierarchicalClusteringS2S

Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.

**Dataset:** [`navjordj/VG_summarization`](https://huggingface.co/datasets/navjordj/VG_summarization) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/navjordj/VG_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | nob | News, Non-fiction, Written | derived | found |



#### WikiCitiesClustering

Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.

**Dataset:** [`jinaai/cities_wiki_clustering`](https://huggingface.co/datasets/jinaai/cities_wiki_clustering) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/wikipedia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Encyclopaedic, Written | derived | found |



#### WikiClusteringP2P

Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).

**Dataset:** [`ryzzlestrizzle/multi-wiki-clustering-p2p`](https://huggingface.co/datasets/ryzzlestrizzle/multi-wiki-clustering-p2p) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/Rysias/wiki-clustering)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | bos, cat, ces, dan, eus, ... (14) | Encyclopaedic, Written | derived | created |



#### WikiClusteringP2P.v2

Clustering of wikipedia articles inspired by BlubrbsClusteringP2P. Labels are taken from top-level categories of the respective languages (e.g., https://lv.wikipedia.org/wiki/Kategorija:Pamatkategorijas).

**Dataset:** [`ryzzlestrizzle/multi-wiki-clustering-p2p`](https://huggingface.co/datasets/ryzzlestrizzle/multi-wiki-clustering-p2p) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/Rysias/wiki-clustering)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | bos, cat, ces, dan, eus, ... (14) | Encyclopaedic, Written | derived | created |



#### WikipediaChemistryTopicsClustering

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy10Clustering`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Clustering) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Chemistry | derived | created |



#### WikipediaSpecialtiesInChemistryClustering

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium5Clustering`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Clustering) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | v_measure | eng | Chemistry | derived | created |



## Compositionality

- **Number of tasks of the given type:** 7 

#### AROCocoOrder

Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.

**Dataset:** [`gowitheflow/ARO-COCO-order`](https://huggingface.co/datasets/gowitheflow/ARO-COCO-order) • **License:** mit • [Learn more →](https://openreview.net/forum?id=KRLUvxh8uaX)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | text_acc | eng | Encyclopaedic | expert-annotated | created |



#### AROFlickrOrder

Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.

**Dataset:** [`gowitheflow/ARO-Flickr-Order`](https://huggingface.co/datasets/gowitheflow/ARO-Flickr-Order) • **License:** mit • [Learn more →](https://openreview.net/forum?id=KRLUvxh8uaX)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | text_acc | eng | Encyclopaedic | expert-annotated | created |



#### AROVisualAttribution

Compositionality Evaluation of images to their captions.

**Dataset:** [`gowitheflow/ARO-Visual-Attribution`](https://huggingface.co/datasets/gowitheflow/ARO-Visual-Attribution) • **License:** mit • [Learn more →](https://openreview.net/forum?id=KRLUvxh8uaX)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | text_acc | eng | Encyclopaedic | expert-annotated | created |



#### AROVisualRelation

Compositionality Evaluation of images to their captions.

**Dataset:** [`gowitheflow/ARO-Visual-Relation`](https://huggingface.co/datasets/gowitheflow/ARO-Visual-Relation) • **License:** mit • [Learn more →](https://openreview.net/forum?id=KRLUvxh8uaX)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | text_acc | eng | Encyclopaedic | expert-annotated | created |



#### ImageCoDe

Identify the correct image from a set of similar images based on a precise caption.

**Dataset:** [`JamieSJS/imagecode-multi`](https://huggingface.co/datasets/JamieSJS/imagecode-multi) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.acl-long.241.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | image_acc | eng | Web, Written | derived | found |



#### SugarCrepe

Compositionality Evaluation of images to their captions.

**Dataset:** [`yjkimstats/SUGARCREPE_fmt`](https://huggingface.co/datasets/yjkimstats/SUGARCREPE_fmt) • **License:** mit • [Learn more →](https://proceedings.neurips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | text_acc | eng | Encyclopaedic | expert-annotated | created |



#### Winoground

Compositionality Evaluation of images to their captions.

**Dataset:** [`facebook/winoground`](https://huggingface.co/datasets/facebook/winoground) • **License:** https://huggingface.co/datasets/facebook/winoground/blob/main/license_agreement.txt • [Learn more →](https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Social | expert-annotated | created |



## DocumentUnderstanding

- **Number of tasks of the given type:** 58 

#### JinaVDRAirbnbSyntheticRetrieval

Retrieve rendered tables from Airbnb listings based on templated queries.

**Dataset:** [`jinaai/airbnb-synthetic-retrieval_beir`](https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, deu, eng, fra, hin, ... (10) | Web | derived | found |



#### JinaVDRArabicChartQARetrieval

Retrieve Arabic charts based on queries.

**Dataset:** [`jinaai/arabic_chartqa_ar_beir`](https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara | Academic | derived | found |



#### JinaVDRArabicInfographicsVQARetrieval

Retrieve Arabic infographics based on queries.

**Dataset:** [`jinaai/arabic_infographicsvqa_ar_beir`](https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara | Academic | derived | found |



#### JinaVDRArxivQARetrieval

Retrieve figures from scientific papers from arXiv based on LLM generated queries.

**Dataset:** [`jinaai/arxivqa_beir`](https://huggingface.co/datasets/jinaai/arxivqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/arxivqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | LM-generated | found |



#### JinaVDRAutomobileCatelogRetrieval

Retrieve automobile marketing documents based on LLM generated queries.

**Dataset:** [`jinaai/automobile_catalogue_jp_beir`](https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | jpn | Engineering, Web | LM-generated | found |



#### JinaVDRBeveragesCatalogueRetrieval

Retrieve beverages marketing documents based on LLM generated queries.

**Dataset:** [`jinaai/beverages_catalogue_ru_beir`](https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | rus | Web | LM-generated | found |



#### JinaVDRCharXivOCRRetrieval

Retrieve charts from scientific papers based on human annotated queries.

**Dataset:** [`jinaai/CharXiv-en_beir`](https://huggingface.co/datasets/jinaai/CharXiv-en_beir) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/CharXiv-en_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRChartQARetrieval

Retrieve charts based on LLM generated queries.

**Dataset:** [`jinaai/ChartQA_beir`](https://huggingface.co/datasets/jinaai/ChartQA_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/ChartQA_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRDocQAAI

Retrieve AI documents based on LLM generated queries.

**Dataset:** [`jinaai/docqa_artificial_intelligence_beir`](https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRDocQAEnergyRetrieval

Retrieve energy industry documents based on LLM generated queries.

**Dataset:** [`jinaai/docqa_energy_beir`](https://huggingface.co/datasets/jinaai/docqa_energy_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_energy_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRDocQAGovReportRetrieval

Retrieve government reports based on LLM generated queries.

**Dataset:** [`jinaai/docqa_gov_report_beir`](https://huggingface.co/datasets/jinaai/docqa_gov_report_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_gov_report_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Government | derived | found |



#### JinaVDRDocQAHealthcareIndustryRetrieval

Retrieve healthcare industry documents based on LLM generated queries.

**Dataset:** [`jinaai/docqa_healthcare_industry_beir`](https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Medical | derived | found |



#### JinaVDRDocVQARetrieval

Retrieve industry documents based on human annotated queries.

**Dataset:** [`jinaai/docvqa_beir`](https://huggingface.co/datasets/jinaai/docvqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/docvqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | LM-generated | found |



#### JinaVDRDonutVQAISynHMPRetrieval

Retrieve medical records based on templated queries.

**Dataset:** [`jinaai/donut_vqa_beir`](https://huggingface.co/datasets/jinaai/donut_vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/donut_vqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Medical | derived | found |



#### JinaVDREuropeanaDeNewsRetrieval

Retrieve German news articles based on LLM generated queries.

**Dataset:** [`jinaai/europeana-de-news_beir`](https://huggingface.co/datasets/jinaai/europeana-de-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-de-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu | News | LM-generated | found |



#### JinaVDREuropeanaEsNewsRetrieval

Retrieve Spanish news articles based on LLM generated queries.

**Dataset:** [`jinaai/europeana-es-news_beir`](https://huggingface.co/datasets/jinaai/europeana-es-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-es-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | spa | News | LM-generated | found |



#### JinaVDREuropeanaFrNewsRetrieval

Retrieve French news articles from Europeana based on LLM generated queries.

**Dataset:** [`jinaai/europeana-fr-news_beir`](https://huggingface.co/datasets/jinaai/europeana-fr-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-fr-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | fra | News | LM-generated | found |



#### JinaVDREuropeanaItScansRetrieval

Retrieve Italian historical articles based on LLM generated queries.

**Dataset:** [`jinaai/europeana-it-scans_beir`](https://huggingface.co/datasets/jinaai/europeana-it-scans_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-it-scans_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ita | News | LM-generated | found |



#### JinaVDREuropeanaNlLegalRetrieval

Retrieve Dutch historical legal documents based on LLM generated queries.

**Dataset:** [`jinaai/europeana-nl-legal_beir`](https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | nld | Legal | LM-generated | found |



#### JinaVDRGitHubReadmeRetrieval

Retrieve GitHub readme files based their description.

**Dataset:** [`jinaai/github-readme-retrieval-multilingual_beir`](https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, ben, deu, eng, fra, ... (17) | Web | derived | found |



#### JinaVDRHindiGovVQARetrieval

Retrieve Hindi government documents based on LLM generated queries.

**Dataset:** [`jinaai/hindi-gov-vqa_beir`](https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | hin | Government | LM-generated | found |



#### JinaVDRHungarianDocQARetrieval

Retrieve Hungarian documents in various formats based on human annotated queries.

**Dataset:** [`jinaai/hungarian_doc_qa_beir`](https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | hun | Web | derived | found |



#### JinaVDRInfovqaRetrieval

Retrieve infographics based on human annotated queries.

**Dataset:** [`jinaai/infovqa_beir`](https://huggingface.co/datasets/jinaai/infovqa_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/infovqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRJDocQARetrieval

Retrieve Japanese documents in various formats based on human annotated queries.

**Dataset:** [`jinaai/jdocqa_beir`](https://huggingface.co/datasets/jinaai/jdocqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/jdocqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | jpn | Web | LM-generated | found |



#### JinaVDRJina2024YearlyBookRetrieval

Retrieve pages from the 2024 Jina yearbook based on human annotated questions.

**Dataset:** [`jinaai/jina_2024_yearly_book_beir`](https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRMMTabRetrieval

Retrieve tables from the MMTab dataset based on queries.

**Dataset:** [`jinaai/MMTab_beir`](https://huggingface.co/datasets/jinaai/MMTab_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/MMTab_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRMPMQARetrieval

Retrieve product manuals based on human annotated queries.

**Dataset:** [`jinaai/mpmqa_small_beir`](https://huggingface.co/datasets/jinaai/mpmqa_small_beir) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/jinaai/mpmqa_small_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | human-annotated | found |



#### JinaVDRMedicalPrescriptionsRetrieval

Retrieve medical prescriptions based on templated queries.

**Dataset:** [`jinaai/medical-prescriptions_beir`](https://huggingface.co/datasets/jinaai/medical-prescriptions_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/medical-prescriptions_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Medical | derived | found |



#### JinaVDROWIDChartsRetrieval

Retrieve charts from the OWID dataset based on accompanied text snippets.

**Dataset:** [`jinaai/owid_charts_en_beir`](https://huggingface.co/datasets/jinaai/owid_charts_en_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/owid_charts_en_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDROpenAINewsRetrieval

Retrieve news articles from the OpenAI news website based on human annotated queries.

**Dataset:** [`jinaai/openai-news_beir`](https://huggingface.co/datasets/jinaai/openai-news_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/openai-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | News, Web | human-annotated | found |



#### JinaVDRPlotQARetrieval

Retrieve plots from the PlotQA dataset based on LLM generated queries.

**Dataset:** [`jinaai/plotqa_beir`](https://huggingface.co/datasets/jinaai/plotqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/plotqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRRamensBenchmarkRetrieval

Retrieve ramen restaurant marketing documents based on LLM generated queries.

**Dataset:** [`jinaai/ramen_benchmark_jp_beir`](https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | jpn | Web | LM-generated | found |



#### JinaVDRShanghaiMasterPlanRetrieval

Retrieve pages from the Shanghai Master Plan based on human annotated queries.

**Dataset:** [`jinaai/shanghai_master_plan_beir`](https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | zho | Web | human-annotated | found |



#### JinaVDRShiftProjectRetrieval

Retrieve documents with graphs from the Shift Project based on LLM generated queries.

**Dataset:** [`jinaai/shiftproject_beir`](https://huggingface.co/datasets/jinaai/shiftproject_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/shiftproject_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRStanfordSlideRetrieval

Retrieve scientific and engineering slides based on human annotated queries.

**Dataset:** [`jinaai/stanford_slide_beir`](https://huggingface.co/datasets/jinaai/stanford_slide_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/stanford_slide_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | human-annotated | found |



#### JinaVDRStudentEnrollmentSyntheticRetrieval

Retrieve student enrollment data based on templated queries.

**Dataset:** [`jinaai/student-enrollment_beir`](https://huggingface.co/datasets/jinaai/student-enrollment_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/student-enrollment_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### JinaVDRTQARetrieval

Retrieve textbook pages (images and text) based on LLM generated queries from the text.

**Dataset:** [`jinaai/tqa_beir`](https://huggingface.co/datasets/jinaai/tqa_beir) • **License:** cc-by-nc-3.0 • [Learn more →](https://huggingface.co/datasets/jinaai/tqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### JinaVDRTabFQuadRetrieval

Retrieve tables from industry documents based on LLM generated queries.

**Dataset:** [`jinaai/tabfquad_beir`](https://huggingface.co/datasets/jinaai/tabfquad_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/tabfquad_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### JinaVDRTableVQARetrieval

Retrieve scientific tables based on LLM generated queries.

**Dataset:** [`jinaai/table-vqa_beir`](https://huggingface.co/datasets/jinaai/table-vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/table-vqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### JinaVDRTatQARetrieval

Retrieve financial reports based on human annotated queries.

**Dataset:** [`jinaai/tatqa_beir`](https://huggingface.co/datasets/jinaai/tatqa_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/tatqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### JinaVDRTweetStockSyntheticsRetrieval

Retrieve rendered tables of stock prices based on templated queries.

**Dataset:** [`jinaai/tweet-stock-synthetic-retrieval_beir`](https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, deu, eng, fra, hin, ... (10) | Social | derived | found |



#### JinaVDRWikimediaCommonsDocumentsRetrieval

Retrieve historical documents from Wikimedia Commons based on their description.

**Dataset:** [`jinaai/wikimedia-commons-documents-ml_beir`](https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, ben, deu, eng, fra, ... (20) | Web | derived | found |



#### JinaVDRWikimediaCommonsMapsRetrieval

Retrieve maps from Wikimedia Commons based on their description.

**Dataset:** [`jinaai/wikimedia-commons-maps_beir`](https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



#### MIRACLVisionRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`nvidia/miracl-vision`](https://huggingface.co/datasets/nvidia/miracl-vision) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic | derived | created |



#### Vidore2BioMedicalLecturesRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/biomedical_lectures_v2`](https://huggingface.co/datasets/vidore/biomedical_lectures_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu, eng, fra, spa | Academic | derived | found |



#### Vidore2ESGReportsHLRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/esg_reports_human_labeled_v2`](https://huggingface.co/datasets/vidore/esg_reports_human_labeled_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### Vidore2ESGReportsRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/esg_reports_v2`](https://huggingface.co/datasets/vidore/esg_reports_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu, eng, fra, spa | Academic | derived | found |



#### Vidore2EconomicsReportsRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/economics_reports_v2`](https://huggingface.co/datasets/vidore/economics_reports_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu, eng, fra, spa | Academic | derived | found |



#### VidoreArxivQARetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/arxivqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/arxivqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreDocVQARetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/docvqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/docvqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreInfoVQARetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/infovqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreShiftProjectRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/shiftproject_test_beir`](https://huggingface.co/datasets/vidore/shiftproject_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreSyntheticDocQAAIRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_artificial_intelligence_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_artificial_intelligence_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreSyntheticDocQAEnergyRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_energy_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_energy_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreSyntheticDocQAGovernmentReportsRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_government_reports_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_government_reports_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreSyntheticDocQAHealthcareIndustryRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_healthcare_industry_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_healthcare_industry_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreTabfquadRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/tabfquad_test_subsampled_beir`](https://huggingface.co/datasets/vidore/tabfquad_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



#### VidoreTatdqaRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/tatdqa_test_beir`](https://huggingface.co/datasets/vidore/tatdqa_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



## ImageClassification

- **Number of tasks of the given type:** 22 

#### Birdsnap

Classifying bird images from 500 species.

**Dataset:** [`isaacchung/birdsnap`](https://huggingface.co/datasets/isaacchung/birdsnap) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2014/html/Berg_Birdsnap_Large-scale_Fine-grained_2014_CVPR_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### CIFAR10

Classifying images from 10 classes.

**Dataset:** [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Web | derived | created |



#### CIFAR100

Classifying images from 100 classes.

**Dataset:** [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar100)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Web | derived | created |



#### Caltech101

Classifying images of 101 widely varied objects.

**Dataset:** [`mteb/Caltech101`](https://huggingface.co/datasets/mteb/Caltech101) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/1384978)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### Country211

Classifying images of 211 countries.

**Dataset:** [`clip-benchmark/wds_country211`](https://huggingface.co/datasets/clip-benchmark/wds_country211) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clip-benchmark/wds_country211)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Scene | derived | created |



#### DTD

Describable Textures Dataset in 47 categories.

**Dataset:** [`tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd) • **License:** not specified • [Learn more →](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### EuroSAT

Classifying satellite images.

**Dataset:** [`timm/eurosat-rgb`](https://huggingface.co/datasets/timm/eurosat-rgb) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8736785)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### FER2013

Classifying facial emotions.

**Dataset:** [`clip-benchmark/wds_fer2013`](https://huggingface.co/datasets/clip-benchmark/wds_fer2013) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1412.6572)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### FGVCAircraft

Classifying aircraft images from 41 manufacturers and 102 variants.

**Dataset:** [`HuggingFaceM4/FGVC-Aircraft`](https://huggingface.co/datasets/HuggingFaceM4/FGVC-Aircraft) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1306.5151)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### Food101Classification

Classifying food.

**Dataset:** [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/ethz/food101)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Web | derived | created |



#### GTSRB

The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class classification dataset for traffic signs. It consists of dataset of more than 50,000 traffic sign images. The dataset comprises 43 classes with unbalanced class frequencies.

**Dataset:** [`clip-benchmark/wds_gtsrb`](https://huggingface.co/datasets/clip-benchmark/wds_gtsrb) • **License:** not specified • [Learn more →](https://benchmark.ini.rub.de/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Scene | derived | created |



#### Imagenet1k

ImageNet, a large-scale ontology of images built upon the backbone of the WordNet structure.

**Dataset:** [`clip-benchmark/wds_imagenet1k`](https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/5206848)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Scene | human-annotated | created |



#### MNIST

Classifying handwritten digits.

**Dataset:** [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist) • **License:** not specified • [Learn more →](https://en.wikipedia.org/wiki/MNIST_database)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### OxfordFlowersClassification

Classifying flowers

**Dataset:** [`nelorth/oxford-flowers`](https://huggingface.co/datasets/nelorth/oxford-flowers) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/nelorth/oxford-flowers/viewer/default/train)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Reviews | derived | found |



#### OxfordPets

Classifying animal images.

**Dataset:** [`isaacchung/OxfordPets`](https://huggingface.co/datasets/isaacchung/OxfordPets) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/abstract/document/6248092)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### PatchCamelyon

Histopathology diagnosis classification dataset.

**Dataset:** [`clip-benchmark/wds_vtab-pcam`](https://huggingface.co/datasets/clip-benchmark/wds_vtab-pcam) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Medical | derived | created |



#### RESISC45

Remote Sensing Image Scene Classification by Northwestern Polytechnical University (NWPU).

**Dataset:** [`timm/resisc45`](https://huggingface.co/datasets/timm/resisc45) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/abstract/document/7891544)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### STL10

Classifying 96x96 images from 10 classes.

**Dataset:** [`tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10) • **License:** not specified • [Learn more →](https://cs.stanford.edu/~acoates/stl10/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### SUN397

Large scale scene recognition in 397 categories.

**Dataset:** [`dpdl-benchmark/sun397`](https://huggingface.co/datasets/dpdl-benchmark/sun397) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/abstract/document/5539970)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### StanfordCars

Classifying car images from 196 makes.

**Dataset:** [`isaacchung/StanfordCars`](https://huggingface.co/datasets/isaacchung/StanfordCars) • **License:** not specified • [Learn more →](https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Encyclopaedic | derived | created |



#### UCF101

UCF101 is an action recognition data set of realistic
action videos collected from YouTube, having 101 action categories. This
version of the dataset does not contain images but images saved frame by
frame. Train and test splits are generated based on the authors' first
version train/test list.

**Dataset:** [`flwrlabs/ucf101`](https://huggingface.co/datasets/flwrlabs/ucf101) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/flwrlabs/ucf101)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | accuracy | eng | Scene | derived | created |



#### VOC2007

Classifying bird images from 500 species.

**Dataset:** [`HuggingFaceM4/pascal_voc`](https://huggingface.co/datasets/HuggingFaceM4/pascal_voc) • **License:** not specified • [Learn more →](http://host.robots.ox.ac.uk/pascal/VOC/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | lrap | eng | Encyclopaedic | derived | created |



## ImageClustering

- **Number of tasks of the given type:** 5 

#### CIFAR100Clustering

Clustering images from 100 classes.

**Dataset:** [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar100)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | nmi | eng | Web | derived | created |



#### CIFAR10Clustering

Clustering images from 10 classes.

**Dataset:** [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | nmi | eng | Web | derived | created |



#### ImageNet10Clustering

Clustering images from an 10-class subset of ImageNet which are generally easy to distinguish.

**Dataset:** [`JamieSJS/imagenet-10`](https://huggingface.co/datasets/JamieSJS/imagenet-10) • **License:** not specified • [Learn more →](https://www.kaggle.com/datasets/liusha249/imagenet10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | nmi | eng | Web | derived | created |



#### ImageNetDog15Clustering

Clustering images from a 15-class dogs-only subset of the dog classes in ImageNet.

**Dataset:** [`JamieSJS/imagenet-dog-15`](https://huggingface.co/datasets/JamieSJS/imagenet-dog-15) • **License:** not specified • [Learn more →](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | nmi | eng | Web | derived | created |



#### TinyImageNetClustering

Clustering over 200 classes.

**Dataset:** [`zh-plus/tiny-imagenet`](https://huggingface.co/datasets/zh-plus/tiny-imagenet) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/zh-plus/tiny-imagenet/viewer/default/valid)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to category (i2c) | nmi | eng | Reviews | derived | found |



## InstructionReranking

- **Number of tasks of the given type:** 5 

#### Core17InstructionRetrieval

Measuring retrieval instruction following ability on Core17 narratives for the FollowIR benchmark.

**Dataset:** [`jhu-clsp/core17-instructions-mteb`](https://huggingface.co/datasets/jhu-clsp/core17-instructions-mteb) • **License:** mit • [Learn more →](https://arxiv.org/abs/2403.15246)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | p-MRR | eng | News, Written | derived | found |



#### News21InstructionRetrieval

Measuring retrieval instruction following ability on News21 narratives for the FollowIR benchmark.

**Dataset:** [`jhu-clsp/news21-instructions-mteb`](https://huggingface.co/datasets/jhu-clsp/news21-instructions-mteb) • **License:** mit • [Learn more →](https://arxiv.org/abs/2403.15246)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | p-MRR | eng | News, Written | derived | found |



#### Robust04InstructionRetrieval

Measuring retrieval instruction following ability on Robust04 narratives for the FollowIR benchmark.

**Dataset:** [`jhu-clsp/robust04-instructions-mteb`](https://huggingface.co/datasets/jhu-clsp/robust04-instructions-mteb) • **License:** mit • [Learn more →](https://arxiv.org/abs/2403.15246)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | p-MRR | eng | News, Written | derived | found |



#### mFollowIR

This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages.

**Dataset:** [`jhu-clsp/mFollowIR-parquet-mteb`](https://huggingface.co/datasets/jhu-clsp/mFollowIR-parquet-mteb) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | p-MRR | fas, rus, zho | News, Written | expert-annotated | found |



#### mFollowIRCrossLingual

This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages with English queries/instructions.

**Dataset:** [`jhu-clsp/mFollowIR-cross-lingual-parquet-mteb`](https://huggingface.co/datasets/jhu-clsp/mFollowIR-cross-lingual-parquet-mteb) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | p-MRR | eng, fas, rus, zho | News, Written | expert-annotated | found |



## InstructionRetrieval

- **Number of tasks of the given type:** 8 

#### IFIRAila

Benchmark aila subset in aila within instruction following abilities. The instructions simulate lawyers' or legal assistants' nuanced queries to retrieve relevant legal documents. 

**Dataset:** [`if-ir/aila`](https://huggingface.co/datasets/if-ir/aila) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Legal, Written | human-annotated | found |



#### IFIRCds

Benchmark IFIR cds subset within instruction following abilities. The instructions simulate a doctor's nuanced queries to retrieve suitable clinical trails, treatment and diagnosis information. 

**Dataset:** [`if-ir/cds`](https://huggingface.co/datasets/if-ir/cds) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Medical, Written | human-annotated | found |



#### IFIRFiQA

Benchmark IFIR fiqa subset within instruction following abilities. The instructions simulate people's daily life queries to retrieve suitable financial suggestions. 

**Dataset:** [`if-ir/fiqa`](https://huggingface.co/datasets/if-ir/fiqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Financial, Written | human-annotated | created |



#### IFIRFire

Benchmark IFIR fire subset within instruction following abilities. The instructions simulate lawyers' or legal assistants' nuanced queries to retrieve relevant legal documents. 

**Dataset:** [`if-ir/fire`](https://huggingface.co/datasets/if-ir/fire) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Legal, Written | human-annotated | found |



#### IFIRNFCorpus

Benchmark IFIR nfcorpus subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature in the medical and biological domains. 

**Dataset:** [`if-ir/nfcorpus`](https://huggingface.co/datasets/if-ir/nfcorpus) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Academic, Medical, Written | human-annotated | found |



#### IFIRPm

Benchmark IFIR pm subset within instruction following abilities. The instructions simulate a doctor's nuanced queries to retrieve suitable clinical trails, treatment and diagnosis information. 

**Dataset:** [`if-ir/pm`](https://huggingface.co/datasets/if-ir/pm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Medical, Written | human-annotated | found |



#### IFIRScifact

Benchmark IFIR scifact_open subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature. 

**Dataset:** [`if-ir/scifact_open`](https://huggingface.co/datasets/if-ir/scifact_open) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Academic, Written | human-annotated | found |



#### InstructIR

A benchmark specifically designed to evaluate the instruction following ability in information retrieval models. Our approach focuses on user-aligned instructions tailored to each query instance, reflecting the diverse characteristics inherent in real-world search scenarios. **NOTE**: scores on this may differ unless you include instruction first, then "[SEP]" and then the query via redefining `combine_query_and_instruction` in your model.

**Dataset:** [`mteb/InstructIR-mteb`](https://huggingface.co/datasets/mteb/InstructIR-mteb) • **License:** mit • [Learn more →](https://github.com/kaistAI/InstructIR/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | robustness_at_10 | eng | Web | human-annotated | created |



## MultilabelClassification

- **Number of tasks of the given type:** 7 

#### BrazilianToxicTweetsClassification


        ToLD-Br is the biggest dataset for toxic tweets in Brazilian Portuguese, crowdsourced by 42 annotators selected from
        a pool of 129 volunteers. Annotators were selected aiming to create a plural group in terms of demographics (ethnicity,
        sexual orientation, age, gender). Each tweet was labeled by three annotators in 6 possible categories: LGBTQ+phobia,
        Xenophobia, Obscene, Insult, Misogyny and Racism.
        

**Dataset:** [`mteb/told-br`](https://huggingface.co/datasets/mteb/told-br) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/told-br)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | por | Constructed, Written | expert-annotated | found |



#### CEDRClassification

Classification of sentences by emotions, labeled into 5 categories (joy, sadness, surprise, fear, and anger).

**Dataset:** [`ai-forever/cedr-classification`](https://huggingface.co/datasets/ai-forever/cedr-classification) • **License:** apache-2.0 • [Learn more →](https://www.sciencedirect.com/science/article/pii/S1877050921013247)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Blog, Social, Web, Written | human-annotated | found |



#### EmitClassification

The EMit dataset is a comprehensive resource for the detection of emotions in Italian social media texts.
        The EMit dataset consists of social media messages about TV shows, TV series, music videos, and advertisements.
        Each message is annotated with one or more of the 8 primary emotions defined by Plutchik
        (anger, anticipation, disgust, fear, joy, sadness, surprise, trust), as well as an additional label “love.”
        

**Dataset:** [`MattiaSangermano/emit`](https://huggingface.co/datasets/MattiaSangermano/emit) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/oaraque/emit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Social, Written | expert-annotated | found |



#### KorHateSpeechMLClassification


        The Korean Multi-label Hate Speech Dataset, K-MHaS, consists of 109,692 utterances from Korean online news comments,
        labelled with 8 fine-grained hate speech classes (labels: Politics, Origin, Physical, Age, Gender, Religion, Race, Profanity)
        or Not Hate Speech class. Each utterance provides from a single to four labels that can handles Korean language patterns effectively.
        For more details, please refer to the paper about K-MHaS, published at COLING 2022.
        This dataset is based on the Korean online news comments available on Kaggle and Github.
        The unlabeled raw data was collected between January 2018 and June 2020.
        The language producers are users who left the comments on the Korean online news platform between 2018 and 2020.
        

**Dataset:** [`jeanlee/kmhas_korean_hate_speech`](https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-multi-label-hate-speech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### MalteseNewsClassification

A multi-label topic classification dataset for Maltese News
        Articles. The data was collected from the press_mt subset from Korpus
        Malti v4.0. Article contents were cleaned to filter out JavaScript, CSS,
        & repeated non-Maltese sub-headings. The labels are based on the category
        field from this corpus.
        

**Dataset:** [`MLRS/maltese_news_categories`](https://huggingface.co/datasets/MLRS/maltese_news_categories) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/MLRS/maltese_news_categories)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mlt | Constructed, Written | expert-annotated | found |



#### MultiEURLEXMultilabelClassification

EU laws in 23 EU languages containing annotated labels for 21 EUROVOC concepts.

**Dataset:** [`mteb/eurlex-multilingual`](https://huggingface.co/datasets/mteb/eurlex-multilingual) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/coastalcph/multi_eurlex)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bul, ces, dan, deu, ell, ... (23) | Government, Legal, Written | expert-annotated | found |



#### SensitiveTopicsClassification

Multilabel classification of sentences across 18 sensitive topics.

**Dataset:** [`ai-forever/sensitive-topics-classification`](https://huggingface.co/datasets/ai-forever/sensitive-topics-classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |



## PairClassification

- **Number of tasks of the given type:** 44 

#### ArEntail

A manually-curated Arabic natural language inference dataset from news headlines.

**Dataset:** [`arbml/ArEntail`](https://huggingface.co/datasets/arbml/ArEntail) • **License:** not specified • [Learn more →](https://link.springer.com/article/10.1007/s10579-024-09731-1)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | ara | News, Written | human-annotated | found |



#### ArmenianParaphrasePC

asparius/Armenian-Paraphrase-PC

**Dataset:** [`asparius/Armenian-Paraphrase-PC`](https://huggingface.co/datasets/asparius/Armenian-Paraphrase-PC) • **License:** apache-2.0 • [Learn more →](https://github.com/ivannikov-lab/arpa-paraphrase-corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | hye | News, Written | derived | found |



#### Assin2RTE

Recognizing Textual Entailment part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.

**Dataset:** [`nilc-nlp/assin2`](https://huggingface.co/datasets/nilc-nlp/assin2) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | por | Written | human-annotated | found |



#### CDSC-E

Compositional Distributional Semantics Corpus for textual entailment.

**Dataset:** [`PL-MTEB/cdsce-pairclassification`](https://huggingface.co/datasets/PL-MTEB/cdsce-pairclassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/P17-1073.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | pol | Written | human-annotated | found |



#### CExaPPC

ExaPPC is a large paraphrase corpus consisting of monolingual sentence-level paraphrases using different sources.

**Dataset:** [`PNLPhub/C-ExaPPC`](https://huggingface.co/datasets/PNLPhub/C-ExaPPC) • **License:** not specified • [Learn more →](https://github.com/exaco/exappc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Social, Web | derived | found |



#### CTKFactsNLI

Czech Natural Language Inference dataset of around 3K evidence-claim pairs labelled with SUPPORTS, REFUTES or NOT ENOUGH INFO veracity labels. Extracted from a round of fact-checking experiments.

**Dataset:** [`ctu-aic/ctkfacts_nli`](https://huggingface.co/datasets/ctu-aic/ctkfacts_nli) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/2201.11115)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | ces | News, Written | human-annotated | found |



#### Cmnli

Chinese Multi-Genre NLI

**Dataset:** [`C-MTEB/CMNLI`](https://huggingface.co/datasets/C-MTEB/CMNLI) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clue/viewer/cmnli)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_accuracy | cmn | not specified | not specified | not specified |



#### DisCoTexPairClassification

The DisCoTEX dataset aims at assessing discourse coherence in Italian texts. This dataset focuses on Italian real-world texts and provides resources to model coherence in natural language.

**Dataset:** [`MattiaSangermano/DisCoTex-last-sentence`](https://huggingface.co/datasets/MattiaSangermano/DisCoTex-last-sentence) • **License:** not specified • [Learn more →](https://github.com/davidecolla/DisCoTex)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | ita | Social, Written | derived | found |



#### FalseFriendsGermanEnglish

A dataset to identify False Friends / false cognates between English and German. A generally challenging task for multilingual models.

**Dataset:** [`aari1995/false_friends_de_en_mteb`](https://huggingface.co/datasets/aari1995/false_friends_de_en_mteb) • **License:** mit • [Learn more →](https://drive.google.com/file/d/1jgq0nBnV-UiYNxbKNrrr2gxDEHm-DMKH/view?usp=share_link)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | deu | Written | human-annotated | created |



#### FarsTail

This dataset, named FarsTail, includes 10,367 samples which are provided in both the Persian language as well as the indexed format to be useful for non-Persian researchers. The samples are generated from 3,539 multiple-choice questions with the least amount of annotator interventions in a way similar to the SciTail dataset

**Dataset:** [`azarijafari/FarsTail`](https://huggingface.co/datasets/azarijafari/FarsTail) • **License:** not specified • [Learn more →](https://link.springer.com/article/10.1007/s00500-023-08959-3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Academic, Written | human-annotated | found |



#### FarsiParaphraseDetection

Farsi Paraphrase Detection

**Dataset:** [`alighasemi/farsi_paraphrase_detection`](https://huggingface.co/datasets/alighasemi/farsi_paraphrase_detection) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/alighasemi/farsi_paraphrase_detection)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | not specified | derived | found |



#### IndicXnliPairClassification

INDICXNLI is similar to existing XNLI dataset in shape/form, but
        focusses on Indic language family.
        The train (392,702), validation (2,490), and evaluation sets (5,010) of English
        XNLI were translated from English into each of the eleven Indic languages. IndicTrans
        is a large Transformer-based sequence to sequence model. It is trained on Samanantar
        dataset (Ramesh et al., 2021), which is the largest parallel multi- lingual corpus
        over eleven Indic languages.
        

**Dataset:** [`Divyanshu/indicxnli`](https://huggingface.co/datasets/Divyanshu/indicxnli) • **License:** cc-by-4.0 • [Learn more →](https://gem-benchmark.com/data_cards/opusparcus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | asm, ben, guj, hin, kan, ... (11) | Fiction, Government, Non-fiction, Written | derived | machine-translated |



#### KLUE-NLI

Textual Entailment between a hypothesis sentence and a premise sentence. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | kor | Encyclopaedic, News, Written | human-annotated | found |



#### LegalBenchPC

This LegalBench pair classification task is a combination of the following datasets:

        - Citation Prediction Classification: Given a legal statement and a case citation, determine if the citation is supportive of the legal statement.
        - Consumer Contracts QA: The task consists of 400 yes/no questions relating to consumer contracts (specifically, online terms of service) and is relevant to the legal skill of contract interpretation.
        - Contract QA: Answer yes/no questions about whether contractual clauses discuss particular issues like confidentiality requirements, BIPA consent, PII data breaches, breach of contract etc.
        - Hearsay: Classify if a particular piece of evidence qualifies as hearsay. Each sample in the dataset describes (1) an issue being litigated or an assertion a party wishes to prove, and (2) a piece of evidence a party wishes to introduce. The goal is to determine if—as it relates to the issue—the evidence would be considered hearsay under the definition provided above.
        - Privacy Policy Entailment: Given a privacy policy clause and a description of the clause, determine if the description is correct. This is a binary classification task in which the LLM is provided with a clause from a privacy policy, and a description of that clause (e.g., “The policy describes collection of the user’s HTTP cookies, flash cookies, pixel tags, or similar identifiers by a party to the contract.”).
        - Privacy Policy QA: Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question. This is a binary classification task in which the LLM is provided with a question (e.g., “do you publish my data”) and a clause from a privacy policy. The LLM must determine if the clause contains an answer to the question, and classify the question-clause pair.
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_accuracy | eng | Legal, Written | expert-annotated | found |



#### Ocnli

Original Chinese Natural Language Inference dataset

**Dataset:** [`C-MTEB/OCNLI`](https://huggingface.co/datasets/C-MTEB/OCNLI) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2010.05444)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_accuracy | cmn | not specified | not specified | not specified |



#### OpusparcusPC

Opusparcus is a paraphrase corpus for six European language: German, English, Finnish, French, Russian, and Swedish. The paraphrases consist of subtitles from movies and TV shows.

**Dataset:** [`GEM/opusparcus`](https://huggingface.co/datasets/GEM/opusparcus) • **License:** cc-by-nc-4.0 • [Learn more →](https://gem-benchmark.com/data_cards/opusparcus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | deu, eng, fin, fra, rus, ... (6) | Spoken, Spoken | human-annotated | created |



#### PSC

Polish Summaries Corpus

**Dataset:** [`PL-MTEB/psc-pairclassification`](https://huggingface.co/datasets/PL-MTEB/psc-pairclassification) • **License:** cc-by-3.0 • [Learn more →](http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | pol | News, Written | derived | found |



#### ParsinluEntail

A Persian textual entailment task (deciding sent1 entails sent2). The questions are partially translated from the SNLI dataset and partially generated by expert annotators.

**Dataset:** [`persiannlp/parsinlu_entailment`](https://huggingface.co/datasets/persiannlp/parsinlu_entailment) • **License:** not specified • [Learn more →](https://github.com/persiannlp/parsinlu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Reviews, Written | derived | found |



#### ParsinluQueryParaphPC

A Persian query paraphrasng task (deciding whether two questions are paraphrases of each other). The questions are partially generated from Google auto-complete, and partially translated from the Quora paraphrasing dataset.

**Dataset:** [`persiannlp/parsinlu_query_paraphrasing`](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Reviews, Written | derived | found |



#### PawsXPairClassification

{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification

**Dataset:** [`google-research-datasets/paws-x`](https://huggingface.co/datasets/google-research-datasets/paws-x) • **License:** https://huggingface.co/datasets/google-research-datasets/paws-x#licensing-information • [Learn more →](https://arxiv.org/abs/1908.11828)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | cmn, deu, eng, fra, jpn, ... (7) | Encyclopaedic, Web, Written | human-annotated | human-translated |



#### PpcPC

Polish Paraphrase Corpus

**Dataset:** [`PL-MTEB/ppc-pairclassification`](https://huggingface.co/datasets/PL-MTEB/ppc-pairclassification) • **License:** gpl-3.0 • [Learn more →](https://arxiv.org/pdf/2207.12759.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | pol | Fiction, News, Non-fiction, Social, Spoken, ... (7) | derived | found |



#### PubChemAISentenceParaphrasePC

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemAISentenceParaphrasePC`](https://huggingface.co/datasets/BASF-AI/PubChemAISentenceParaphrasePC) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Chemistry | LM-generated | created |



#### PubChemSMILESPC

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemSMILESPairClassification`](https://huggingface.co/datasets/BASF-AI/PubChemSMILESPairClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Chemistry | derived | created |



#### PubChemSynonymPC

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemSynonymPC`](https://huggingface.co/datasets/BASF-AI/PubChemSynonymPC) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Chemistry | derived | created |



#### PubChemWikiPairClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemWikiMultilingualPC`](https://huggingface.co/datasets/BASF-AI/PubChemWikiMultilingualPC) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | ces, deu, eng, fra, hin, ... (13) | Chemistry | derived | created |



#### PubChemWikiParagraphsPC

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemWikiParagraphsPC`](https://huggingface.co/datasets/BASF-AI/PubChemWikiParagraphsPC) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Chemistry | derived | created |



#### RTE3

Recognising Textual Entailment Challenge (RTE-3) aim to provide the NLP community with a benchmark to test progress in recognizing textual entailment

**Dataset:** [`maximoss/rte3-multi`](https://huggingface.co/datasets/maximoss/rte3-multi) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/W07-1401/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | deu, eng, fra, ita | Encyclopaedic, News, Web, Written | expert-annotated | found |



#### SICK-BR-PC

SICK-BR is a Portuguese inference corpus, human translated from SICK

**Dataset:** [`eduagarcia/sick-br`](https://huggingface.co/datasets/eduagarcia/sick-br) • **License:** not specified • [Learn more →](https://linux.ime.usp.br/~thalen/SICK_PT.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | por | Web, Written | human-annotated | human-translated and localized |



#### SICK-E-PL

Polish version of SICK dataset for textual entailment.

**Dataset:** [`PL-MTEB/sicke-pl-pairclassification`](https://huggingface.co/datasets/PL-MTEB/sicke-pl-pairclassification) • **License:** not specified • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | pol | Reviews | not specified | not specified |



#### SprintDuplicateQuestions

Duplicate questions from the Sprint community.

**Dataset:** [`mteb/sprintduplicatequestions-pairclassification`](https://huggingface.co/datasets/mteb/sprintduplicatequestions-pairclassification) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1131/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Programming, Written | derived | found |



#### SprintDuplicateQuestions-VN

A translated dataset from Duplicate questions from the Sprint community.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/sprintduplicatequestions-pairclassification-vn`](https://huggingface.co/datasets/GreenNode/sprintduplicatequestions-pairclassification-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.aclweb.org/anthology/D18-1131/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | max_ap | vie | Programming, Written | derived | machine-translated and LM verified |



#### SynPerChatbotRAGFAQPC

Synthetic Persian Chatbot RAG FAQ Pair Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-faq-pair-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-pair-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerQAPC

Synthetic Persian QA Pair Classification

**Dataset:** [`MCINext/synthetic-persian-qa-pair-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-pair-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Blog, News, Religious, Web | LM-generated | LM-generated and verified |



#### SynPerTextKeywordsPC

Synthetic Persian Text Keywords Pair Classification

**Dataset:** [`MCINext/synthetic-persian-text-keyword-pair-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-text-keyword-pair-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | fas | Blog, News, Religious, Web | LM-generated | LM-generated and verified |



#### TERRa

Textual Entailment Recognition for Russian. This task requires to recognize, given two text fragments, whether the meaning of one text is entailed (can be inferred) from the other text.

**Dataset:** [`ai-forever/terra-pairclassification`](https://huggingface.co/datasets/ai-forever/terra-pairclassification) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2010.15925)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | rus | News, Web, Written | human-annotated | found |



#### TalemaaderPC

The Danish Language and Literature Society has developed a dataset for evaluating language models in Danish.
The dataset contains a total of 1000 Danish idioms and fixed expressions with transferred meanings based on the Danish Dictionary's collection of fixed expressions with associated definitions.
For each of the 1000 idioms and fixed expressions, three false definitions have also been prepared.
The dataset can be used to test the performance of language models in identifying correct definitions for Danish idioms and fixed expressions.


**Dataset:** [`mteb/talemaader_pc`](https://huggingface.co/datasets/mteb/talemaader_pc) • **License:** cc-by-4.0 • [Learn more →](https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_accuracy | dan | Academic, Written | derived | created |



#### TwitterSemEval2015

Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.

**Dataset:** [`mteb/twittersemeval2015-pairclassification`](https://huggingface.co/datasets/mteb/twittersemeval2015-pairclassification) • **License:** not specified • [Learn more →](https://alt.qcri.org/semeval2015/task1/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Social, Written | human-annotated | found |



#### TwitterSemEval2015-VN

A translated dataset from Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/twittersemeval2015-pairclassification-vn`](https://huggingface.co/datasets/GreenNode/twittersemeval2015-pairclassification-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://alt.qcri.org/semeval2015/task1/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | max_ap | vie | Social, Written | derived | machine-translated and LM verified |



#### TwitterURLCorpus

Paraphrase-Pairs of Tweets.

**Dataset:** [`mteb/twitterurlcorpus-pairclassification`](https://huggingface.co/datasets/mteb/twitterurlcorpus-pairclassification) • **License:** not specified • [Learn more →](https://languagenet.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | eng | Social, Written | derived | found |



#### TwitterURLCorpus-VN

A translated dataset from Paraphrase-Pairs of Tweets.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/twitterurlcorpus-pairclassification-vn`](https://huggingface.co/datasets/GreenNode/twitterurlcorpus-pairclassification-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://languagenet.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | max_ap | vie | Social, Written | derived | machine-translated and LM verified |



#### XNLI



**Dataset:** [`mteb/xnli`](https://huggingface.co/datasets/mteb/xnli) • **License:** not specified • [Learn more →](https://aclanthology.org/D18-1269/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | ara, bul, deu, ell, eng, ... (14) | Fiction, Government, Non-fiction, Written | expert-annotated | created |



#### XNLIV2

This is subset of 'XNLI 2.0: Improving XNLI dataset and performance on Cross Lingual Understanding' with languages that were not part of the original XNLI plus three (verified) languages that are not strongly covered in MTEB

**Dataset:** [`mteb/XNLIV2`](https://huggingface.co/datasets/mteb/XNLIV2) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2301.06527)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | asm, ben, bho, ell, guj, ... (13) | Fiction, Government, Non-fiction, Written | expert-annotated | machine-translated and verified |



#### XStance

A Multilingual Multi-Target Dataset for Stance Detection in French, German, and Italian.

**Dataset:** [`ZurichNLP/x_stance`](https://huggingface.co/datasets/ZurichNLP/x_stance) • **License:** cc-by-nc-4.0 • [Learn more →](https://github.com/ZurichNLP/xstance)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | deu, fra, ita | Social, Written | human-annotated | created |



#### indonli

IndoNLI is the first human-elicited Natural Language Inference (NLI) dataset for Indonesian. IndoNLI is annotated by both crowd workers and experts.

**Dataset:** [`afaji/indonli`](https://huggingface.co/datasets/afaji/indonli) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_ap | ind | Encyclopaedic, News, Web, Written | expert-annotated | found |



## Regression

- **Number of tasks of the given type:** 2 

#### RuSciBenchCitedCountRegression

Predicts the number of times a scientific article has been cited by other papers.
        The prediction is based on the article's title and abstract. The data is sourced from the Russian electronic
        library of scientific publications (eLibrary.ru) and includes papers with both Russian and English abstracts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | kendalltau | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchYearPublRegression

Predicts the publication year of a scientific article. The prediction is based on the
        article's title and abstract. The data is sourced from the Russian electronic library of scientific
        publications (eLibrary.ru) and includes papers with both Russian and English abstracts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | kendalltau | eng, rus | Academic, Non-fiction, Written | derived | found |



## Reranking

- **Number of tasks of the given type:** 29 

#### AlloprofReranking

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`mteb/AlloprofReranking`](https://huggingface.co/datasets/mteb/AlloprofReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | fra | Academic, Web, Written | expert-annotated | found |



#### AskUbuntuDupQuestions

AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar

**Dataset:** [`mteb/AskUbuntuDupQuestions`](https://huggingface.co/datasets/mteb/AskUbuntuDupQuestions) • **License:** not specified • [Learn more →](https://github.com/taolei87/askubuntu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Programming, Web | human-annotated | found |



#### AskUbuntuDupQuestions-VN

A translated dataset from AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/askubuntudupquestions-reranking-vn`](https://huggingface.co/datasets/GreenNode/askubuntudupquestions-reranking-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/taolei87/askubuntu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | map | vie | Programming, Web | derived | machine-translated and LM verified |



#### BuiltBenchReranking

Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchReranking`](https://huggingface.co/datasets/mteb/BuiltBenchReranking) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | eng | Engineering, Written | derived | created |



#### CMedQAv1-reranking

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv1-reranking`](https://huggingface.co/datasets/mteb/CMedQAv1-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | Medical, Written | expert-annotated | found |



#### CMedQAv2-reranking

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv2-reranking`](https://huggingface.co/datasets/mteb/CMedQAv2-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | Medical, Written | not specified | not specified |



#### CodeRAGLibraryDocumentationSolutions

Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.

**Dataset:** [`code-rag-bench/library-documentation`](https://huggingface.co/datasets/code-rag-bench/library-documentation) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### CodeRAGOnlineTutorials

Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.

**Dataset:** [`code-rag-bench/online-tutorials`](https://huggingface.co/datasets/code-rag-bench/online-tutorials) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### CodeRAGProgrammingSolutions

Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.

**Dataset:** [`code-rag-bench/programming-solutions`](https://huggingface.co/datasets/code-rag-bench/programming-solutions) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### CodeRAGStackoverflowPosts

Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.

**Dataset:** [`code-rag-bench/stackoverflow-posts`](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### ESCIReranking



**Dataset:** [`mteb/ESCIReranking`](https://huggingface.co/datasets/mteb/ESCIReranking) • **License:** apache-2.0 • [Learn more →](https://github.com/amazon-science/esci-data/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng, jpn, spa | Written | derived | created |



#### JQaRAReranking

JQaRA: Japanese Question Answering with Retrieval Augmentation  - 検索拡張(RAG)評価のための日本語 Q&A データセット. JQaRA is an information retrieval task for questions against 100 candidate data (including one or more correct answers).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/hotchpotch/JQaRA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | jpn | Encyclopaedic, Non-fiction, Written | derived | found |



#### JaCWIRReranking

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of 5000 question texts and approximately 500k web page titles and web page introductions or summaries (meta descriptions, etc.). The question texts are created based on one of the 500k web pages, and that data is used as a positive example for the question text.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | jpn | Web, Written | derived | found |



#### MIRACLReranking

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`mteb/MIRACLReranking`](https://huggingface.co/datasets/mteb/MIRACLReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://project-miracl.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



#### MMarcoReranking

mMARCO is a multilingual version of the MS MARCO passage ranking dataset

**Dataset:** [`mteb/MMarcoReranking`](https://huggingface.co/datasets/mteb/MMarcoReranking) • **License:** not specified • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | not specified | not specified | not specified |



#### MindSmallReranking

Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research

**Dataset:** [`mteb/MindSmallReranking`](https://huggingface.co/datasets/mteb/MindSmallReranking) • **License:** https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf • [Learn more →](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_over_subqueries_map_at_1000 | eng | News, Written | expert-annotated | found |



#### NamaaMrTydiReranking

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations. This dataset adapts the arabic test split for Reranking evaluation purposes by the addition of multiple (Hard) Negatives to each query and positive

**Dataset:** [`mteb/NamaaMrTydiReranking`](https://huggingface.co/datasets/mteb/NamaaMrTydiReranking) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/NAMAA-Space)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | ara | Encyclopaedic, Written | human-annotated | found |



#### NevIR

Paired evaluation of real world negation in retrieval, with questions and passages. Since models generally prefer one passage over the other always, there are two questions that the model must get right to understand the negation (hence the `paired_accuracy` metric).

**Dataset:** [`orionweller/NevIR-mteb`](https://huggingface.co/datasets/orionweller/NevIR-mteb) • **License:** mit • [Learn more →](https://github.com/orionw/NevIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | paired_accuracy | eng | Web | human-annotated | created |



#### RuBQReranking

Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.

**Dataset:** [`mteb/RuBQReranking`](https://huggingface.co/datasets/mteb/RuBQReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | rus | Encyclopaedic, Written | human-annotated | created |



#### SciDocsRR

Ranking of related scientific papers based on their title.

**Dataset:** [`mteb/SciDocsRR`](https://huggingface.co/datasets/mteb/SciDocsRR) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Academic, Non-fiction, Written | not specified | found |



#### SciDocsRR-VN

A translated dataset from Ranking of related scientific papers based on their title.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scidocs-reranking-vn`](https://huggingface.co/datasets/GreenNode/scidocs-reranking-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | map | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### StackOverflowDupQuestions

Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python

**Dataset:** [`mteb/StackOverflowDupQuestions`](https://huggingface.co/datasets/mteb/StackOverflowDupQuestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Blog, Programming, Written | derived | found |



#### StackOverflowDupQuestions-VN

A translated dataset from Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stackoverflowdupquestions-reranking-vn`](https://huggingface.co/datasets/GreenNode/stackoverflowdupquestions-reranking-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | map | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### SyntecReranking

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`mteb/SyntecReranking`](https://huggingface.co/datasets/mteb/SyntecReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | fra | Legal, Written | human-annotated | found |



#### T2Reranking

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Reranking`](https://huggingface.co/datasets/mteb/T2Reranking) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2304.03679)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | not specified | not specified | not specified |



#### VoyageMMarcoReranking

a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite

**Dataset:** [`mteb/VoyageMMarcoReranking`](https://huggingface.co/datasets/mteb/VoyageMMarcoReranking) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2312.16144)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | jpn | Academic, Non-fiction, Written | derived | found |



#### WebLINXCandidatesReranking

WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.

**Dataset:** [`mteb/WebLINXCandidatesReranking`](https://huggingface.co/datasets/mteb/WebLINXCandidatesReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/weblinx)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | mrr_at_10 | eng | Academic, Web, Written | expert-annotated | created |



#### WikipediaRerankingMultilingual

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRerankingMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRerankingMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-multilingual)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | ben, bul, ces, dan, deu, ... (16) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified |



#### XGlueWPRReranking

XGLUE is a new benchmark dataset to evaluate the performance of cross-lingual pre-trained models
        with respect to cross-lingual natural language understanding and generation. XGLUE is composed of 11 tasks spans 19 languages.

**Dataset:** [`forresty/xglue`](https://huggingface.co/datasets/forresty/xglue) • **License:** http://hdl.handle.net/11234/1-3105 • [Learn more →](https://github.com/microsoft/XGLUE)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | deu, eng, fra, ita, por, ... (7) | Written | human-annotated | found |



## Retrieval

- **Number of tasks of the given type:** 326 

#### AILACasedocs

The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.

**Dataset:** [`mteb/AILA_casedocs`](https://huggingface.co/datasets/mteb/AILA_casedocs) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### AILAStatutes

This dataset is structured for the task of identifying the most relevant statutes for a given situation.

**Dataset:** [`mteb/AILA_statutes`](https://huggingface.co/datasets/mteb/AILA_statutes) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### ARCChallenge

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.

**Dataset:** [`RAR-b/ARC-Challenge`](https://huggingface.co/datasets/RAR-b/ARC-Challenge) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/arc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### AlloprofRetrieval

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Encyclopaedic, Written | human-annotated | found |



#### AlphaNLI

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on AlphaNLI.

**Dataset:** [`RAR-b/alphanli`](https://huggingface.co/datasets/RAR-b/alphanli) • **License:** cc-by-nc-4.0 • [Learn more →](https://leaderboard.allenai.org/anli/submissions/get-started)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### AppsRetrieval

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.09938)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming, Written | derived | found |



#### ArguAna

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/arguana`](https://huggingface.co/datasets/mteb/arguana) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | not specified | not specified |



#### ArguAna-Fa

ArguAna-Fa

**Dataset:** [`MCINext/arguana-fa`](https://huggingface.co/datasets/MCINext/arguana-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/arguana-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Blog | derived | found |



#### ArguAna-NL

ArguAna involves the task of retrieval of the best counterargument to an argument. ArguAna-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-arguana`](https://huggingface.co/datasets/clips/beir-nl-arguana) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-arguana)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### ArguAna-PL

ArguAna-PL

**Dataset:** [`mteb/ArguAna-PL`](https://huggingface.co/datasets/mteb/ArguAna-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clarin-knext/arguana-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Medical, Written | not specified | not specified |



#### ArguAna-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/arguana-vn`](https://huggingface.co/datasets/GreenNode/arguana-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Medical, Written | derived | machine-translated and LM verified |



#### AutoRAGRetrieval

This dataset enables the evaluation of Korean RAG performance across various domains—finance, public sector, healthcare, legal, and commerce—by providing publicly accessible documents, questions, and answers.

**Dataset:** [`yjoonjang/markers_bm`](https://huggingface.co/datasets/yjoonjang/markers_bm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2410.20878)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kor | Financial, Government, Legal, Medical, Social | human-annotated | created |



#### BIRCO-ArguAna

Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument that directly refutes the query’s stance.

**Dataset:** [`mteb/BIRCO-ArguAna-Test`](https://huggingface.co/datasets/mteb/BIRCO-ArguAna-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Written | expert-annotated | found |



#### BIRCO-ClinicalTrial

Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant.

**Dataset:** [`mteb/BIRCO-ClinicalTrial-Test`](https://huggingface.co/datasets/mteb/BIRCO-ClinicalTrial-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | expert-annotated | found |



#### BIRCO-DorisMae

Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries that are complex research questions from computer scientists. Each query has a candidate pool of approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant).

**Dataset:** [`mteb/BIRCO-DorisMae-Test`](https://huggingface.co/datasets/mteb/BIRCO-DorisMae-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | expert-annotated | found |



#### BIRCO-Relic

Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries which are excerpts from literary analyses with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. The objective is to retrieve the passage that best completes the literary analysis.

**Dataset:** [`mteb/BIRCO-Relic-Test`](https://huggingface.co/datasets/mteb/BIRCO-Relic-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction | expert-annotated | found |



#### BIRCO-WTB

Retrieval task using the WhatsThatBook dataset from BIRCO. This dataset contains 100 queries where each query is an ambiguous description of a book. Each query has a candidate pool of 50 book descriptions. The objective is to retrieve the correct book description.

**Dataset:** [`mteb/BIRCO-WTB-Test`](https://huggingface.co/datasets/mteb/BIRCO-WTB-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction | expert-annotated | found |



#### BSARDRetrieval

The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.

**Dataset:** [`maastrichtlawtech/bsard`](https://huggingface.co/datasets/maastrichtlawtech/bsard) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_100 | fra | Legal, Spoken | expert-annotated | found |



#### BSARDRetrieval.v2

BSARD is a French native dataset for legal information retrieval. BSARDRetrieval.v2 covers multi-article queries, fixing issues (#2906) with the previous data loading. 

**Dataset:** [`maastrichtlawtech/bsard`](https://huggingface.co/datasets/maastrichtlawtech/bsard) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | recall_at_100 | fra | Legal, Spoken | expert-annotated | found |



#### BarExamQA

A benchmark for retrieving legal provisions that answer US bar exam questions.

**Dataset:** [`isaacus/mteb-barexam-qa`](https://huggingface.co/datasets/isaacus/mteb-barexam-qa) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/reglab/barexam_qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Legal | expert-annotated | found |



#### BelebeleRetrieval

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants (including 115 distinct languages and their scripts)

**Dataset:** [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2308.16884)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | acm, afr, als, amh, apc, ... (115) | News, Web, Written | expert-annotated | created |



#### BillSumCA

A benchmark for retrieving Californian bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumCA`](https://huggingface.co/datasets/isaacus/mteb-BillSumCA) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



#### BillSumUS

A benchmark for retrieving US federal bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumUS`](https://huggingface.co/datasets/isaacus/mteb-BillSumUS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



#### BrightLongRetrieval

Bright retrieval dataset with long documents.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### BrightRetrieval

Bright retrieval dataset.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### BuiltBenchRetrieval

Retrieval of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchRetrieval`](https://huggingface.co/datasets/mteb/BuiltBenchRetrieval) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Engineering, Written | derived | created |



#### COIRCodeSearchNetRetrieval

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



#### CQADupstack-Android-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Android-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Android-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-android-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Programming, Web, Written | derived | machine-translated |



#### CQADupstack-English-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-English-PL`](https://huggingface.co/datasets/mteb/CQADupstack-English-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-english-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Written | derived | machine-translated |



#### CQADupstack-Gaming-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gaming-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gaming-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gaming-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### CQADupstack-Gis-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gis-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gis-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gis-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Mathematica-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Mathematica-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Mathematica-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-mathematica-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Physics-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Physics-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Physics-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-physics-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Programmers-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Programmers-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Programmers-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-programmers-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Programming, Written | derived | machine-translated |



#### CQADupstack-Stats-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Stats-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Stats-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-stats-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Tex-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Tex-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Tex-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-tex-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Unix-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Unix-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Unix-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-unix-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Programming, Web, Written | derived | machine-translated |



#### CQADupstack-Webmasters-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Webmasters-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Webmasters-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-webmasters-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### CQADupstack-Wordpress-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Wordpress-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Wordpress-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-wordpress-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Programming, Web, Written | derived | machine-translated |



#### CQADupstackAndroid-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackAndroid-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-android-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-android-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Programming, Web, Written | derived | machine-translated and LM verified |



#### CQADupstackAndroidRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-android`](https://huggingface.co/datasets/mteb/cqadupstack-android) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Programming, Web, Written | derived | found |



#### CQADupstackAndroidRetrieval-Fa

CQADupstackAndroidRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-android-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackEnglish-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackEnglishRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-english`](https://huggingface.co/datasets/mteb/cqadupstack-english) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Written | derived | found |



#### CQADupstackEnglishRetrieval-Fa

CQADupstackEnglishRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-english-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackGaming-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackGamingRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gaming`](https://huggingface.co/datasets/mteb/cqadupstack-gaming) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | derived | found |



#### CQADupstackGamingRetrieval-Fa

CQADupstackGamingRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gaming-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackGis-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackGis-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-gis-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-gis-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackGisRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gis`](https://huggingface.co/datasets/mteb/cqadupstack-gis) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### CQADupstackGisRetrieval-Fa

CQADupstackGisRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gis-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackMathematica-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackMathematica-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-mathematica-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-mathematica-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackMathematicaRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-mathematica`](https://huggingface.co/datasets/mteb/cqadupstack-mathematica) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



#### CQADupstackMathematicaRetrieval-Fa

CQADupstackMathematicaRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-mathematica-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackPhysics-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackPhysics-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-physics-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-physics-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackPhysicsRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-physics`](https://huggingface.co/datasets/mteb/cqadupstack-physics) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



#### CQADupstackPhysicsRetrieval-Fa

CQADupstackPhysicsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-physics-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackProgrammers-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackProgrammers-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-programmers-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-programmers-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Programming, Written | derived | machine-translated and LM verified |



#### CQADupstackProgrammersRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-programmers`](https://huggingface.co/datasets/mteb/cqadupstack-programmers) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Programming, Written | derived | found |



#### CQADupstackProgrammersRetrieval-Fa

CQADupstackProgrammersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-programmers-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackStats-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackStats-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-stats-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-stats-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackStatsRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-stats`](https://huggingface.co/datasets/mteb/cqadupstack-stats) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



#### CQADupstackStatsRetrieval-Fa

CQADupstackStatsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-stats-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackTex-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackTex-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-tex-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-tex-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackTexRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-tex`](https://huggingface.co/datasets/mteb/cqadupstack-tex) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### CQADupstackTexRetrieval-Fa

CQADupstackTexRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-tex-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackUnix-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackUnix-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-unix-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-unix-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Programming, Web, Written | derived | machine-translated and LM verified |



#### CQADupstackUnixRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-unix`](https://huggingface.co/datasets/mteb/cqadupstack-unix) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Web, Written | derived | found |



#### CQADupstackUnixRetrieval-Fa

CQADupstackUnixRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-unix-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackWebmasters-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackWebmasters-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-webmasters-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-webmasters-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Web, Written | derived | machine-translated and LM verified |



#### CQADupstackWebmastersRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-webmasters`](https://huggingface.co/datasets/mteb/cqadupstack-webmasters) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | derived | found |



#### CQADupstackWebmastersRetrieval-Fa

CQADupstackWebmastersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-webmasters-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackWordpress-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackWordpress-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-wordpress-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-wordpress-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Programming, Web, Written | derived | machine-translated and LM verified |



#### CQADupstackWordpressRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-wordpress`](https://huggingface.co/datasets/mteb/cqadupstack-wordpress) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Web, Written | derived | found |



#### CQADupstackWordpressRetrieval-Fa

CQADupstackWordpressRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-wordpress-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CUREv1

Collection of query-passage pairs curated by medical professionals, across 10 disciplines and 3 cross-lingual settings.

**Dataset:** [`clinia/CUREv1`](https://huggingface.co/datasets/clinia/CUREv1) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/clinia/CUREv1)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, fra, spa | Academic, Medical, Written | expert-annotated | created |



#### ChatDoctorRetrieval

A medical retrieval task based on ChatDoctor_HealthCareMagic dataset containing 112,000 real-world medical question-and-answer pairs. Each query is a medical question from patients (e.g., 'What are the symptoms of diabetes?'), and the corpus contains medical responses and healthcare information. The task is to retrieve the correct medical information that answers the patient's question. The dataset includes grammatical inconsistencies which help separate strong healthcare retrieval models from weak ones. Queries are patient medical questions while the corpus contains relevant medical responses, diagnoses, and treatment information from healthcare professionals.

**Dataset:** [`embedding-benchmark/ChatDoctor_HealthCareMagic`](https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Medical | expert-annotated | found |



#### ChemHotpotQARetrieval

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemHotpotQARetrieval`](https://huggingface.co/datasets/BASF-AI/ChemHotpotQARetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Chemistry | derived | found |



#### ChemNQRetrieval

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemNQRetrieval`](https://huggingface.co/datasets/BASF-AI/ChemNQRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Chemistry | derived | found |



#### ClimateFEVER

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims (queries) regarding climate-change. The underlying corpus is the same as FVER.

**Dataset:** [`mteb/climate-fever`](https://huggingface.co/datasets/mteb/climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### ClimateFEVER-Fa

ClimateFEVER-Fa

**Dataset:** [`MCINext/climate-fever-fa`](https://huggingface.co/datasets/MCINext/climate-fever-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/climate-fever-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### ClimateFEVER-NL

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. ClimateFEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-climate-fever`](https://huggingface.co/datasets/clips/beir-nl-climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-climate-fever)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### ClimateFEVER-VN

A translated dataset from CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/climate-fever-vn`](https://huggingface.co/datasets/GreenNode/climate-fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### ClimateFEVER.v2

CLIMATE-FEVER is a dataset following the FEVER methodology, containing 1,535 real-world climate change claims. This updated version addresses corpus mismatches and qrel inconsistencies in MTEB, restoring labels while refining corpus-query alignment for better accuracy. 

**Dataset:** [`mteb/climate-fever-v2`](https://huggingface.co/datasets/mteb/climate-fever-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Written | human-annotated | found |



#### ClimateFEVERHardNegatives

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/ClimateFEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/ClimateFEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### CmedqaRetrieval

Online medical consultation text. Used the CMedQAv2 as its underlying dataset.

**Dataset:** [`mteb/CmedqaRetrieval`](https://huggingface.co/datasets/mteb/CmedqaRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Medical, Written | not specified | not specified |



#### CodeEditSearchRetrieval

The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.

**Dataset:** [`cassanof/CodeEditSearch`](https://huggingface.co/datasets/cassanof/CodeEditSearch) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | c, c++, go, java, javascript, ... (13) | Programming, Written | derived | found |



#### CodeFeedbackMT

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-mt`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-mt) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.14658)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### CodeFeedbackST

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-st`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-st) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### CodeSearchNetCCRetrieval

The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet-ccr`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet-ccr) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



#### CodeSearchNetRetrieval

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`code-search-net/code_search_net`](https://huggingface.co/datasets/code-search-net/code_search_net) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



#### CodeTransOceanContest

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet

**Dataset:** [`CoIR-Retrieval/codetrans-contest`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-contest) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | c++, python | Programming, Written | derived | found |



#### CodeTransOceanDL

The dataset is a collection of equivalent Python Deep Learning code snippets written in different machine learning framework. The task is to retrieve the equivalent code snippet in another framework, given a query code snippet from one framework.

**Dataset:** [`CoIR-Retrieval/codetrans-dl`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-dl) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming, Written | derived | found |



#### CosQA

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/cosqa`](https://huggingface.co/datasets/CoIR-Retrieval/cosqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.13239)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming, Written | derived | found |



#### CovidRetrieval

COVID-19 news articles

**Dataset:** [`mteb/CovidRetrieval`](https://huggingface.co/datasets/mteb/CovidRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Entertainment, Medical | human-annotated | not specified |



#### CrossLingualSemanticDiscriminationWMT19

Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT19 DE-FR test set

**Dataset:** [`Andrianos/clsd_wmt19_21`](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_1 | deu, fra | News, Written | derived | LM-generated and verified |



#### CrossLingualSemanticDiscriminationWMT21

Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT21 DE-FR test set

**Dataset:** [`Andrianos/clsd_wmt19_21`](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_1 | deu, fra | News, Written | derived | LM-generated and verified |



#### DBPedia

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/dbpedia`](https://huggingface.co/datasets/mteb/dbpedia) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### DBPedia-Fa

DBPedia-Fa

**Dataset:** [`MCINext/dbpedia-fa`](https://huggingface.co/datasets/MCINext/dbpedia-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/dbpedia-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



#### DBPedia-NL

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. DBPedia-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-dbpedia-entity`](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### DBPedia-PL

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/DBPedia-PL`](https://huggingface.co/datasets/mteb/DBPedia-PL) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Encyclopaedic, Written | derived | machine-translated |



#### DBPedia-PLHardNegatives

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_PL_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Encyclopaedic, Written | derived | machine-translated |



#### DBPedia-VN

A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/dbpedia-vn`](https://huggingface.co/datasets/GreenNode/dbpedia-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### DBPediaHardNegatives

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### DS1000Retrieval

A code retrieval task based on 1,000 data science programming problems from DS-1000. Each query is a natural language description of a data science task (e.g., 'Create a scatter plot of column A vs column B with matplotlib'), and the corpus contains Python code implementations using libraries like pandas, numpy, matplotlib, scikit-learn, and scipy. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations focused on data science workflows.

**Dataset:** [`embedding-benchmark/DS1000`](https://huggingface.co/datasets/embedding-benchmark/DS1000) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/DS1000)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, python | Programming | expert-annotated | found |



#### DanFEVER

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.

**Dataset:** [`strombergnlp/danfever`](https://huggingface.co/datasets/strombergnlp/danfever) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found |



#### DanFeverRetrieval

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset. DanFeverRetrieval fixed an issue in DanFever where some corpus entries were incorrectly removed.

**Dataset:** [`strombergnlp/danfever`](https://huggingface.co/datasets/strombergnlp/danfever) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found |



#### DuRetrieval

A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine

**Dataset:** [`mteb/DuRetrieval`](https://huggingface.co/datasets/mteb/DuRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### EcomRetrieval

EcomRetrieval

**Dataset:** [`mteb/EcomRetrieval`](https://huggingface.co/datasets/mteb/EcomRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### EstQA

EstQA is an Estonian question answering dataset based on Wikipedia.

**Dataset:** [`kardosdrur/estonian-qa`](https://huggingface.co/datasets/kardosdrur/estonian-qa) • **License:** not specified • [Learn more →](https://www.semanticscholar.org/paper/Extractive-Question-Answering-for-Estonian-Language-182912IAPM-Alum%C3%A4e/ea4f60ab36cadca059c880678bc4c51e293a85d6?utm_source=direct_link)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | est | Encyclopaedic, Written | human-annotated | found |



#### FEVER

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`mteb/fever`](https://huggingface.co/datasets/mteb/fever) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### FEVER-NL

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. FEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-fever`](https://huggingface.co/datasets/clips/beir-nl-fever) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fever)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### FEVER-VN

A translated dataset from FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences
            extracted from Wikipedia and subsequently verified without knowledge of the sentence they were
            derived from.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fever-vn`](https://huggingface.co/datasets/GreenNode/fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### FEVERHardNegatives

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/FEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | not specified |



#### FQuADRetrieval

This dataset has been built from the French SQuad dataset.

**Dataset:** [`manu/fquad2_test`](https://huggingface.co/datasets/manu/fquad2_test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/manu/fquad2_test)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Encyclopaedic, Written | human-annotated | created |



#### FaithDial

FaithDial is a faithful knowledge-grounded dialogue benchmark.It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). It consists of conversation histories along with manually labelled relevant passage. For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, as the gold passage associated with these instances is non-ambiguous.

**Dataset:** [`McGill-NLP/FaithDial`](https://huggingface.co/datasets/McGill-NLP/FaithDial) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/FaithDial)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### FeedbackQARetrieval

Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment

**Dataset:** [`lt2c/fqa`](https://huggingface.co/datasets/lt2c/fqa) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.03025)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | precision_at_1 | eng | Government, Medical, Web, Written | human-annotated | created |



#### FiQA-PL

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/FiQA-PL`](https://huggingface.co/datasets/mteb/FiQA-PL) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Financial, Written | human-annotated | found |



#### FiQA2018

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/fiqa`](https://huggingface.co/datasets/mteb/fiqa) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Financial, Written | human-annotated | found |



#### FiQA2018-Fa

FiQA2018-Fa

**Dataset:** [`MCINext/fiqa-fa`](https://huggingface.co/datasets/MCINext/fiqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/fiqa-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### FiQA2018-NL

Financial Opinion Mining and Question Answering. FiQA2018-NL is a Dutch translation

**Dataset:** [`clips/beir-nl-fiqa`](https://huggingface.co/datasets/clips/beir-nl-fiqa) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fiqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### FiQA2018-VN

A translated dataset from Financial Opinion Mining and Question Answering
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fiqa-vn`](https://huggingface.co/datasets/GreenNode/fiqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Financial, Written | derived | machine-translated and LM verified |



#### FinQARetrieval

A financial retrieval task based on FinQA dataset containing numerical reasoning questions over financial documents. Each query is a financial question requiring numerical computation (e.g., 'What is the percentage change in operating expenses from 2019 to 2020?'), and the corpus contains financial document text with tables and numerical data. The task is to retrieve the correct financial information that enables answering the numerical question. Queries are numerical reasoning questions while the corpus contains financial text passages with embedded tables, figures, and quantitative financial data from earnings reports.

**Dataset:** [`embedding-benchmark/FinQA`](https://huggingface.co/datasets/embedding-benchmark/FinQA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Financial | expert-annotated | found |



#### FinanceBenchRetrieval

A financial retrieval task based on FinanceBench dataset containing financial questions and answers. Each query is a financial question (e.g., 'What was the total revenue in Q3 2023?'), and the corpus contains financial document excerpts and annual reports. The task is to retrieve the correct financial information that answers the question. Queries are financial questions while the corpus contains relevant excerpts from financial documents, earnings reports, and SEC filings with detailed financial data and metrics.

**Dataset:** [`embedding-benchmark/FinanceBench`](https://huggingface.co/datasets/embedding-benchmark/FinanceBench) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinanceBench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Financial | expert-annotated | found |



#### FreshStackRetrieval

A code retrieval task based on FreshStack dataset containing programming problems across multiple languages. Each query is a natural language description of a programming task (e.g., 'Write a function to reverse a string using recursion'), and the corpus contains code implementations in Python, JavaScript, and Go. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains function implementations with proper syntax and logic across different programming languages.

**Dataset:** [`embedding-benchmark/FreshStack_mteb`](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, go, javascript, python | Programming | expert-annotated | found |



#### GeorgianFAQRetrieval

Frequently asked questions (FAQs) and answers mined from Georgian websites via Common Crawl.

**Dataset:** [`jupyterjazz/georgian-faq`](https://huggingface.co/datasets/jupyterjazz/georgian-faq) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jupyterjazz/georgian-faq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kat | Web, Written | derived | created |



#### GerDaLIR

GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.

**Dataset:** [`jinaai/ger_da_lir`](https://huggingface.co/datasets/jinaai/ger_da_lir) • **License:** not specified • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal | not specified | not specified |



#### GerDaLIRSmall

The dataset consists of documents, passages and relevance labels in German. In contrast to the original dataset, only documents that have corresponding queries in the query set are chosen to create a smaller corpus for evaluation purposes.

**Dataset:** [`mteb/GerDaLIRSmall`](https://huggingface.co/datasets/mteb/GerDaLIRSmall) • **License:** mit • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal, Written | derived | found |



#### GermanDPR

GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer

**Dataset:** [`deepset/germandpr`](https://huggingface.co/datasets/deepset/germandpr) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germandpr)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Non-fiction, Web, Written | human-annotated | found |



#### GermanGovServiceRetrieval

LHM-Dienstleistungen-QA is a German question answering dataset for government services of the Munich city administration. It associates questions with a textual context containing the answer

**Dataset:** [`it-at-m/LHM-Dienstleistungen-QA`](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_5 | deu | Government, Written | derived | found |



#### GermanQuAD-Retrieval

Context Retrieval for German Question Answering

**Dataset:** [`mteb/germanquad-retrieval`](https://huggingface.co/datasets/mteb/germanquad-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germanquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | mrr_at_5 | deu | Non-fiction, Web, Written | human-annotated | found |



#### GovReport

A dataset for evaluating the ability of information retrieval models to retrieve lengthy US government reports from their summaries.

**Dataset:** [`isaacus/mteb-GovReport`](https://huggingface.co/datasets/isaacus/mteb-GovReport) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/launch/gov_report)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



#### GreekCivicsQA

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`ilsp/greek_civics_qa`](https://huggingface.co/datasets/ilsp/greek_civics_qa) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ell | Academic, Written | derived | found |



#### GreenNodeTableMarkdownRetrieval

GreenNodeTable documents

**Dataset:** [`GreenNode/GreenNode-Table-Markdown-Retrieval-VN`](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN) • **License:** mit • [Learn more →](https://huggingface.co/GreenNode)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Financial, Non-fiction | human-annotated | found |



#### HC3FinanceRetrieval

A financial retrieval task based on HC3 Finance dataset containing human vs AI-generated financial text detection. Each query is a financial question or prompt (e.g., 'Explain the impact of interest rate changes on bond prices'), and the corpus contains both human-written and AI-generated financial responses. The task is to retrieve the most relevant and accurate financial content that addresses the query. Queries are financial questions while the corpus contains detailed financial explanations, analysis, and educational content covering various financial concepts and market dynamics.

**Dataset:** [`embedding-benchmark/HC3Finance`](https://huggingface.co/datasets/embedding-benchmark/HC3Finance) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HC3Finance)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Financial | expert-annotated | found |



#### HagridRetrieval

HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)is a dataset for generative information-seeking scenarios. It consists of queriesalong with a set of manually labelled relevant passages

**Dataset:** [`miracl/hagrid`](https://huggingface.co/datasets/miracl/hagrid) • **License:** apache-2.0 • [Learn more →](https://github.com/project-miracl/hagrid)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | expert-annotated | found |



#### HellaSwag

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on HellaSwag.

**Dataset:** [`RAR-b/hellaswag`](https://huggingface.co/datasets/RAR-b/hellaswag) • **License:** mit • [Learn more →](https://rowanzellers.com/hellaswag/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### HotpotQA

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/hotpotqa`](https://huggingface.co/datasets/mteb/hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



#### HotpotQA-Fa

HotpotQA-Fa

**Dataset:** [`MCINext/hotpotqa-fa`](https://huggingface.co/datasets/MCINext/hotpotqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/hotpotqa-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



#### HotpotQA-NL

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strongsupervision for supporting facts to enable more explainable question answering systems. HotpotQA-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-hotpotqa`](https://huggingface.co/datasets/clips/beir-nl-hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Web, Written | derived | machine-translated and verified |



#### HotpotQA-PL

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/HotpotQA-PL`](https://huggingface.co/datasets/mteb/HotpotQA-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### HotpotQA-PLHardNegatives

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_PL_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### HotpotQA-VN

A translated dataset from HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong
            supervision for supporting facts to enable more explainable question answering systems.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/hotpotqa-vn`](https://huggingface.co/datasets/GreenNode/hotpotqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Web, Written | derived | machine-translated and LM verified |



#### HotpotQAHardNegatives

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.  The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



#### HumanEvalRetrieval

A code retrieval task based on 164 Python programming problems from HumanEval. Each query is a natural language description of a programming task (e.g., 'Check if in given list of numbers, are any two numbers closer to each other than given threshold'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper indentation and logic.

**Dataset:** [`embedding-benchmark/HumanEval`](https://huggingface.co/datasets/embedding-benchmark/HumanEval) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HumanEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, python | Programming | derived | found |



#### HunSum2AbstractiveRetrieval

HunSum-2-abstractive is a Hungarian dataset containing news articles along with lead, titles and metadata.

**Dataset:** [`SZTAKI-HLT/HunSum-2-abstractive`](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-2-abstractive) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2404.03555)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | hun | News, Written | derived | found |



#### IndicQARetrieval

IndicQA is a manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages. It is repurposed retrieving relevant context for each question.

**Dataset:** [`mteb/IndicQARetrieval`](https://huggingface.co/datasets/mteb/IndicQARetrieval) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | asm, ben, guj, hin, kan, ... (11) | Web, Written | human-annotated | machine-translated and verified |



#### JaCWIRRetrieval

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of
5000 question texts and approximately 500k web page titles and web page introductions or summaries
(meta descriptions, etc.). The question texts are created based on one of the 500k web pages,
and that data is used as a positive example for the question text.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Web, Written | derived | found |



#### JaGovFaqsRetrieval

JaGovFaqs is a dataset consisting of FAQs manully extracted from the website of Japanese bureaus. The dataset consists of 22k FAQs, where the queries (questions) and corpus (answers) have been shuffled, and the goal is to match the answer with the question.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Web, Written | derived | found |



#### JaQuADRetrieval

Human-annotated question-answer pairs for Japanese wikipedia pages.

**Dataset:** [`SkelterLabsInc/JaQuAD`](https://huggingface.co/datasets/SkelterLabsInc/JaQuAD) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/2202.01764)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found |



#### JaqketRetrieval

JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset that is created based on quiz questions.

**Dataset:** [`mteb/jaqket`](https://huggingface.co/datasets/mteb/jaqket) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/kumapo/JAQKET-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found |



#### Ko-StrategyQA

Ko-StrategyQA

**Dataset:** [`taeminlee/Ko-StrategyQA`](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kor | not specified | not specified | not specified |



#### LEMBNarrativeQARetrieval

narrativeqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction, Non-fiction, Written | derived | found |



#### LEMBNeedleRetrieval

needle subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | eng | Academic, Blog, Written | derived | found |



#### LEMBPasskeyRetrieval

passkey subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | eng | Fiction, Written | derived | found |



#### LEMBQMSumRetrieval

qmsum subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Spoken, Written | derived | found |



#### LEMBSummScreenFDRetrieval

summ_screen_fd subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Spoken, Written | derived | found |



#### LEMBWikimQARetrieval

2wikimqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### LIMITRetrieval

A simple retrieval task designed to test all combinations of top-2 documents. This version includes all 50k docs.

**Dataset:** [`orionweller/LIMIT`](https://huggingface.co/datasets/orionweller/LIMIT) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_2 | eng | Fiction | human-annotated | created |



#### LIMITSmallRetrieval

A simple retrieval task designed to test all combinations of top-2 documents. This version only includes the 46 documents that are relevant to the 1000 queries.

**Dataset:** [`orionweller/LIMIT-small`](https://huggingface.co/datasets/orionweller/LIMIT-small) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_2 | eng | Fiction | human-annotated | created |



#### LeCaRDv2

The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.

**Dataset:** [`mteb/LeCaRDv2`](https://huggingface.co/datasets/mteb/LeCaRDv2) • **License:** mit • [Learn more →](https://github.com/THUIR/LeCaRDv2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | zho | Legal, Written | derived | found |



#### LegalBenchConsumerContractsQA

The dataset includes questions and answers related to contracts.

**Dataset:** [`mteb/legalbench_consumer_contracts_qa`](https://huggingface.co/datasets/mteb/legalbench_consumer_contracts_qa) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### LegalBenchCorporateLobbying

The dataset includes bill titles and bill summaries related to corporate lobbying.

**Dataset:** [`mteb/legalbench_corporate_lobbying`](https://huggingface.co/datasets/mteb/legalbench_corporate_lobbying) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### LegalQuAD

The dataset consists of questions and legal documents in German.

**Dataset:** [`mteb/LegalQuAD`](https://huggingface.co/datasets/mteb/LegalQuAD) • **License:** cc-by-4.0 • [Learn more →](https://github.com/Christoph911/AIKE2021_Appendix)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal, Written | derived | found |



#### LegalSummarization

The dataset consistes of 439 pairs of contracts and their summarizations from https://tldrlegal.com and https://tosdr.org/.

**Dataset:** [`mteb/legal_summarization`](https://huggingface.co/datasets/mteb/legal_summarization) • **License:** apache-2.0 • [Learn more →](https://github.com/lauramanor/legal_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### LitSearchRetrieval


        The dataset contains the query set and retrieval corpus for the paper LitSearch: A Retrieval Benchmark for
        Scientific Literature Search. It introduces LitSearch, a retrieval benchmark comprising 597 realistic literature
        search queries about recent ML and NLP papers. LitSearch is constructed using a combination of (1) questions
        generated by GPT-4 based on paragraphs containing inline citations from research papers and (2) questions about
        recently published papers, manually written by their authors. All LitSearch questions were manually examined or
        edited by experts to ensure high quality.
        

**Dataset:** [`princeton-nlp/LitSearch`](https://huggingface.co/datasets/princeton-nlp/LitSearch) • **License:** mit • [Learn more →](https://github.com/princeton-nlp/LitSearch)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | LM-generated | found |



#### LoTTE

LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing a challenging out-of-domain generalization benchmark.

**Dataset:** [`mteb/LoTTE`](https://huggingface.co/datasets/mteb/LoTTE) • **License:** mit • [Learn more →](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | precision_at_5 | eng | Academic, Social, Web | derived | found |



#### MBPPRetrieval

A code retrieval task based on 378 Python programming problems from MBPP (Mostly Basic Python Programming). Each query is a natural language description of a programming task (e.g., 'Write a function to find the shared elements from the given two lists'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper syntax and logic.

**Dataset:** [`embedding-benchmark/MBPP`](https://huggingface.co/datasets/embedding-benchmark/MBPP) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/MBPP)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, python | Programming | expert-annotated | found |



#### MIRACLRetrieval

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`miracl/mmteb-miracl`](https://huggingface.co/datasets/miracl/mmteb-miracl) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



#### MIRACLRetrievalHardNegatives

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/miracl-hard-negatives`](https://huggingface.co/datasets/mteb/miracl-hard-negatives) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



#### MKQARetrieval

Multilingual Knowledge Questions & Answers (MKQA)contains 10,000 queries sampled from the Google Natural Questions dataset.
        For each query we collect new passage-independent answers. These queries and answers are then human translated into 25 Non-English languages.

**Dataset:** [`apple/mkqa`](https://huggingface.co/datasets/apple/mkqa) • **License:** cc-by-3.0 • [Learn more →](https://github.com/apple/ml-mkqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, dan, deu, eng, fin, ... (24) | Written | human-annotated | found |



#### MLQARetrieval

MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
        MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
        German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
        4 different languages on average.

**Dataset:** [`facebook/mlqa`](https://huggingface.co/datasets/facebook/mlqa) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/mlqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, deu, eng, hin, spa, ... (7) | Encyclopaedic, Written | human-annotated | found |



#### MLQuestions

MLQuestions is a domain adaptation dataset for the machine learning domainIt consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)

**Dataset:** [`McGill-NLP/mlquestions`](https://huggingface.co/datasets/McGill-NLP/mlquestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/McGill-NLP/MLQuestions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Encyclopaedic, Written | human-annotated | found |



#### MMarcoRetrieval

MMarcoRetrieval

**Dataset:** [`mteb/MMarcoRetrieval`](https://huggingface.co/datasets/mteb/MMarcoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2309.07597)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### MSMARCO

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/msmarco`](https://huggingface.co/datasets/mteb/msmarco) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



#### MSMARCO-Fa

MSMARCO-Fa

**Dataset:** [`MCINext/msmarco-fa`](https://huggingface.co/datasets/MCINext/msmarco-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/msmarco-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### MSMARCO-PL

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/MSMARCO-PL`](https://huggingface.co/datasets/mteb/MSMARCO-PL) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### MSMARCO-PLHardNegatives

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/MSMARCO_PL_test_top_250_only_w_correct-v2) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### MSMARCO-VN

A translated dataset from MS MARCO is a collection of datasets focused on deep learning in search
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/msmarco-vn`](https://huggingface.co/datasets/GreenNode/msmarco-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | machine-translated and LM verified |



#### MSMARCOHardNegatives

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/MSMARCO_test_top_250_only_w_correct-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



#### MSMARCOv2

MS MARCO is a collection of datasets focused on deep learning in search. This version is derived from BEIR

**Dataset:** [`mteb/msmarco-v2`](https://huggingface.co/datasets/mteb/msmarco-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



#### MedicalQARetrieval

The dataset consists 2048 medical question and answer pairs.

**Dataset:** [`mteb/medical_qa`](https://huggingface.co/datasets/mteb/medical_qa) • **License:** cc0-1.0 • [Learn more →](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | derived | found |



#### MedicalRetrieval

MedicalRetrieval

**Dataset:** [`mteb/MedicalRetrieval`](https://huggingface.co/datasets/mteb/MedicalRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### MintakaRetrieval

We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers. 

**Dataset:** [`jinaai/mintakaqa`](https://huggingface.co/datasets/jinaai/mintakaqa) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/mintakaqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, deu, fra, hin, ita, ... (8) | Encyclopaedic, Written | derived | human-translated |



#### MrTidyRetrieval

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.

**Dataset:** [`mteb/mrtidy`](https://huggingface.co/datasets/mteb/mrtidy) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/castorini/mr-tydi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, eng, fin, ind, ... (11) | Encyclopaedic, Written | human-annotated | found |



#### MultiLongDocRetrieval

Multi Long Doc Retrieval (MLDR) 'is curated by the multilingual articles from Wikipedia, Wudao and mC4 (see Table 7), and NarrativeQA (Kocˇisky ́ et al., 2018; Gu ̈nther et al., 2023), which is only for English.' (Chen et al., 2024).
        It is constructed by sampling lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.

**Dataset:** [`Shitao/MLDR`](https://huggingface.co/datasets/Shitao/MLDR) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.03216)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, cmn, deu, eng, fra, ... (13) | Encyclopaedic, Fiction, Non-fiction, Web, Written | LM-generated | found |



#### NFCorpus

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nfcorpus`](https://huggingface.co/datasets/mteb/nfcorpus) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



#### NFCorpus-Fa

NFCorpus-Fa

**Dataset:** [`MCINext/nfcorpus-fa`](https://huggingface.co/datasets/MCINext/nfcorpus-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nfcorpus-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



#### NFCorpus-NL

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-nfcorpus`](https://huggingface.co/datasets/clips/beir-nl-nfcorpus) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nfcorpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



#### NFCorpus-PL

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/NFCorpus-PL`](https://huggingface.co/datasets/mteb/NFCorpus-PL) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | not specified |



#### NFCorpus-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nfcorpus-vn`](https://huggingface.co/datasets/GreenNode/nfcorpus-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



#### NLPJournalAbsArticleRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalAbsArticleRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalAbsIntroRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V1 dataset (last update 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalAbsIntroRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V2 dataset (last update 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleAbsRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleAbsRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleIntroRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleIntroRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NQ

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nq`](https://huggingface.co/datasets/mteb/nq) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### NQ-Fa

NQ-Fa

**Dataset:** [`MCINext/nq-fa`](https://huggingface.co/datasets/MCINext/nq-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nq-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



#### NQ-NL

NQ-NL is a translation of NQ

**Dataset:** [`clips/beir-nl-nq`](https://huggingface.co/datasets/clips/beir-nl-nq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### NQ-PL

Natural Questions: A Benchmark for Question Answering Research

**Dataset:** [`mteb/NQ-PL`](https://huggingface.co/datasets/mteb/NQ-PL) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### NQ-PLHardNegatives

Natural Questions: A Benchmark for Question Answering Research. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/NQ_PL_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### NQ-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nq-vn`](https://huggingface.co/datasets/GreenNode/nq-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### NQHardNegatives

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/NQ_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



#### NanoArguAnaRetrieval

NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.

**Dataset:** [`zeta-alpha-ai/NanoArguAna`](https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna) • **License:** cc-by-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | expert-annotated | found |



#### NanoClimateFeverRetrieval

NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.

**Dataset:** [`zeta-alpha-ai/NanoClimateFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2012.00614)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, News, Non-fiction | expert-annotated | found |



#### NanoDBPediaRetrieval

NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.

**Dataset:** [`zeta-alpha-ai/NanoDBPedia`](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic | expert-annotated | found |



#### NanoFEVERRetrieval

NanoFEVER is a smaller version of FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`zeta-alpha-ai/NanoFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER) • **License:** cc-by-4.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Encyclopaedic | expert-annotated | found |



#### NanoFiQA2018Retrieval

NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset.

**Dataset:** [`zeta-alpha-ai/NanoFiQA2018`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018) • **License:** cc-by-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Social | human-annotated | found |



#### NanoHotpotQARetrieval

NanoHotpotQARetrieval is a smaller subset of the HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`zeta-alpha-ai/NanoHotpotQA`](https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA) • **License:** cc-by-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



#### NanoMSMARCORetrieval

NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.

**Dataset:** [`zeta-alpha-ai/NanoMSMARCO`](https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO) • **License:** cc-by-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web | human-annotated | found |



#### NanoNFCorpusRetrieval

NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.

**Dataset:** [`zeta-alpha-ai/NanoNFCorpus`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus) • **License:** cc-by-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | expert-annotated | found |



#### NanoNQRetrieval

NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.

**Dataset:** [`zeta-alpha-ai/NanoNQ`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ) • **License:** cc-by-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Web | human-annotated | found |



#### NanoQuoraRetrieval

NanoQuoraRetrieval is a smaller subset of the QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`zeta-alpha-ai/NanoQuoraRetrieval`](https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Social | human-annotated | found |



#### NanoSCIDOCSRetrieval

NanoFiQA2018 is a smaller subset of SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`zeta-alpha-ai/NanoSCIDOCS`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | expert-annotated | found |



#### NanoSciFactRetrieval

NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`zeta-alpha-ai/NanoSciFact`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact) • **License:** cc-by-4.0 • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | expert-annotated | found |



#### NanoTouche2020Retrieval

NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.

**Dataset:** [`zeta-alpha-ai/NanoTouche2020`](https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020) • **License:** cc-by-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



#### NarrativeQARetrieval

NarrativeQA is a dataset for the task of question answering on long narratives. It consists of realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. 

**Dataset:** [`deepmind/narrativeqa`](https://huggingface.co/datasets/deepmind/narrativeqa) • **License:** not specified • [Learn more →](https://metatext.io/datasets/narrativeqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



#### NeuCLIR2022Retrieval

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/neuclir-2022`](https://huggingface.co/datasets/mteb/neuclir-2022) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NeuCLIR2022RetrievalHardNegatives

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/neuclir-2022-hard-negatives`](https://huggingface.co/datasets/mteb/neuclir-2022-hard-negatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NeuCLIR2023Retrieval

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/neuclir-2023`](https://huggingface.co/datasets/mteb/neuclir-2023) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NeuCLIR2023RetrievalHardNegatives

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/neuclir-2023-hard-negatives`](https://huggingface.co/datasets/mteb/neuclir-2023-hard-negatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NorQuadRetrieval

Human-created question for Norwegian wikipedia passages.

**Dataset:** [`mteb/norquad_retrieval`](https://huggingface.co/datasets/mteb/norquad_retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.17/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### PIQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.

**Dataset:** [`RAR-b/piqa`](https://huggingface.co/datasets/RAR-b/piqa) • **License:** afl-3.0 • [Learn more →](https://arxiv.org/abs/1911.11641)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### PUGGRetrieval

Information Retrieval PUGG dataset for the Polish language.

**Dataset:** [`clarin-pl/PUGG_IR`](https://huggingface.co/datasets/clarin-pl/PUGG_IR) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2024.findings-acl.652/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web | human-annotated | multiple |



#### PersianWebDocumentRetrieval

Persian dataset designed specifically for the task of text information retrieval through the web.

**Dataset:** [`MCINext/persian-web-document-retrieval`](https://huggingface.co/datasets/MCINext/persian-web-document-retrieval) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/10553090)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### PublicHealthQA

A multilingual dataset for public health question answering, based on FAQ sourced from CDC and WHO.

**Dataset:** [`xhluca/publichealth-qa`](https://huggingface.co/datasets/xhluca/publichealth-qa) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/xhluca/publichealth-qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, eng, fra, kor, rus, ... (8) | Government, Medical, Web, Written | derived | found |



#### Quail

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on Quail.

**Dataset:** [`RAR-b/quail`](https://huggingface.co/datasets/RAR-b/quail) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://text-machine.cs.uml.edu/lab2/projects/quail/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### Quora-NL

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. QuoraRetrieval-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-quora`](https://huggingface.co/datasets/clips/beir-nl-quora) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-quora)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Written | derived | machine-translated and verified |



#### Quora-PL

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/Quora-PL`](https://huggingface.co/datasets/mteb/Quora-PL) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### Quora-PLHardNegatives

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/Quora_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/Quora_PL_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### Quora-VN

A translated dataset from QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a
            question, find other (duplicate) questions.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/quora-vn`](https://huggingface.co/datasets/GreenNode/quora-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Blog, Web, Written | derived | machine-translated and LM verified |



#### QuoraRetrieval

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/quora`](https://huggingface.co/datasets/mteb/quora) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Blog, Web, Written | human-annotated | found |



#### QuoraRetrieval-Fa

QuoraRetrieval-Fa

**Dataset:** [`MCINext/quora-fa`](https://huggingface.co/datasets/MCINext/quora-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/quora-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### QuoraRetrievalHardNegatives

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/QuoraRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/QuoraRetrieval_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



#### R2MEDBioinformaticsRetrieval

Bioinformatics retrieval dataset.

**Dataset:** [`R2MED/Bioinformatics`](https://huggingface.co/datasets/R2MED/Bioinformatics) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Bioinformatics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDBiologyRetrieval

Biology retrieval dataset.

**Dataset:** [`R2MED/Biology`](https://huggingface.co/datasets/R2MED/Biology) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Biology)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDIIYiClinicalRetrieval

IIYi-Clinical retrieval dataset.

**Dataset:** [`R2MED/IIYi-Clinical`](https://huggingface.co/datasets/R2MED/IIYi-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/IIYi-Clinical)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDMedQADiagRetrieval

MedQA-Diag retrieval dataset.

**Dataset:** [`R2MED/MedQA-Diag`](https://huggingface.co/datasets/R2MED/MedQA-Diag) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedQA-Diag)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDMedXpertQAExamRetrieval

MedXpertQA-Exam retrieval dataset.

**Dataset:** [`R2MED/MedXpertQA-Exam`](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDMedicalSciencesRetrieval

Medical-Sciences retrieval dataset.

**Dataset:** [`R2MED/Medical-Sciences`](https://huggingface.co/datasets/R2MED/Medical-Sciences) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Medical-Sciences)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDPMCClinicalRetrieval

PMC-Clinical retrieval dataset.

**Dataset:** [`R2MED/PMC-Clinical`](https://huggingface.co/datasets/R2MED/PMC-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Clinical)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDPMCTreatmentRetrieval

PMC-Treatment retrieval dataset.

**Dataset:** [`R2MED/PMC-Treatment`](https://huggingface.co/datasets/R2MED/PMC-Treatment) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Treatment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### RARbCode

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.

**Dataset:** [`RAR-b/humanevalpack-mbpp-pooled`](https://huggingface.co/datasets/RAR-b/humanevalpack-mbpp-pooled) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.06347)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### RARbMath

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.

**Dataset:** [`RAR-b/math-pooled`](https://huggingface.co/datasets/RAR-b/math-pooled) • **License:** mit • [Learn more →](https://arxiv.org/abs/2404.06347)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### RiaNewsRetrieval

News article retrieval by headline. Based on Rossiya Segodnya dataset.

**Dataset:** [`ai-forever/ria-news-retrieval`](https://huggingface.co/datasets/ai-forever/ria-news-retrieval) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | News, Written | derived | found |



#### RiaNewsRetrievalHardNegatives

News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | News, Written | derived | found |



#### RuBQRetrieval

Paragraph retrieval based on RuBQ 2.0. Retrieve paragraphs from Wikipedia that answer the question.

**Dataset:** [`ai-forever/rubq-retrieval`](https://huggingface.co/datasets/ai-forever/rubq-retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | Encyclopaedic, Written | human-annotated | created |



#### RuSciBenchCiteRetrieval

This task is focused on Direct Citation Prediction for scientific papers from eLibrary,
        Russia's largest electronic library of scientific publications. Given a query paper (title and abstract),
        the goal is to retrieve papers that are directly cited by it from a larger corpus of papers.
        The dataset for this task consists of 3,000 query papers, 15,000 relevant (cited) papers,
        and 75,000 irrelevant papers. The task is available for both Russian and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchCociteRetrieval

This task focuses on Co-citation Prediction for scientific papers from eLibrary,
        Russia's largest electronic library of scientific publications. Given a query paper (title and abstract),
        the goal is to retrieve other papers that are co-cited with it. Two papers are considered co-cited
        if they are both cited by at least 5 of the same other papers. Similar to the Direct Citation task,
        this task employs a retrieval setup: for a given query paper, all other papers in the corpus that
        are not co-cited with it are considered negative examples. The task is available for both Russian
        and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, rus | Academic, Non-fiction, Written | derived | found |



#### SCIDOCS

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/scidocs`](https://huggingface.co/datasets/mteb/scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | not specified | found |



#### SCIDOCS-Fa

SCIDOCS-Fa

**Dataset:** [`MCINext/scidocs-fa`](https://huggingface.co/datasets/MCINext/scidocs-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scidocs-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



#### SCIDOCS-NL

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-scidocs`](https://huggingface.co/datasets/clips/beir-nl-scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Non-fiction, Written | derived | machine-translated and verified |



#### SCIDOCS-PL

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/SCIDOCS-PL`](https://huggingface.co/datasets/mteb/SCIDOCS-PL) • **License:** not specified • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | not specified |



#### SCIDOCS-VN

A translated dataset from SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation
            prediction, to document classification and recommendation.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scidocs-vn`](https://huggingface.co/datasets/GreenNode/scidocs-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### SIQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SIQA.

**Dataset:** [`RAR-b/siqa`](https://huggingface.co/datasets/RAR-b/siqa) • **License:** not specified • [Learn more →](https://leaderboard.allenai.org/socialiqa/submissions/get-started)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### SKQuadRetrieval

Retrieval SK Quad evaluates Slovak search performance using questions and answers derived from the SK-QuAD dataset. It measures relevance with scores assigned to answers based on their relevancy to corresponding questions, which is vital for improving Slovak language search systems.

**Dataset:** [`TUKE-KEMT/retrieval-skquad`](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | slk | Encyclopaedic | human-annotated | found |



#### SNLRetrieval

Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.

**Dataset:** [`adrlau/navjordj-SNL_summarization_copy`](https://huggingface.co/datasets/adrlau/navjordj-SNL_summarization_copy) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLRetrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### SadeemQuestionRetrieval

SadeemQuestion: A Benchmark Data Set for Community Question-Retrieval Research

**Dataset:** [`sadeem-ai/sadeem-ar-eval-retrieval-questions`](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara | Written, Written | derived | found |



#### SciFact

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact) • **License:** not specified • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



#### SciFact-Fa

SciFact-Fa

**Dataset:** [`MCINext/scifact-fa`](https://huggingface.co/datasets/MCINext/scifact-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scifact-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



#### SciFact-NL

SciFactNL verifies scientific claims in Dutch using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`clips/beir-nl-scifact`](https://huggingface.co/datasets/clips/beir-nl-scifact) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



#### SciFact-PL

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/SciFact-PL`](https://huggingface.co/datasets/mteb/SciFact-PL) • **License:** not specified • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Medical, Written | not specified | not specified |



#### SciFact-VN

A translated dataset from SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scifact-vn`](https://huggingface.co/datasets/GreenNode/scifact-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



#### SlovakSumRetrieval


            SlovakSum, a Slovak news summarization dataset consisting of over 200 thousand
            news articles with titles and short abstracts obtained from multiple Slovak newspapers.

            Originally intended as a summarization task, but since no human annotations were provided
            here reformulated to a retrieval task.
        

**Dataset:** [`NaiveNeuron/slovaksum`](https://huggingface.co/datasets/NaiveNeuron/slovaksum) • **License:** openrail • [Learn more →](https://huggingface.co/datasets/NaiveNeuron/slovaksum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | slk | News, Social, Web, Written | derived | found |



#### SpanishPassageRetrievalS2P

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`jinaai/spanish_passage_retrieval`](https://huggingface.co/datasets/jinaai/spanish_passage_retrieval) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | spa | not specified | not specified | not specified |



#### SpanishPassageRetrievalS2S

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`jinaai/spanish_passage_retrieval`](https://huggingface.co/datasets/jinaai/spanish_passage_retrieval) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | spa | not specified | not specified | not specified |



#### SpartQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SpartQA.

**Dataset:** [`RAR-b/spartqa`](https://huggingface.co/datasets/RAR-b/spartqa) • **License:** mit • [Learn more →](https://github.com/HLR/SpartQA_generation)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### StackOverflowQA

The dataset is a collection of natural language queries and their corresponding response which may include some text mixed with code snippets. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/stackoverflow-qa`](https://huggingface.co/datasets/CoIR-Retrieval/stackoverflow-qa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### StatcanDialogueDatasetRetrieval

A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.

**Dataset:** [`McGill-NLP/statcan-dialogue-dataset-retrieval`](https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval) • **License:** https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md • [Learn more →](https://mcgill-nlp.github.io/statcan-dialogue-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, fra | Government, Web, Written | derived | found |



#### SweFaqRetrieval

A Swedish QA dataset derived from FAQ

**Dataset:** [`AI-Sweden/SuperLim`](https://huggingface.co/datasets/AI-Sweden/SuperLim) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | swe | Government, Non-fiction, Written | derived | found |



#### SwednRetrieval

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure

**Dataset:** [`sbx/superlim-2`](https://huggingface.co/datasets/sbx/superlim-2) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | swe | News, Non-fiction, Written | derived | found |



#### SynPerChatbotRAGFAQRetrieval

Synthetic Persian Chatbot RAG FAQ Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-faq-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotRAGTopicsRetrieval

Synthetic Persian Chatbot RAG Topics Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-topics-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotTopicsRetrieval

Synthetic Persian Chatbot Topics Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-topics-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerQARetrieval

Synthetic Persian QA Retrieval

**Dataset:** [`MCINext/synthetic-persian-qa-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | LM-generated | LM-generated and verified |



#### SyntecRetrieval

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`lyon-nlp/mteb-fr-retrieval-syntec-s2p`](https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Legal, Written | human-annotated | created |



#### SyntheticText2SQL

The dataset is a collection of natural language queries and their corresponding sql snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/synthetic-text2sql`](https://huggingface.co/datasets/CoIR-Retrieval/synthetic-text2sql) • **License:** mit • [Learn more →](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, sql | Programming, Written | derived | found |



#### T2Retrieval

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Retrieval`](https://huggingface.co/datasets/mteb/T2Retrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2304.03679)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Academic, Financial, Government, Medical, Non-fiction | human-annotated | not specified |



#### TRECCOVID

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/trec-covid`](https://huggingface.co/datasets/mteb/trec-covid) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



#### TRECCOVID-Fa

TRECCOVID-Fa

**Dataset:** [`MCINext/trec-covid-fa`](https://huggingface.co/datasets/MCINext/trec-covid-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/trec-covid-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



#### TRECCOVID-NL

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic. TRECCOVID-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-trec-covid`](https://huggingface.co/datasets/clips/beir-nl-trec-covid) • **License:** cc-by-4.0 • [Learn more →](https://colab.research.google.com/drive/1R99rjeAGt8S9IfAIRR3wS052sNu3Bjo-#scrollTo=4HduGW6xHnrZ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



#### TRECCOVID-PL

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/TRECCOVID-PL`](https://huggingface.co/datasets/mteb/TRECCOVID-PL) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Medical, Non-fiction, Written | derived | machine-translated |



#### TRECCOVID-VN

A translated dataset from TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/trec-covid-vn`](https://huggingface.co/datasets/GreenNode/trec-covid-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



#### TV2Nordretrieval

News Article and corresponding summaries extracted from the Danish newspaper TV2 Nord.

**Dataset:** [`alexandrainst/nordjylland-news-summarization`](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | News, Non-fiction, Written | derived | found |



#### TempReasonL1

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l1.

**Dataset:** [`RAR-b/TempReason-l1`](https://huggingface.co/datasets/RAR-b/TempReason-l1) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL2Context

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-context.

**Dataset:** [`RAR-b/TempReason-l2-context`](https://huggingface.co/datasets/RAR-b/TempReason-l2-context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL2Fact

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-fact.

**Dataset:** [`RAR-b/TempReason-l2-fact`](https://huggingface.co/datasets/RAR-b/TempReason-l2-fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL2Pure

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-pure.

**Dataset:** [`RAR-b/TempReason-l2-pure`](https://huggingface.co/datasets/RAR-b/TempReason-l2-pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL3Context

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-context.

**Dataset:** [`RAR-b/TempReason-l3-context`](https://huggingface.co/datasets/RAR-b/TempReason-l3-context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL3Fact

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-fact.

**Dataset:** [`RAR-b/TempReason-l3-fact`](https://huggingface.co/datasets/RAR-b/TempReason-l3-fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL3Pure

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-pure.

**Dataset:** [`RAR-b/TempReason-l3-pure`](https://huggingface.co/datasets/RAR-b/TempReason-l3-pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TopiOCQA

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage.

**Dataset:** [`McGill-NLP/TopiOCQA`](https://huggingface.co/datasets/McGill-NLP/TopiOCQA) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### TopiOCQAHardNegatives

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/TopiOCQA_validation_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/TopiOCQA_validation_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### Touche2020

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/touche2020`](https://huggingface.co/datasets/mteb/touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



#### Touche2020-Fa

Touche2020-Fa

**Dataset:** [`MCINext/touche2020-fa`](https://huggingface.co/datasets/MCINext/touche2020-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/touche2020-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | derived | found |



#### Touche2020-NL

Touché Task 1: Argument Retrieval for Controversial Questions. Touche2020-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-webis-touche2020`](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Non-fiction | derived | machine-translated and verified |



#### Touche2020-PL

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/Touche2020-PL`](https://huggingface.co/datasets/mteb/Touche2020-PL) • **License:** not specified • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic | derived | machine-translated |



#### Touche2020-VN

A translated dataset from Touché Task 1: Argument Retrieval for Controversial Questions
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/webis-touche2020-vn`](https://huggingface.co/datasets/GreenNode/webis-touche2020-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic | derived | machine-translated and LM verified |



#### Touche2020Retrieval.v3

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/webis-touche2020-v3`](https://huggingface.co/datasets/mteb/webis-touche2020-v3) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/castorini/touche-error-analysis)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



#### TurHistQuadRetrieval

Question Answering dataset on Ottoman History in Turkish

**Dataset:** [`asparius/TurHistQuAD`](https://huggingface.co/datasets/asparius/TurHistQuAD) • **License:** mit • [Learn more →](https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | tur | Academic, Encyclopaedic, Non-fiction, Written | derived | found |



#### TwitterHjerneRetrieval

Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer.

**Dataset:** [`sorenmulli/da-hashtag-twitterhjerne`](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Social, Written | derived | found |



#### VDRMultilingualRetrieval

Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German

**Dataset:** [`llamaindex/vdr-multilingual-test`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_5 | deu, eng, fra, ita, spa | Web | LM-generated | found |



#### VideoRetrieval

VideoRetrieval

**Dataset:** [`mteb/VideoRetrieval`](https://huggingface.co/datasets/mteb/VideoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### VieQuADRetrieval

A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.

**Dataset:** [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) • **License:** mit • [Learn more →](https://aclanthology.org/2020.coling-main.233.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Non-fiction, Written | human-annotated | found |



#### WebFAQRetrieval

WebFAQ is a broad-coverage corpus of natural question-answer pairs in 75 languages, gathered from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-retrieval`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, aze, ben, bul, cat, ... (49) | Web, Written | derived | found |



#### WikiSQLRetrieval

A code retrieval task based on WikiSQL dataset with natural language questions and corresponding SQL queries. Each query is a natural language question (e.g., 'What is the name of the team that has scored the most goals?'), and the corpus contains SQL query implementations. The task is to retrieve the correct SQL query that answers the natural language question. Queries are natural language questions while the corpus contains SQL SELECT statements with proper syntax and logic for querying database tables.

**Dataset:** [`embedding-benchmark/WikiSQL_mteb`](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb) • **License:** bsd-3-clause • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, sql | Programming | expert-annotated | found |



#### WikipediaRetrievalMultilingual

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRetrievalMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRetrievalMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ben, bul, ces, dan, deu, ... (16) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified |



#### WinoGrande

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on winogrande.

**Dataset:** [`RAR-b/winogrande`](https://huggingface.co/datasets/RAR-b/winogrande) • **License:** not specified • [Learn more →](https://winogrande.allenai.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### XMarket

XMarket

**Dataset:** [`jinaai/xmarket_ml`](https://huggingface.co/datasets/jinaai/xmarket_ml) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/xmarket_ml)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu, eng, spa | not specified | not specified | not specified |



#### XPQARetrieval

XPQARetrieval

**Dataset:** [`jinaai/xpqa`](https://huggingface.co/datasets/jinaai/xpqa) • **License:** cdla-sharing-1.0 • [Learn more →](https://arxiv.org/abs/2305.09249)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, cmn, deu, eng, fra, ... (13) | Reviews, Written | human-annotated | found |



#### XQuADRetrieval

XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question.

**Dataset:** [`google/xquad`](https://huggingface.co/datasets/google/xquad) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/xquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | arb, deu, ell, eng, hin, ... (12) | Web, Written | human-annotated | created |



#### ZacLegalTextRetrieval

Zalo Legal Text documents

**Dataset:** [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) • **License:** mit • [Learn more →](https://challenge.zalo.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Legal | human-annotated | found |



#### mMARCO-NL

mMARCO is a multi-lingual (translated) collection of datasets focused on deep learning in search

**Dataset:** [`clips/beir-nl-mmarco`](https://huggingface.co/datasets/clips/beir-nl-mmarco) • **License:** apache-2.0 • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Web, Written | derived | machine-translated and verified |



## STS

- **Number of tasks of the given type:** 43 

#### AFQMC

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/AFQMC`](https://huggingface.co/datasets/C-MTEB/AFQMC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### ATEC

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/ATEC`](https://huggingface.co/datasets/C-MTEB/ATEC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### Assin2STS

Semantic Textual Similarity part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.

**Dataset:** [`nilc-nlp/assin2`](https://huggingface.co/datasets/nilc-nlp/assin2) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | por | Written | human-annotated | found |



#### BIOSSES

Biomedical Semantic Similarity Estimation.

**Dataset:** [`mteb/biosses-sts`](https://huggingface.co/datasets/mteb/biosses-sts) • **License:** not specified • [Learn more →](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Medical | derived | found |



#### BIOSSES-VN

A translated dataset from Biomedical Semantic Similarity Estimation.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/biosses-sts-vn`](https://huggingface.co/datasets/GreenNode/biosses-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | cosine_spearman | vie | Medical | derived | machine-translated and LM verified |



#### BQ

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/BQ`](https://huggingface.co/datasets/C-MTEB/BQ) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### CDSC-R

Compositional Distributional Semantics Corpus for textual relatedness.

**Dataset:** [`PL-MTEB/cdscr-sts`](https://huggingface.co/datasets/PL-MTEB/cdscr-sts) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/P17-1073.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | pol | Web, Written | human-annotated | human-translated and localized |



#### FaroeseSTS

Semantic Text Similarity (STS) corpus for Faroese.

**Dataset:** [`vesteinn/faroese-sts`](https://huggingface.co/datasets/vesteinn/faroese-sts) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.74.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fao | News, Web, Written | human-annotated | found |



#### Farsick

A Persian Semantic Textual Similarity And Natural Language Inference Dataset

**Dataset:** [`MCINext/farsick-sts`](https://huggingface.co/datasets/MCINext/farsick-sts) • **License:** not specified • [Learn more →](https://github.com/ZahraGhasemi-AI/FarSick)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fas | not specified | derived | found |



#### FinParaSTS

Finnish paraphrase-based semantic similarity corpus

**Dataset:** [`TurkuNLP/turku_paraphrase_corpus`](https://huggingface.co/datasets/TurkuNLP/turku_paraphrase_corpus) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TurkuNLP/turku_paraphrase_corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fin | News, Subtitles, Written | expert-annotated | found |



#### GermanSTSBenchmark

Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. Translations were originally done by T-Systems on site services GmbH.

**Dataset:** [`jinaai/german-STSbenchmark`](https://huggingface.co/datasets/jinaai/german-STSbenchmark) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | deu | not specified | not specified | not specified |



#### IndicCrosslingualSTS

This is a Semantic Textual Similarity testset between English and 12 high-resource Indic languages.

**Dataset:** [`mteb/IndicCrosslingualSTS`](https://huggingface.co/datasets/mteb/IndicCrosslingualSTS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jaygala24/indic_sts)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | asm, ben, eng, guj, hin, ... (13) | Government, News, Non-fiction, Spoken, Spoken, ... (7) | expert-annotated | created |



#### JSICK

JSICK is the Japanese NLI and STS dataset by manually translating the English dataset SICK (Marelli et al., 2014) into Japanese.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | jpn | Web, Written | human-annotated | found |



#### JSTS

Japanese Semantic Textual Similarity Benchmark dataset construct from YJ Image Captions Dataset (Miyazaki and Shimizu, 2016) and annotated by crowdsource annotators.

**Dataset:** [`mteb/JSTS`](https://huggingface.co/datasets/mteb/JSTS) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.317.pdf#page=2.00)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | jpn | Web, Written | human-annotated | found |



#### KLUE-STS

Human-annotated STS dataset of Korean reviews, news, and spoken word sets. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | kor | News, Reviews, Spoken, Spoken, Written | human-annotated | found |



#### KorSTS

Benchmark dataset for STS in Korean. Created by machine translation and human post editing of the STS-B dataset.

**Dataset:** [`dkoterwa/kor-sts`](https://huggingface.co/datasets/dkoterwa/kor-sts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2004.03289)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | kor | News, Web | not specified | machine-translated and localized |



#### LCQMC

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/LCQMC`](https://huggingface.co/datasets/C-MTEB/LCQMC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### PAWSX

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/PAWSX`](https://huggingface.co/datasets/C-MTEB/PAWSX) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### QBQTC



**Dataset:** [`C-MTEB/QBQTC`](https://huggingface.co/datasets/C-MTEB/QBQTC) • **License:** not specified • [Learn more →](https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### Query2Query

Query to Query Datasets.

**Dataset:** [`MCINext/query-to-query-sts`](https://huggingface.co/datasets/MCINext/query-to-query-sts) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fas | not specified | derived | found |



#### RUParaPhraserSTS

ParaPhraser is a news headlines corpus with precise, near and non-paraphrases.

**Dataset:** [`merionum/ru_paraphraser`](https://huggingface.co/datasets/merionum/ru_paraphraser) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | rus | News, Written | human-annotated | found |



#### RonSTS

High-quality Romanian translation of STSBenchmark.

**Dataset:** [`dumitrescustefan/ro_sts`](https://huggingface.co/datasets/dumitrescustefan/ro_sts) • **License:** cc-by-4.0 • [Learn more →](https://openreview.net/forum?id=JH61CD7afTv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ron | News, Social, Web, Written | human-annotated | machine-translated and verified |



#### RuSTSBenchmarkSTS

Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Russian and verified. The dataset was checked with RuCOLA model to ensure that the translation is good and filtered.

**Dataset:** [`ai-forever/ru-stsbenchmark-sts`](https://huggingface.co/datasets/ai-forever/ru-stsbenchmark-sts) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | rus | News, Social, Web, Written | human-annotated | machine-translated and verified |



#### SICK-BR-STS

SICK-BR is a Portuguese inference corpus, human translated from SICK

**Dataset:** [`eduagarcia/sick-br`](https://huggingface.co/datasets/eduagarcia/sick-br) • **License:** not specified • [Learn more →](https://linux.ime.usp.br/~thalen/SICK_PT.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | por | Web, Written | human-annotated | human-translated and localized |



#### SICK-R

Semantic Textual Similarity SICK-R dataset

**Dataset:** [`mteb/sickr-sts`](https://huggingface.co/datasets/mteb/sickr-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/L14-1314/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Web, Written | human-annotated | not specified |



#### SICK-R-PL

Polish version of SICK dataset for textual relatedness.

**Dataset:** [`PL-MTEB/sickr-pl-sts`](https://huggingface.co/datasets/PL-MTEB/sickr-pl-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | pol | Web, Written | human-annotated | human-translated and localized |



#### SICK-R-VN

A translated dataset from Semantic Textual Similarity SICK-R dataset as described here:
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/sickr-sts-vn`](https://huggingface.co/datasets/GreenNode/sickr-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | cosine_spearman | vie | Web, Written | derived | machine-translated and LM verified |



#### SICKFr

SICK dataset french version

**Dataset:** [`Lajavaness/SICK-fr`](https://huggingface.co/datasets/Lajavaness/SICK-fr) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Lajavaness/SICK-fr)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fra | not specified | not specified | not specified |



#### STS12

SemEval-2012 Task 6.

**Dataset:** [`mteb/sts12-sts`](https://huggingface.co/datasets/mteb/sts12-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S12-1051.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Encyclopaedic, News, Written | human-annotated | created |



#### STS13

SemEval STS 2013 dataset.

**Dataset:** [`mteb/sts13-sts`](https://huggingface.co/datasets/mteb/sts13-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S13-1004/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | News, Non-fiction, Web, Written | human-annotated | created |



#### STS14

SemEval STS 2014 dataset. Currently only the English dataset

**Dataset:** [`mteb/sts14-sts`](https://huggingface.co/datasets/mteb/sts14-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S14-1002)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, Spoken, Web | derived | created |



#### STS15

SemEval STS 2015 dataset

**Dataset:** [`mteb/sts15-sts`](https://huggingface.co/datasets/mteb/sts15-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S15-2010)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, News, Spoken, Web, Written | human-annotated | created |



#### STS16

SemEval-2016 Task 4

**Dataset:** [`mteb/sts16-sts`](https://huggingface.co/datasets/mteb/sts16-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S16-1001)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, Spoken, Web | human-annotated | created |



#### STS17

Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation

**Dataset:** [`mteb/sts17-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts17-crosslingual-sts) • **License:** not specified • [Learn more →](https://alt.qcri.org/semeval2017/task1/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, deu, eng, fra, ita, ... (9) | News, Web, Written | human-annotated | created |



#### STS22

SemEval 2022 Task 8: Multilingual News Article Similarity

**Dataset:** [`mteb/sts22-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, cmn, deu, eng, fra, ... (10) | News, Written | human-annotated | found |



#### STS22.v2

SemEval 2022 Task 8: Multilingual News Article Similarity. Version 2 filters updated on STS22 by removing pairs where one of entries contain empty sentences.

**Dataset:** [`mteb/sts22-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, cmn, deu, eng, fra, ... (10) | News, Written | human-annotated | found |



#### STSB

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/STSB`](https://huggingface.co/datasets/C-MTEB/STSB) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### STSBenchmark

Semantic Textual Similarity Benchmark (STSbenchmark) dataset.

**Dataset:** [`mteb/stsbenchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, News, Written | human-annotated | machine-translated and verified |



#### STSBenchmark-VN

A translated dataset from Semantic Textual Similarity Benchmark (STSbenchmark) dataset.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stsbenchmark-sts-vn`](https://huggingface.co/datasets/GreenNode/stsbenchmark-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | cosine_spearman | vie | Blog, News, Written | derived | machine-translated and LM verified |



#### STSBenchmarkMultilingualSTS

Semantic Textual Similarity Benchmark (STSbenchmark) dataset, but translated using DeepL API.

**Dataset:** [`mteb/stsb_multi_mt`](https://huggingface.co/datasets/mteb/stsb_multi_mt) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn, deu, eng, fra, ita, ... (10) | News, Social, Spoken, Web, Written | human-annotated | machine-translated |



#### STSES

Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)

**Dataset:** [`PlanTL-GOB-ES/sts-es`](https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | spa | Written | not specified | not specified |



#### SemRel24STS

SemRel2024 is a collection of Semantic Textual Relatedness (STR) datasets for 14 languages, including African and Asian languages. The datasets are composed of sentence pairs, each assigned a relatedness score between 0 (completely) unrelated and 1 (maximally related) with a large range of expected relatedness values.

**Dataset:** [`SemRel/SemRel2024`](https://huggingface.co/datasets/SemRel/SemRel2024) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SemRel/SemRel2024)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | afr, amh, arb, arq, ary, ... (12) | Spoken, Written | human-annotated | created |



#### SynPerSTS

Synthetic Persian Semantic Textual Similarity Dataset

**Dataset:** [`MCINext/synthetic-persian-sts`](https://huggingface.co/datasets/MCINext/synthetic-persian-sts) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fas | Blog, News, Religious, Web | LM-generated | LM-generated and verified |



## Summarization

- **Number of tasks of the given type:** 4 

#### SummEval

News Article Summary Semantic Similarity Estimation.

**Dataset:** [`mteb/summeval`](https://huggingface.co/datasets/mteb/summeval) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | News, Written | human-annotated | created |



#### SummEvalFr

News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.

**Dataset:** [`lyon-nlp/summarization-summeval-fr-p2p`](https://huggingface.co/datasets/lyon-nlp/summarization-summeval-fr-p2p) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fra | News, Written | human-annotated | machine-translated |



#### SummEvalFrSummarization.v2

News Article Summary Semantic Similarity Estimation translated from english to french with DeepL. This version fixes a bug in the evaluation script that caused the main score to be computed incorrectly.

**Dataset:** [`lyon-nlp/summarization-summeval-fr-p2p`](https://huggingface.co/datasets/lyon-nlp/summarization-summeval-fr-p2p) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fra | News, Written | human-annotated | machine-translated |



#### SummEvalSummarization.v2

News Article Summary Semantic Similarity Estimation. This version fixes a bug in the evaluation script that caused the main score to be computed incorrectly.

**Dataset:** [`mteb/summeval`](https://huggingface.co/datasets/mteb/summeval) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | News, Written | human-annotated | created |



## VisionCentricQA

- **Number of tasks of the given type:** 6 

#### BLINKIT2IMultiChoice

Retrieve images based on images and specific retrieval instructions.

**Dataset:** [`JamieSJS/blink-it2i-multi`](https://huggingface.co/datasets/JamieSJS/blink-it2i-multi) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2404.12390)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image (it2i) | accuracy | eng | Encyclopaedic | derived | found |



#### BLINKIT2TMultiChoice

Retrieve the correct text answer based on images and specific retrieval instructions.

**Dataset:** [`JamieSJS/blink-it2t-multi`](https://huggingface.co/datasets/JamieSJS/blink-it2t-multi) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2404.12390)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | accuracy | eng | Encyclopaedic | derived | found |



#### CVBenchCount

count the number of objects in the image.

**Dataset:** [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | accuracy | eng | Academic | derived | found |



#### CVBenchDepth

judge the depth of the objects in the image with similarity matching.

**Dataset:** [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | accuracy | eng | Academic | derived | found |



#### CVBenchDistance

judge the distance of the objects in the image with similarity matching.

**Dataset:** [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | accuracy | eng | Academic | derived | found |



#### CVBenchRelation

decide the relation of the objects in the image.

**Dataset:** [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2406.16860)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to text (it2t) | accuracy | eng | Academic | derived | found |



## VisualSTS(eng)

- **Number of tasks of the given type:** 5 

#### STS12VisualSTS

SemEval-2012 Task 6.then rendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts12`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts12) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | eng | Encyclopaedic, News, Written | human-annotated | rendered |



#### STS13VisualSTS

SemEval STS 2013 dataset.then rendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts13`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts13) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | eng | News, Non-fiction, Web, Written | human-annotated | rendered |



#### STS14VisualSTS

SemEval STS 2014 dataset. Currently only the English dataset.rendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts14`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts14) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | eng | Blog, Spoken, Web | derived | rendered |



#### STS15VisualSTS

SemEval STS 2015 datasetrendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts15`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts15) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | eng | Blog, News, Spoken, Web, Written | human-annotated | rendered |



#### STS16VisualSTS

SemEval STS 2016 datasetrendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts16`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts16) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | eng | Blog, Spoken, Web | human-annotated | rendered |



## VisualSTS(multi)

- **Number of tasks of the given type:** 2 

#### STS17MultilingualVisualSTS

Semantic Textual Similarity 17 (STS-17) dataset, rendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts17`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts17) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | ara, deu, eng, fra, ita, ... (9) | News, Social, Spoken, Web, Written | human-annotated | rendered |



#### STSBenchmarkMultilingualVisualSTS

Semantic Textual Similarity Benchmark (STSbenchmark) dataset, translated into target languages using DeepL API,then rendered into images.built upon multi-sts created by Philip May

**Dataset:** [`Pixel-Linguist/rendered-stsb`](https://huggingface.co/datasets/Pixel-Linguist/rendered-stsb) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to image (i2i) | cosine_spearman | cmn, deu, eng, fra, ita, ... (10) | News, Social, Spoken, Web, Written | human-annotated | rendered |



## ZeroShotClassification

- **Number of tasks of the given type:** 24 

#### BirdsnapZeroShot

Classifying bird images from 500 species.

**Dataset:** [`isaacchung/birdsnap`](https://huggingface.co/datasets/isaacchung/birdsnap) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2014/html/Berg_Birdsnap_Large-scale_Fine-grained_2014_CVPR_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### CIFAR100ZeroShot

Classifying images from 100 classes.

**Dataset:** [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar100)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Web | derived | created |



#### CIFAR10ZeroShot

Classifying images from 10 classes.

**Dataset:** [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/uoft-cs/cifar10)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Web | derived | created |



#### CLEVRCountZeroShot

CLEVR count objects task.

**Dataset:** [`clip-benchmark/wds_vtab-clevr_count_all`](https://huggingface.co/datasets/clip-benchmark/wds_vtab-clevr_count_all) • **License:** not specified • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Constructed | human-annotated | created |



#### CLEVRZeroShot

CLEVR closest object distance identification task.

**Dataset:** [`clip-benchmark/wds_vtab-clevr_closest_object_distance`](https://huggingface.co/datasets/clip-benchmark/wds_vtab-clevr_closest_object_distance) • **License:** cc-by-4.0 • [Learn more →](https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Constructed | human-annotated | created |



#### Caltech101ZeroShot

Classifying images of 101 widely varied objects.

**Dataset:** [`mteb/Caltech101`](https://huggingface.co/datasets/mteb/Caltech101) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/1384978)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### Country211ZeroShot

Classifying images of 211 countries.

**Dataset:** [`clip-benchmark/wds_country211`](https://huggingface.co/datasets/clip-benchmark/wds_country211) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clip-benchmark/wds_country211)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Scene | derived | created |



#### DTDZeroShot

Describable Textures Dataset in 47 categories.

**Dataset:** [`tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd) • **License:** not specified • [Learn more →](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### EuroSATZeroShot

Classifying satellite images.

**Dataset:** [`timm/eurosat-rgb`](https://huggingface.co/datasets/timm/eurosat-rgb) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8736785)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### FER2013ZeroShot

Classifying facial emotions.

**Dataset:** [`clip-benchmark/wds_fer2013`](https://huggingface.co/datasets/clip-benchmark/wds_fer2013) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1412.6572)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### FGVCAircraftZeroShot

Classifying aircraft images from 41 manufacturers and 102 variants.

**Dataset:** [`HuggingFaceM4/FGVC-Aircraft`](https://huggingface.co/datasets/HuggingFaceM4/FGVC-Aircraft) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1306.5151)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### Food101ZeroShot

Classifying food.

**Dataset:** [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/ethz/food101)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Web | derived | created |



#### GTSRBZeroShot

The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class classification dataset for traffic signs. It consists of dataset of more than 50,000 traffic sign images. The dataset comprises 43 classes with unbalanced class frequencies.

**Dataset:** [`clip-benchmark/wds_gtsrb`](https://huggingface.co/datasets/clip-benchmark/wds_gtsrb) • **License:** not specified • [Learn more →](https://benchmark.ini.rub.de/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Scene | derived | created |



#### Imagenet1kZeroShot

ImageNet, a large-scale ontology of images built upon the backbone of the WordNet structure.

**Dataset:** [`clip-benchmark/wds_imagenet1k`](https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/5206848)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Scene | human-annotated | created |



#### MNISTZeroShot

Classifying handwritten digits.

**Dataset:** [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist) • **License:** not specified • [Learn more →](https://en.wikipedia.org/wiki/MNIST_database)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### OxfordPetsZeroShot

Classifying animal images.

**Dataset:** [`isaacchung/OxfordPets`](https://huggingface.co/datasets/isaacchung/OxfordPets) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1306.5151)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### PatchCamelyonZeroShot

Histopathology diagnosis classification dataset.

**Dataset:** [`clip-benchmark/wds_vtab-pcam`](https://huggingface.co/datasets/clip-benchmark/wds_vtab-pcam) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Medical | derived | created |



#### RESISC45ZeroShot

Remote Sensing Image Scene Classification by Northwestern Polytechnical University (NWPU).

**Dataset:** [`timm/resisc45`](https://huggingface.co/datasets/timm/resisc45) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/abstract/document/7891544)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### RenderedSST2

RenderedSST2.

**Dataset:** [`clip-benchmark/wds_renderedsst2`](https://huggingface.co/datasets/clip-benchmark/wds_renderedsst2) • **License:** mit • [Learn more →](https://huggingface.co/datasets/clip-benchmark/wds_renderedsst2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Reviews | human-annotated | created |



#### STL10ZeroShot

Classifying 96x96 images from 10 classes.

**Dataset:** [`tanganke/stl10`](https://huggingface.co/datasets/tanganke/stl10) • **License:** not specified • [Learn more →](https://cs.stanford.edu/~acoates/stl10/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### SUN397ZeroShot

Large scale scene recognition in 397 categories.

**Dataset:** [`dpdl-benchmark/sun397`](https://huggingface.co/datasets/dpdl-benchmark/sun397) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/abstract/document/5539970)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Encyclopaedic | derived | created |



#### SciMMIR

SciMMIR.

**Dataset:** [`m-a-p/SciMMIR`](https://huggingface.co/datasets/m-a-p/SciMMIR) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/m-a-p/SciMMIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Academic | human-annotated | created |



#### StanfordCarsZeroShot

Classifying car images from 96 makes.

**Dataset:** [`isaacchung/StanfordCars`](https://huggingface.co/datasets/isaacchung/StanfordCars) • **License:** not specified • [Learn more →](https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Scene | derived | created |



#### UCF101ZeroShot

UCF101 is an action recognition data set of realistic
action videos collected from YouTube, having 101 action categories. This
version of the dataset does not contain images but images saved frame by
frame. Train and test splits are generated based on the authors' first
version train/test list.

**Dataset:** [`flwrlabs/ucf101`](https://huggingface.co/datasets/flwrlabs/ucf101) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/flwrlabs/ucf101)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image to text (i2t) | accuracy | eng | Scene | derived | created |
<!-- END TASK DESCRIPTION -->