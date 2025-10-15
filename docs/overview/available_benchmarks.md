# Available Benchmarks


<!-- This following is auto-generated. Changes will be overwritten. Please change the generating script. -->
<!-- START TASK DESCRIPTION -->
####  BEIR

BEIR is a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your NLP-based retrieval models within the benchmark.

[Learn more →](https://arxiv.org/abs/2104.08663)



??? info "Tasks"

    | name                                                                        | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [TRECCOVID](./available_tasks/retrieval.md#treccovid)                       | Retrieval | text         | eng         |
    | [NFCorpus](./available_tasks/retrieval.md#nfcorpus)                         | Retrieval | text         | eng         |
    | [NQ](./available_tasks/retrieval.md#nq)                                     | Retrieval | text         | eng         |
    | [HotpotQA](./available_tasks/retrieval.md#hotpotqa)                         | Retrieval | text         | eng         |
    | [FiQA2018](./available_tasks/retrieval.md#fiqa2018)                         | Retrieval | text         | eng         |
    | [ArguAna](./available_tasks/retrieval.md#arguana)                           | Retrieval | text         | eng         |
    | [Touche2020](./available_tasks/retrieval.md#touche2020)                     | Retrieval | text         | eng         |
    | [CQADupstackRetrieval](./available_tasks/retrieval.md#cqadupstackretrieval) | Retrieval | text         | eng         |
    | [QuoraRetrieval](./available_tasks/retrieval.md#quoraretrieval)             | Retrieval | text         | eng         |
    | [DBPedia](./available_tasks/retrieval.md#dbpedia)                           | Retrieval | text         | eng         |
    | [SCIDOCS](./available_tasks/retrieval.md#scidocs)                           | Retrieval | text         | eng         |
    | [FEVER](./available_tasks/retrieval.md#fever)                               | Retrieval | text         | eng         |
    | [ClimateFEVER](./available_tasks/retrieval.md#climatefever)                 | Retrieval | text         | eng         |
    | [SciFact](./available_tasks/retrieval.md#scifact)                           | Retrieval | text         | eng         |
    | [MSMARCO](./available_tasks/retrieval.md#msmarco)                           | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{thakur2021beir,
      author = {Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
      journal = {arXiv preprint arXiv:2104.08663},
      title = {Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models},
      year = {2021},
    }

    ```




####  BEIR-NL

BEIR-NL is a Dutch adaptation of the publicly available BEIR benchmark, created through automated translation.

[Learn more →](https://arxiv.org/abs/2412.08329)



??? info "Tasks"

    | name                                                              | type      | modalities   | languages   |
    |:------------------------------------------------------------------|:----------|:-------------|:------------|
    | [ArguAna-NL](./available_tasks/retrieval.md#arguana-nl)           | Retrieval | text         | nld         |
    | [CQADupstack-NL](./available_tasks/retrieval.md#cqadupstack-nl)   | Retrieval | text         | nld         |
    | [FEVER-NL](./available_tasks/retrieval.md#fever-nl)               | Retrieval | text         | nld         |
    | [NQ-NL](./available_tasks/retrieval.md#nq-nl)                     | Retrieval | text         | nld         |
    | [Touche2020-NL](./available_tasks/retrieval.md#touche2020-nl)     | Retrieval | text         | nld         |
    | [FiQA2018-NL](./available_tasks/retrieval.md#fiqa2018-nl)         | Retrieval | text         | nld         |
    | [Quora-NL](./available_tasks/retrieval.md#quora-nl)               | Retrieval | text         | nld         |
    | [HotpotQA-NL](./available_tasks/retrieval.md#hotpotqa-nl)         | Retrieval | text         | nld         |
    | [SCIDOCS-NL](./available_tasks/retrieval.md#scidocs-nl)           | Retrieval | text         | nld         |
    | [ClimateFEVER-NL](./available_tasks/retrieval.md#climatefever-nl) | Retrieval | text         | nld         |
    | [mMARCO-NL](./available_tasks/retrieval.md#mmarco-nl)             | Retrieval | text         | nld         |
    | [SciFact-NL](./available_tasks/retrieval.md#scifact-nl)           | Retrieval | text         | nld         |
    | [DBPedia-NL](./available_tasks/retrieval.md#dbpedia-nl)           | Retrieval | text         | nld         |
    | [NFCorpus-NL](./available_tasks/retrieval.md#nfcorpus-nl)         | Retrieval | text         | nld         |
    | [TRECCOVID-NL](./available_tasks/retrieval.md#treccovid-nl)       | Retrieval | text         | nld         |


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




####  BRIGHT

BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
    BRIGHT is the first text retrieval
    benchmark that requires intensive reasoning to retrieve relevant documents with
    a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
    economics, psychology, mathematics, and coding. These queries are drawn from
    naturally occurring and carefully curated human data.


[Learn more →](https://brightbenchmark.github.io/)



??? info "Tasks"

    | name                                                              | type      | modalities   | languages   |
    |:------------------------------------------------------------------|:----------|:-------------|:------------|
    | [BrightRetrieval](./available_tasks/retrieval.md#brightretrieval) | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }

    ```




####  BRIGHT (long)

BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
BRIGHT is the first text retrieval
benchmark that requires intensive reasoning to retrieve relevant documents with
a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
economics, psychology, mathematics, and coding. These queries are drawn from
naturally occurring and carefully curated human data.

This is the long version of the benchmark, which only filter longer documents.


[Learn more →](https://brightbenchmark.github.io/)



??? info "Tasks"

    | name                                                                      | type      | modalities   | languages   |
    |:--------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [BrightLongRetrieval](./available_tasks/retrieval.md#brightlongretrieval) | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{su2024bright,
      author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
      journal = {arXiv preprint arXiv:2407.12883},
      title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
      year = {2024},
    }

    ```




####  BuiltBench(eng)

"Built-Bench" is an ongoing effort aimed at evaluating text embedding models in the context of built asset management, spanning over various dicsiplines such as architeture, engineering, constrcution, and operations management of the built environment.

[Learn more →](https://arxiv.org/abs/2411.12056)



??? info "Tasks"

    | name                                                                               | type       | modalities   | languages   |
    |:-----------------------------------------------------------------------------------|:-----------|:-------------|:------------|
    | [BuiltBenchClusteringP2P](./available_tasks/clustering.md#builtbenchclusteringp2p) | Clustering | text         | eng         |
    | [BuiltBenchClusteringS2S](./available_tasks/clustering.md#builtbenchclusterings2s) | Clustering | text         | eng         |
    | [BuiltBenchRetrieval](./available_tasks/retrieval.md#builtbenchretrieval)          | Retrieval  | text         | eng         |
    | [BuiltBenchReranking](./available_tasks/reranking.md#builtbenchreranking)          | Reranking  | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }

    ```




####  ChemTEB

ChemTEB evaluates the performance of text embedding models on chemical domain data.

[Learn more →](https://arxiv.org/abs/2412.00532)



??? info "Tasks"

    | name                                                                                                                                     | type               | modalities   | languages                         |
    |:-----------------------------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [PubChemSMILESBitextMining](./available_tasks/bitextmining.md#pubchemsmilesbitextmining)                                                 | BitextMining       | text         | eng                               |
    | [SDSEyeProtectionClassification](./available_tasks/classification.md#sdseyeprotectionclassification)                                     | Classification     | text         | eng                               |
    | [SDSGlovesClassification](./available_tasks/classification.md#sdsglovesclassification)                                                   | Classification     | text         | eng                               |
    | [WikipediaBioMetChemClassification](./available_tasks/classification.md#wikipediabiometchemclassification)                               | Classification     | text         | eng                               |
    | [WikipediaGreenhouseEnantiopureClassification](./available_tasks/classification.md#wikipediagreenhouseenantiopureclassification)         | Classification     | text         | eng                               |
    | [WikipediaSolidStateColloidalClassification](./available_tasks/classification.md#wikipediasolidstatecolloidalclassification)             | Classification     | text         | eng                               |
    | [WikipediaOrganicInorganicClassification](./available_tasks/classification.md#wikipediaorganicinorganicclassification)                   | Classification     | text         | eng                               |
    | [WikipediaCryobiologySeparationClassification](./available_tasks/classification.md#wikipediacryobiologyseparationclassification)         | Classification     | text         | eng                               |
    | [WikipediaChemistryTopicsClassification](./available_tasks/classification.md#wikipediachemistrytopicsclassification)                     | Classification     | text         | eng                               |
    | [WikipediaTheoreticalAppliedClassification](./available_tasks/classification.md#wikipediatheoreticalappliedclassification)               | Classification     | text         | eng                               |
    | [WikipediaChemFieldsClassification](./available_tasks/classification.md#wikipediachemfieldsclassification)                               | Classification     | text         | eng                               |
    | [WikipediaLuminescenceClassification](./available_tasks/classification.md#wikipedialuminescenceclassification)                           | Classification     | text         | eng                               |
    | [WikipediaIsotopesFissionClassification](./available_tasks/classification.md#wikipediaisotopesfissionclassification)                     | Classification     | text         | eng                               |
    | [WikipediaSaltsSemiconductorsClassification](./available_tasks/classification.md#wikipediasaltssemiconductorsclassification)             | Classification     | text         | eng                               |
    | [WikipediaBiolumNeurochemClassification](./available_tasks/classification.md#wikipediabiolumneurochemclassification)                     | Classification     | text         | eng                               |
    | [WikipediaCrystallographyAnalyticalClassification](./available_tasks/classification.md#wikipediacrystallographyanalyticalclassification) | Classification     | text         | eng                               |
    | [WikipediaCompChemSpectroscopyClassification](./available_tasks/classification.md#wikipediacompchemspectroscopyclassification)           | Classification     | text         | eng                               |
    | [WikipediaChemEngSpecialtiesClassification](./available_tasks/classification.md#wikipediachemengspecialtiesclassification)               | Classification     | text         | eng                               |
    | [WikipediaChemistryTopicsClustering](./available_tasks/clustering.md#wikipediachemistrytopicsclustering)                                 | Clustering         | text         | eng                               |
    | [WikipediaSpecialtiesInChemistryClustering](./available_tasks/clustering.md#wikipediaspecialtiesinchemistryclustering)                   | Clustering         | text         | eng                               |
    | [PubChemAISentenceParaphrasePC](./available_tasks/pairclassification.md#pubchemaisentenceparaphrasepc)                                   | PairClassification | text         | eng                               |
    | [PubChemSMILESPC](./available_tasks/pairclassification.md#pubchemsmilespc)                                                               | PairClassification | text         | eng                               |
    | [PubChemSynonymPC](./available_tasks/pairclassification.md#pubchemsynonympc)                                                             | PairClassification | text         | eng                               |
    | [PubChemWikiParagraphsPC](./available_tasks/pairclassification.md#pubchemwikiparagraphspc)                                               | PairClassification | text         | eng                               |
    | [PubChemWikiPairClassification](./available_tasks/pairclassification.md#pubchemwikipairclassification)                                   | PairClassification | text         | ces, deu, eng, fra, hin, ... (13) |
    | [ChemNQRetrieval](./available_tasks/retrieval.md#chemnqretrieval)                                                                        | Retrieval          | text         | eng                               |
    | [ChemHotpotQARetrieval](./available_tasks/retrieval.md#chemhotpotqaretrieval)                                                            | Retrieval          | text         | eng                               |


??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \\& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




####  CoIR

CoIR: A Comprehensive Benchmark for Code Information Retrieval Models

[Learn more →](https://github.com/CoIR-team/coir)



??? info "Tasks"

    | name                                                                                    | type      | modalities   | languages                                  |
    |:----------------------------------------------------------------------------------------|:----------|:-------------|:-------------------------------------------|
    | [AppsRetrieval](./available_tasks/retrieval.md#appsretrieval)                           | Retrieval | text         | eng, python                                |
    | [CodeFeedbackMT](./available_tasks/retrieval.md#codefeedbackmt)                         | Retrieval | text         | eng                                        |
    | [CodeFeedbackST](./available_tasks/retrieval.md#codefeedbackst)                         | Retrieval | text         | eng                                        |
    | [CodeSearchNetCCRetrieval](./available_tasks/retrieval.md#codesearchnetccretrieval)     | Retrieval | text         | go, java, javascript, php, python, ... (6) |
    | [CodeTransOceanContest](./available_tasks/retrieval.md#codetransoceancontest)           | Retrieval | text         | c++, python                                |
    | [CodeTransOceanDL](./available_tasks/retrieval.md#codetransoceandl)                     | Retrieval | text         | python                                     |
    | [CosQA](./available_tasks/retrieval.md#cosqa)                                           | Retrieval | text         | eng, python                                |
    | [COIRCodeSearchNetRetrieval](./available_tasks/retrieval.md#coircodesearchnetretrieval) | Retrieval | text         | go, java, javascript, php, python, ... (6) |
    | [StackOverflowQA](./available_tasks/retrieval.md#stackoverflowqa)                       | Retrieval | text         | eng                                        |
    | [SyntheticText2SQL](./available_tasks/retrieval.md#synthetictext2sql)                   | Retrieval | text         | eng, sql                                   |


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




####  CodeRAG

A benchmark for evaluating code retrieval augmented generation, testing models' ability to retrieve relevant programming solutions, tutorials and documentation.

[Learn more →](https://arxiv.org/abs/2406.14497)



??? info "Tasks"

    | name                                                                                                        | type      | modalities   | languages   |
    |:------------------------------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [CodeRAGLibraryDocumentationSolutions](./available_tasks/reranking.md#coderaglibrarydocumentationsolutions) | Reranking | text         | python      |
    | [CodeRAGOnlineTutorials](./available_tasks/reranking.md#coderagonlinetutorials)                             | Reranking | text         | python      |
    | [CodeRAGProgrammingSolutions](./available_tasks/reranking.md#coderagprogrammingsolutions)                   | Reranking | text         | python      |
    | [CodeRAGStackoverflowPosts](./available_tasks/reranking.md#coderagstackoverflowposts)                       | Reranking | text         | python      |


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




####  Encodechka

A benchmark for evaluating text embedding models on Russian data.

[Learn more →](https://github.com/avidale/encodechka)



??? info "Tasks"

    | name                                                                                                       | type               | modalities   | languages                         |
    |:-----------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [RUParaPhraserSTS](./available_tasks/sts.md#ruparaphrasersts)                                              | STS                | text         | rus                               |
    | [SentiRuEval2016](./available_tasks/classification.md#sentirueval2016)                                     | Classification     | text         | rus                               |
    | [RuToxicOKMLCUPClassification](./available_tasks/classification.md#rutoxicokmlcupclassification)           | Classification     | text         | rus                               |
    | [InappropriatenessClassificationv2](./available_tasks/classification.md#inappropriatenessclassificationv2) | Classification     | text         | rus                               |
    | [RuNLUIntentClassification](./available_tasks/classification.md#runluintentclassification)                 | Classification     | text         | rus                               |
    | [XNLI](./available_tasks/pairclassification.md#xnli)                                                       | PairClassification | text         | ara, bul, deu, ell, eng, ... (14) |
    | [RuSTSBenchmarkSTS](./available_tasks/sts.md#rustsbenchmarksts)                                            | STS                | text         | rus                               |


??? quote "Citation"


    ```bibtex

    @misc{dale_encodechka,
      author = {Dale, David},
      editor = {habr.com},
      month = {June},
      note = {[Online; posted 12-June-2022]},
      title = {Russian rating of sentence encoders},
      url = {https://habr.com/ru/articles/669674/},
      year = {2022},
    }

    ```




####  FollowIR

Retrieval w/Instructions is the task of finding relevant documents for a query that has detailed instructions.

[Learn more →](https://arxiv.org/abs/2403.15246)



??? info "Tasks"

    | name                                                                                                   | type                 | modalities   | languages   |
    |:-------------------------------------------------------------------------------------------------------|:---------------------|:-------------|:------------|
    | [Robust04InstructionRetrieval](./available_tasks/instructionreranking.md#robust04instructionretrieval) | InstructionReranking | text         | eng         |
    | [News21InstructionRetrieval](./available_tasks/instructionreranking.md#news21instructionretrieval)     | InstructionReranking | text         | eng         |
    | [Core17InstructionRetrieval](./available_tasks/instructionreranking.md#core17instructionretrieval)     | InstructionReranking | text         | eng         |


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




####  HUME(v1)

The HUME benchmark is designed to evaluate the performance of text embedding models and humans on a comparable set of tasks. This captures areas where models perform better than human annotators and the reverse. In the paper, we go further into the analysis and what conclusions can be drawn.

[Learn more →](Coming soon (in review))



??? info "Tasks"

    | name                                                                                                                         | type           | modalities   | languages               |
    |:-----------------------------------------------------------------------------------------------------------------------------|:---------------|:-------------|:------------------------|
    | [HUMEEmotionClassification](./available_tasks/classification.md#humeemotionclassification)                                   | Classification | text         | eng                     |
    | [HUMEToxicConversationsClassification](./available_tasks/classification.md#humetoxicconversationsclassification)             | Classification | text         | eng                     |
    | [HUMETweetSentimentExtractionClassification](./available_tasks/classification.md#humetweetsentimentextractionclassification) | Classification | text         | eng                     |
    | [HUMEMultilingualSentimentClassification](./available_tasks/classification.md#humemultilingualsentimentclassification)       | Classification | text         | ara, eng, nob, rus      |
    | [HUMEArxivClusteringP2P](./available_tasks/clustering.md#humearxivclusteringp2p)                                             | Clustering     | text         | eng                     |
    | [HUMERedditClusteringP2P](./available_tasks/clustering.md#humeredditclusteringp2p)                                           | Clustering     | text         | eng                     |
    | [HUMEWikiCitiesClustering](./available_tasks/clustering.md#humewikicitiesclustering)                                         | Clustering     | text         | eng                     |
    | [HUMESIB200ClusteringS2S](./available_tasks/clustering.md#humesib200clusterings2s)                                           | Clustering     | text         | ara, dan, eng, fra, rus |
    | [HUMECore17InstructionReranking](./available_tasks/reranking.md#humecore17instructionreranking)                              | Reranking      | text         | eng                     |
    | [HUMENews21InstructionReranking](./available_tasks/reranking.md#humenews21instructionreranking)                              | Reranking      | text         | eng                     |
    | [HUMERobust04InstructionReranking](./available_tasks/reranking.md#humerobust04instructionreranking)                          | Reranking      | text         | eng                     |
    | [HUMEWikipediaRerankingMultilingual](./available_tasks/reranking.md#humewikipediarerankingmultilingual)                      | Reranking      | text         | dan, eng, nob           |
    | [HUMESICK-R](./available_tasks/sts.md#humesick-r)                                                                            | STS            | text         | eng                     |
    | [HUMESTS12](./available_tasks/sts.md#humests12)                                                                              | STS            | text         | eng                     |
    | [HUMESTSBenchmark](./available_tasks/sts.md#humestsbenchmark)                                                                | STS            | text         | eng                     |
    | [HUMESTS22](./available_tasks/sts.md#humests22)                                                                              | STS            | text         | ara, eng, fra, rus      |


####  JinaVDR

Multilingual, domain-diverse and layout-rich document retrieval benchmark.

[Learn more →](https://arxiv.org/abs/2506.18902)



??? info "Tasks"

    | name                                                                                                                                | type                  | modalities   | languages                         |
    |:------------------------------------------------------------------------------------------------------------------------------------|:----------------------|:-------------|:----------------------------------|
    | [JinaVDRMedicalPrescriptionsRetrieval](./available_tasks/documentunderstanding.md#jinavdrmedicalprescriptionsretrieval)             | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRStanfordSlideRetrieval](./available_tasks/documentunderstanding.md#jinavdrstanfordslideretrieval)                           | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRDonutVQAISynHMPRetrieval](./available_tasks/documentunderstanding.md#jinavdrdonutvqaisynhmpretrieval)                       | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRTableVQARetrieval](./available_tasks/documentunderstanding.md#jinavdrtablevqaretrieval)                                     | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRChartQARetrieval](./available_tasks/documentunderstanding.md#jinavdrchartqaretrieval)                                       | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRTQARetrieval](./available_tasks/documentunderstanding.md#jinavdrtqaretrieval)                                               | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDROpenAINewsRetrieval](./available_tasks/documentunderstanding.md#jinavdropenainewsretrieval)                                 | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDREuropeanaDeNewsRetrieval](./available_tasks/documentunderstanding.md#jinavdreuropeanadenewsretrieval)                       | DocumentUnderstanding | text, image  | deu                               |
    | [JinaVDREuropeanaEsNewsRetrieval](./available_tasks/documentunderstanding.md#jinavdreuropeanaesnewsretrieval)                       | DocumentUnderstanding | text, image  | spa                               |
    | [JinaVDREuropeanaItScansRetrieval](./available_tasks/documentunderstanding.md#jinavdreuropeanaitscansretrieval)                     | DocumentUnderstanding | text, image  | ita                               |
    | [JinaVDREuropeanaNlLegalRetrieval](./available_tasks/documentunderstanding.md#jinavdreuropeananllegalretrieval)                     | DocumentUnderstanding | text, image  | nld                               |
    | [JinaVDRHindiGovVQARetrieval](./available_tasks/documentunderstanding.md#jinavdrhindigovvqaretrieval)                               | DocumentUnderstanding | text, image  | hin                               |
    | [JinaVDRAutomobileCatelogRetrieval](./available_tasks/documentunderstanding.md#jinavdrautomobilecatelogretrieval)                   | DocumentUnderstanding | text, image  | jpn                               |
    | [JinaVDRBeveragesCatalogueRetrieval](./available_tasks/documentunderstanding.md#jinavdrbeveragescatalogueretrieval)                 | DocumentUnderstanding | text, image  | rus                               |
    | [JinaVDRRamensBenchmarkRetrieval](./available_tasks/documentunderstanding.md#jinavdrramensbenchmarkretrieval)                       | DocumentUnderstanding | text, image  | jpn                               |
    | [JinaVDRJDocQARetrieval](./available_tasks/documentunderstanding.md#jinavdrjdocqaretrieval)                                         | DocumentUnderstanding | text, image  | jpn                               |
    | [JinaVDRHungarianDocQARetrieval](./available_tasks/documentunderstanding.md#jinavdrhungariandocqaretrieval)                         | DocumentUnderstanding | text, image  | hun                               |
    | [JinaVDRArabicChartQARetrieval](./available_tasks/documentunderstanding.md#jinavdrarabicchartqaretrieval)                           | DocumentUnderstanding | text, image  | ara                               |
    | [JinaVDRArabicInfographicsVQARetrieval](./available_tasks/documentunderstanding.md#jinavdrarabicinfographicsvqaretrieval)           | DocumentUnderstanding | text, image  | ara                               |
    | [JinaVDROWIDChartsRetrieval](./available_tasks/documentunderstanding.md#jinavdrowidchartsretrieval)                                 | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRMPMQARetrieval](./available_tasks/documentunderstanding.md#jinavdrmpmqaretrieval)                                           | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRJina2024YearlyBookRetrieval](./available_tasks/documentunderstanding.md#jinavdrjina2024yearlybookretrieval)                 | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRWikimediaCommonsMapsRetrieval](./available_tasks/documentunderstanding.md#jinavdrwikimediacommonsmapsretrieval)             | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRPlotQARetrieval](./available_tasks/documentunderstanding.md#jinavdrplotqaretrieval)                                         | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRMMTabRetrieval](./available_tasks/documentunderstanding.md#jinavdrmmtabretrieval)                                           | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRCharXivOCRRetrieval](./available_tasks/documentunderstanding.md#jinavdrcharxivocrretrieval)                                 | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRStudentEnrollmentSyntheticRetrieval](./available_tasks/documentunderstanding.md#jinavdrstudentenrollmentsyntheticretrieval) | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRGitHubReadmeRetrieval](./available_tasks/documentunderstanding.md#jinavdrgithubreadmeretrieval)                             | DocumentUnderstanding | text, image  | ara, ben, deu, eng, fra, ... (17) |
    | [JinaVDRTweetStockSyntheticsRetrieval](./available_tasks/documentunderstanding.md#jinavdrtweetstocksyntheticsretrieval)             | DocumentUnderstanding | text, image  | ara, deu, eng, fra, hin, ... (10) |
    | [JinaVDRAirbnbSyntheticRetrieval](./available_tasks/documentunderstanding.md#jinavdrairbnbsyntheticretrieval)                       | DocumentUnderstanding | text, image  | ara, deu, eng, fra, hin, ... (10) |
    | [JinaVDRShanghaiMasterPlanRetrieval](./available_tasks/documentunderstanding.md#jinavdrshanghaimasterplanretrieval)                 | DocumentUnderstanding | text, image  | zho                               |
    | [JinaVDRWikimediaCommonsDocumentsRetrieval](./available_tasks/documentunderstanding.md#jinavdrwikimediacommonsdocumentsretrieval)   | DocumentUnderstanding | text, image  | ara, ben, deu, eng, fra, ... (20) |
    | [JinaVDREuropeanaFrNewsRetrieval](./available_tasks/documentunderstanding.md#jinavdreuropeanafrnewsretrieval)                       | DocumentUnderstanding | text, image  | fra                               |
    | [JinaVDRDocQAHealthcareIndustryRetrieval](./available_tasks/documentunderstanding.md#jinavdrdocqahealthcareindustryretrieval)       | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRDocQAAI](./available_tasks/documentunderstanding.md#jinavdrdocqaai)                                                         | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRShiftProjectRetrieval](./available_tasks/documentunderstanding.md#jinavdrshiftprojectretrieval)                             | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRTatQARetrieval](./available_tasks/documentunderstanding.md#jinavdrtatqaretrieval)                                           | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRInfovqaRetrieval](./available_tasks/documentunderstanding.md#jinavdrinfovqaretrieval)                                       | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRDocVQARetrieval](./available_tasks/documentunderstanding.md#jinavdrdocvqaretrieval)                                         | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRDocQAGovReportRetrieval](./available_tasks/documentunderstanding.md#jinavdrdocqagovreportretrieval)                         | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRTabFQuadRetrieval](./available_tasks/documentunderstanding.md#jinavdrtabfquadretrieval)                                     | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRDocQAEnergyRetrieval](./available_tasks/documentunderstanding.md#jinavdrdocqaenergyretrieval)                               | DocumentUnderstanding | text, image  | eng                               |
    | [JinaVDRArxivQARetrieval](./available_tasks/documentunderstanding.md#jinavdrarxivqaretrieval)                                       | DocumentUnderstanding | text, image  | eng                               |


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




####  LongEmbed

LongEmbed is a benchmark oriented at exploring models' performance on long-context retrieval.
    The benchmark comprises two synthetic tasks and four carefully chosen real-world tasks,
    featuring documents of varying length and dispersed target information.


[Learn more →](https://arxiv.org/abs/2404.12096v2)



??? info "Tasks"

    | name                                                                                  | type      | modalities   | languages   |
    |:--------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [LEMBNarrativeQARetrieval](./available_tasks/retrieval.md#lembnarrativeqaretrieval)   | Retrieval | text         | eng         |
    | [LEMBNeedleRetrieval](./available_tasks/retrieval.md#lembneedleretrieval)             | Retrieval | text         | eng         |
    | [LEMBPasskeyRetrieval](./available_tasks/retrieval.md#lembpasskeyretrieval)           | Retrieval | text         | eng         |
    | [LEMBQMSumRetrieval](./available_tasks/retrieval.md#lembqmsumretrieval)               | Retrieval | text         | eng         |
    | [LEMBSummScreenFDRetrieval](./available_tasks/retrieval.md#lembsummscreenfdretrieval) | Retrieval | text         | eng         |
    | [LEMBWikimQARetrieval](./available_tasks/retrieval.md#lembwikimqaretrieval)           | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{zhu2024longembed,
      author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
      journal = {arXiv preprint arXiv:2404.12096},
      title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
      year = {2024},
    }

    ```




####  MIEB(Img)

A image-only version of MIEB(Multilingual) that consists of 49 tasks.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info "Tasks"

    | name                                                                                                         | type                | modalities   | languages                         |
    |:-------------------------------------------------------------------------------------------------------------|:--------------------|:-------------|:----------------------------------|
    | [CUB200I2IRetrieval](./available_tasks/any2anyretrieval.md#cub200i2iretrieval)                               | Any2AnyRetrieval    | image        | eng                               |
    | [FORBI2IRetrieval](./available_tasks/any2anyretrieval.md#forbi2iretrieval)                                   | Any2AnyRetrieval    | image        | eng                               |
    | [GLDv2I2IRetrieval](./available_tasks/any2anyretrieval.md#gldv2i2iretrieval)                                 | Any2AnyRetrieval    | image        | eng                               |
    | [METI2IRetrieval](./available_tasks/any2anyretrieval.md#meti2iretrieval)                                     | Any2AnyRetrieval    | image        | eng                               |
    | [NIGHTSI2IRetrieval](./available_tasks/any2anyretrieval.md#nightsi2iretrieval)                               | Any2AnyRetrieval    | image        | eng                               |
    | [ROxfordEasyI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordeasyi2iretrieval)                     | Any2AnyRetrieval    | image        | eng                               |
    | [ROxfordMediumI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordmediumi2iretrieval)                 | Any2AnyRetrieval    | image        | eng                               |
    | [ROxfordHardI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordhardi2iretrieval)                     | Any2AnyRetrieval    | image        | eng                               |
    | [RP2kI2IRetrieval](./available_tasks/any2anyretrieval.md#rp2ki2iretrieval)                                   | Any2AnyRetrieval    | image        | eng                               |
    | [RParisEasyI2IRetrieval](./available_tasks/any2anyretrieval.md#rpariseasyi2iretrieval)                       | Any2AnyRetrieval    | image        | eng                               |
    | [RParisMediumI2IRetrieval](./available_tasks/any2anyretrieval.md#rparismediumi2iretrieval)                   | Any2AnyRetrieval    | image        | eng                               |
    | [RParisHardI2IRetrieval](./available_tasks/any2anyretrieval.md#rparishardi2iretrieval)                       | Any2AnyRetrieval    | image        | eng                               |
    | [SketchyI2IRetrieval](./available_tasks/any2anyretrieval.md#sketchyi2iretrieval)                             | Any2AnyRetrieval    | image        | eng                               |
    | [SOPI2IRetrieval](./available_tasks/any2anyretrieval.md#sopi2iretrieval)                                     | Any2AnyRetrieval    | image        | eng                               |
    | [StanfordCarsI2IRetrieval](./available_tasks/any2anyretrieval.md#stanfordcarsi2iretrieval)                   | Any2AnyRetrieval    | image        | eng                               |
    | [Birdsnap](./available_tasks/imageclassification.md#birdsnap)                                                | ImageClassification | image        | eng                               |
    | [Caltech101](./available_tasks/imageclassification.md#caltech101)                                            | ImageClassification | image        | eng                               |
    | [CIFAR10](./available_tasks/imageclassification.md#cifar10)                                                  | ImageClassification | image        | eng                               |
    | [CIFAR100](./available_tasks/imageclassification.md#cifar100)                                                | ImageClassification | image        | eng                               |
    | [Country211](./available_tasks/imageclassification.md#country211)                                            | ImageClassification | image        | eng                               |
    | [DTD](./available_tasks/imageclassification.md#dtd)                                                          | ImageClassification | image        | eng                               |
    | [EuroSAT](./available_tasks/imageclassification.md#eurosat)                                                  | ImageClassification | image        | eng                               |
    | [FER2013](./available_tasks/imageclassification.md#fer2013)                                                  | ImageClassification | image        | eng                               |
    | [FGVCAircraft](./available_tasks/imageclassification.md#fgvcaircraft)                                        | ImageClassification | image        | eng                               |
    | [Food101Classification](./available_tasks/imageclassification.md#food101classification)                      | ImageClassification | image        | eng                               |
    | [GTSRB](./available_tasks/imageclassification.md#gtsrb)                                                      | ImageClassification | image        | eng                               |
    | [Imagenet1k](./available_tasks/imageclassification.md#imagenet1k)                                            | ImageClassification | image        | eng                               |
    | [MNIST](./available_tasks/imageclassification.md#mnist)                                                      | ImageClassification | image        | eng                               |
    | [OxfordFlowersClassification](./available_tasks/imageclassification.md#oxfordflowersclassification)          | ImageClassification | image        | eng                               |
    | [OxfordPets](./available_tasks/imageclassification.md#oxfordpets)                                            | ImageClassification | image        | eng                               |
    | [PatchCamelyon](./available_tasks/imageclassification.md#patchcamelyon)                                      | ImageClassification | image        | eng                               |
    | [RESISC45](./available_tasks/imageclassification.md#resisc45)                                                | ImageClassification | image        | eng                               |
    | [StanfordCars](./available_tasks/imageclassification.md#stanfordcars)                                        | ImageClassification | image        | eng                               |
    | [STL10](./available_tasks/imageclassification.md#stl10)                                                      | ImageClassification | image        | eng                               |
    | [SUN397](./available_tasks/imageclassification.md#sun397)                                                    | ImageClassification | image        | eng                               |
    | [UCF101](./available_tasks/imageclassification.md#ucf101)                                                    | ImageClassification | image        | eng                               |
    | [CIFAR10Clustering](./available_tasks/imageclustering.md#cifar10clustering)                                  | ImageClustering     | image        | eng                               |
    | [CIFAR100Clustering](./available_tasks/imageclustering.md#cifar100clustering)                                | ImageClustering     | image        | eng                               |
    | [ImageNetDog15Clustering](./available_tasks/imageclustering.md#imagenetdog15clustering)                      | ImageClustering     | image        | eng                               |
    | [ImageNet10Clustering](./available_tasks/imageclustering.md#imagenet10clustering)                            | ImageClustering     | image        | eng                               |
    | [TinyImageNetClustering](./available_tasks/imageclustering.md#tinyimagenetclustering)                        | ImageClustering     | image        | eng                               |
    | [VOC2007](./available_tasks/imageclassification.md#voc2007)                                                  | ImageClassification | image        | eng                               |
    | [STS12VisualSTS](./available_tasks/visualsts(eng).md#sts12visualsts)                                         | VisualSTS(eng)      | image        | eng                               |
    | [STS13VisualSTS](./available_tasks/visualsts(eng).md#sts13visualsts)                                         | VisualSTS(eng)      | image        | eng                               |
    | [STS14VisualSTS](./available_tasks/visualsts(eng).md#sts14visualsts)                                         | VisualSTS(eng)      | image        | eng                               |
    | [STS15VisualSTS](./available_tasks/visualsts(eng).md#sts15visualsts)                                         | VisualSTS(eng)      | image        | eng                               |
    | [STS16VisualSTS](./available_tasks/visualsts(eng).md#sts16visualsts)                                         | VisualSTS(eng)      | image        | eng                               |
    | [STS17MultilingualVisualSTS](./available_tasks/visualsts(multi).md#sts17multilingualvisualsts)               | VisualSTS(multi)    | image        | ara, deu, eng, fra, ita, ... (9)  |
    | [STSBenchmarkMultilingualVisualSTS](./available_tasks/visualsts(multi).md#stsbenchmarkmultilingualvisualsts) | VisualSTS(multi)    | image        | cmn, deu, eng, fra, ita, ... (10) |


??? quote "Citation"


    ```bibtex

    @article{xiao2025mieb,
      author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
      doi = {10.48550/ARXIV.2504.10471},
      journal = {arXiv preprint arXiv:2504.10471},
      publisher = {arXiv},
      title = {MIEB: Massive Image Embedding Benchmark},
      url = {https://arxiv.org/abs/2504.10471},
      year = {2025},
    }

    ```




####  MIEB(Multilingual)

MIEB(Multilingual) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 130 tasks and a total of 39 languages.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document undestanding, visual STS, and CV-centric tasks. This benchmark consists of MIEB(eng) + 3 multilingual retrieval
    datasets + the multilingual parts of VisualSTS-b and VisualSTS-16.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info "Tasks"

    | name                                                                                                                                          | type                         | modalities   | languages                         |
    |:----------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|:-------------|:----------------------------------|
    | [Birdsnap](./available_tasks/imageclassification.md#birdsnap)                                                                                 | ImageClassification          | image        | eng                               |
    | [Caltech101](./available_tasks/imageclassification.md#caltech101)                                                                             | ImageClassification          | image        | eng                               |
    | [CIFAR10](./available_tasks/imageclassification.md#cifar10)                                                                                   | ImageClassification          | image        | eng                               |
    | [CIFAR100](./available_tasks/imageclassification.md#cifar100)                                                                                 | ImageClassification          | image        | eng                               |
    | [Country211](./available_tasks/imageclassification.md#country211)                                                                             | ImageClassification          | image        | eng                               |
    | [DTD](./available_tasks/imageclassification.md#dtd)                                                                                           | ImageClassification          | image        | eng                               |
    | [EuroSAT](./available_tasks/imageclassification.md#eurosat)                                                                                   | ImageClassification          | image        | eng                               |
    | [FER2013](./available_tasks/imageclassification.md#fer2013)                                                                                   | ImageClassification          | image        | eng                               |
    | [FGVCAircraft](./available_tasks/imageclassification.md#fgvcaircraft)                                                                         | ImageClassification          | image        | eng                               |
    | [Food101Classification](./available_tasks/imageclassification.md#food101classification)                                                       | ImageClassification          | image        | eng                               |
    | [GTSRB](./available_tasks/imageclassification.md#gtsrb)                                                                                       | ImageClassification          | image        | eng                               |
    | [Imagenet1k](./available_tasks/imageclassification.md#imagenet1k)                                                                             | ImageClassification          | image        | eng                               |
    | [MNIST](./available_tasks/imageclassification.md#mnist)                                                                                       | ImageClassification          | image        | eng                               |
    | [OxfordFlowersClassification](./available_tasks/imageclassification.md#oxfordflowersclassification)                                           | ImageClassification          | image        | eng                               |
    | [OxfordPets](./available_tasks/imageclassification.md#oxfordpets)                                                                             | ImageClassification          | image        | eng                               |
    | [PatchCamelyon](./available_tasks/imageclassification.md#patchcamelyon)                                                                       | ImageClassification          | image        | eng                               |
    | [RESISC45](./available_tasks/imageclassification.md#resisc45)                                                                                 | ImageClassification          | image        | eng                               |
    | [StanfordCars](./available_tasks/imageclassification.md#stanfordcars)                                                                         | ImageClassification          | image        | eng                               |
    | [STL10](./available_tasks/imageclassification.md#stl10)                                                                                       | ImageClassification          | image        | eng                               |
    | [SUN397](./available_tasks/imageclassification.md#sun397)                                                                                     | ImageClassification          | image        | eng                               |
    | [UCF101](./available_tasks/imageclassification.md#ucf101)                                                                                     | ImageClassification          | image        | eng                               |
    | [VOC2007](./available_tasks/imageclassification.md#voc2007)                                                                                   | ImageClassification          | image        | eng                               |
    | [CIFAR10Clustering](./available_tasks/imageclustering.md#cifar10clustering)                                                                   | ImageClustering              | image        | eng                               |
    | [CIFAR100Clustering](./available_tasks/imageclustering.md#cifar100clustering)                                                                 | ImageClustering              | image        | eng                               |
    | [ImageNetDog15Clustering](./available_tasks/imageclustering.md#imagenetdog15clustering)                                                       | ImageClustering              | image        | eng                               |
    | [ImageNet10Clustering](./available_tasks/imageclustering.md#imagenet10clustering)                                                             | ImageClustering              | image        | eng                               |
    | [TinyImageNetClustering](./available_tasks/imageclustering.md#tinyimagenetclustering)                                                         | ImageClustering              | image        | eng                               |
    | [BirdsnapZeroShot](./available_tasks/zeroshotclassification.md#birdsnapzeroshot)                                                              | ZeroShotClassification       | image, text  | eng                               |
    | [Caltech101ZeroShot](./available_tasks/zeroshotclassification.md#caltech101zeroshot)                                                          | ZeroShotClassification       | text, image  | eng                               |
    | [CIFAR10ZeroShot](./available_tasks/zeroshotclassification.md#cifar10zeroshot)                                                                | ZeroShotClassification       | text, image  | eng                               |
    | [CIFAR100ZeroShot](./available_tasks/zeroshotclassification.md#cifar100zeroshot)                                                              | ZeroShotClassification       | text, image  | eng                               |
    | [CLEVRZeroShot](./available_tasks/zeroshotclassification.md#clevrzeroshot)                                                                    | ZeroShotClassification       | text, image  | eng                               |
    | [CLEVRCountZeroShot](./available_tasks/zeroshotclassification.md#clevrcountzeroshot)                                                          | ZeroShotClassification       | text, image  | eng                               |
    | [Country211ZeroShot](./available_tasks/zeroshotclassification.md#country211zeroshot)                                                          | ZeroShotClassification       | image, text  | eng                               |
    | [DTDZeroShot](./available_tasks/zeroshotclassification.md#dtdzeroshot)                                                                        | ZeroShotClassification       | image, text  | eng                               |
    | [EuroSATZeroShot](./available_tasks/zeroshotclassification.md#eurosatzeroshot)                                                                | ZeroShotClassification       | image, text  | eng                               |
    | [FER2013ZeroShot](./available_tasks/zeroshotclassification.md#fer2013zeroshot)                                                                | ZeroShotClassification       | image, text  | eng                               |
    | [FGVCAircraftZeroShot](./available_tasks/zeroshotclassification.md#fgvcaircraftzeroshot)                                                      | ZeroShotClassification       | text, image  | eng                               |
    | [Food101ZeroShot](./available_tasks/zeroshotclassification.md#food101zeroshot)                                                                | ZeroShotClassification       | text, image  | eng                               |
    | [GTSRBZeroShot](./available_tasks/zeroshotclassification.md#gtsrbzeroshot)                                                                    | ZeroShotClassification       | image        | eng                               |
    | [Imagenet1kZeroShot](./available_tasks/zeroshotclassification.md#imagenet1kzeroshot)                                                          | ZeroShotClassification       | image, text  | eng                               |
    | [MNISTZeroShot](./available_tasks/zeroshotclassification.md#mnistzeroshot)                                                                    | ZeroShotClassification       | image, text  | eng                               |
    | [OxfordPetsZeroShot](./available_tasks/zeroshotclassification.md#oxfordpetszeroshot)                                                          | ZeroShotClassification       | text, image  | eng                               |
    | [PatchCamelyonZeroShot](./available_tasks/zeroshotclassification.md#patchcamelyonzeroshot)                                                    | ZeroShotClassification       | image, text  | eng                               |
    | [RenderedSST2](./available_tasks/zeroshotclassification.md#renderedsst2)                                                                      | ZeroShotClassification       | text, image  | eng                               |
    | [RESISC45ZeroShot](./available_tasks/zeroshotclassification.md#resisc45zeroshot)                                                              | ZeroShotClassification       | image, text  | eng                               |
    | [StanfordCarsZeroShot](./available_tasks/zeroshotclassification.md#stanfordcarszeroshot)                                                      | ZeroShotClassification       | image, text  | eng                               |
    | [STL10ZeroShot](./available_tasks/zeroshotclassification.md#stl10zeroshot)                                                                    | ZeroShotClassification       | image, text  | eng                               |
    | [SUN397ZeroShot](./available_tasks/zeroshotclassification.md#sun397zeroshot)                                                                  | ZeroShotClassification       | image, text  | eng                               |
    | [UCF101ZeroShot](./available_tasks/zeroshotclassification.md#ucf101zeroshot)                                                                  | ZeroShotClassification       | image, text  | eng                               |
    | [BLINKIT2IMultiChoice](./available_tasks/visioncentricqa.md#blinkit2imultichoice)                                                             | VisionCentricQA              | text, image  | eng                               |
    | [BLINKIT2TMultiChoice](./available_tasks/visioncentricqa.md#blinkit2tmultichoice)                                                             | VisionCentricQA              | text, image  | eng                               |
    | [CVBenchCount](./available_tasks/visioncentricqa.md#cvbenchcount)                                                                             | VisionCentricQA              | image, text  | eng                               |
    | [CVBenchRelation](./available_tasks/visioncentricqa.md#cvbenchrelation)                                                                       | VisionCentricQA              | text, image  | eng                               |
    | [CVBenchDepth](./available_tasks/visioncentricqa.md#cvbenchdepth)                                                                             | VisionCentricQA              | text, image  | eng                               |
    | [CVBenchDistance](./available_tasks/visioncentricqa.md#cvbenchdistance)                                                                       | VisionCentricQA              | text, image  | eng                               |
    | [AROCocoOrder](./available_tasks/compositionality.md#arococoorder)                                                                            | Compositionality             | text, image  | eng                               |
    | [AROFlickrOrder](./available_tasks/compositionality.md#aroflickrorder)                                                                        | Compositionality             | text, image  | eng                               |
    | [AROVisualAttribution](./available_tasks/compositionality.md#arovisualattribution)                                                            | Compositionality             | text, image  | eng                               |
    | [AROVisualRelation](./available_tasks/compositionality.md#arovisualrelation)                                                                  | Compositionality             | text, image  | eng                               |
    | [SugarCrepe](./available_tasks/compositionality.md#sugarcrepe)                                                                                | Compositionality             | text, image  | eng                               |
    | [Winoground](./available_tasks/compositionality.md#winoground)                                                                                | Compositionality             | text, image  | eng                               |
    | [ImageCoDe](./available_tasks/compositionality.md#imagecode)                                                                                  | Compositionality             | text, image  | eng                               |
    | [STS12VisualSTS](./available_tasks/visualsts(eng).md#sts12visualsts)                                                                          | VisualSTS(eng)               | image        | eng                               |
    | [STS13VisualSTS](./available_tasks/visualsts(eng).md#sts13visualsts)                                                                          | VisualSTS(eng)               | image        | eng                               |
    | [STS14VisualSTS](./available_tasks/visualsts(eng).md#sts14visualsts)                                                                          | VisualSTS(eng)               | image        | eng                               |
    | [STS15VisualSTS](./available_tasks/visualsts(eng).md#sts15visualsts)                                                                          | VisualSTS(eng)               | image        | eng                               |
    | [STS16VisualSTS](./available_tasks/visualsts(eng).md#sts16visualsts)                                                                          | VisualSTS(eng)               | image        | eng                               |
    | [BLINKIT2IRetrieval](./available_tasks/any2anyretrieval.md#blinkit2iretrieval)                                                                | Any2AnyRetrieval             | text, image  | eng                               |
    | [BLINKIT2TRetrieval](./available_tasks/any2anyretrieval.md#blinkit2tretrieval)                                                                | Any2AnyRetrieval             | text, image  | eng                               |
    | [CIRRIT2IRetrieval](./available_tasks/any2anyretrieval.md#cirrit2iretrieval)                                                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [CUB200I2IRetrieval](./available_tasks/any2anyretrieval.md#cub200i2iretrieval)                                                                | Any2AnyRetrieval             | image        | eng                               |
    | [EDIST2ITRetrieval](./available_tasks/any2anyretrieval.md#edist2itretrieval)                                                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [Fashion200kI2TRetrieval](./available_tasks/any2anyretrieval.md#fashion200ki2tretrieval)                                                      | Any2AnyRetrieval             | text, image  | eng                               |
    | [Fashion200kT2IRetrieval](./available_tasks/any2anyretrieval.md#fashion200kt2iretrieval)                                                      | Any2AnyRetrieval             | text, image  | eng                               |
    | [FashionIQIT2IRetrieval](./available_tasks/any2anyretrieval.md#fashioniqit2iretrieval)                                                        | Any2AnyRetrieval             | text, image  | eng                               |
    | [Flickr30kI2TRetrieval](./available_tasks/any2anyretrieval.md#flickr30ki2tretrieval)                                                          | Any2AnyRetrieval             | text, image  | eng                               |
    | [Flickr30kT2IRetrieval](./available_tasks/any2anyretrieval.md#flickr30kt2iretrieval)                                                          | Any2AnyRetrieval             | text, image  | eng                               |
    | [FORBI2IRetrieval](./available_tasks/any2anyretrieval.md#forbi2iretrieval)                                                                    | Any2AnyRetrieval             | image        | eng                               |
    | [GLDv2I2IRetrieval](./available_tasks/any2anyretrieval.md#gldv2i2iretrieval)                                                                  | Any2AnyRetrieval             | image        | eng                               |
    | [GLDv2I2TRetrieval](./available_tasks/any2anyretrieval.md#gldv2i2tretrieval)                                                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [HatefulMemesI2TRetrieval](./available_tasks/any2anyretrieval.md#hatefulmemesi2tretrieval)                                                    | Any2AnyRetrieval             | text, image  | eng                               |
    | [HatefulMemesT2IRetrieval](./available_tasks/any2anyretrieval.md#hatefulmemest2iretrieval)                                                    | Any2AnyRetrieval             | text, image  | eng                               |
    | [ImageCoDeT2IRetrieval](./available_tasks/any2anyretrieval.md#imagecodet2iretrieval)                                                          | Any2AnyRetrieval             | text, image  | eng                               |
    | [InfoSeekIT2ITRetrieval](./available_tasks/any2anyretrieval.md#infoseekit2itretrieval)                                                        | Any2AnyRetrieval             | text, image  | eng                               |
    | [InfoSeekIT2TRetrieval](./available_tasks/any2anyretrieval.md#infoseekit2tretrieval)                                                          | Any2AnyRetrieval             | text, image  | eng                               |
    | [MemotionI2TRetrieval](./available_tasks/any2anyretrieval.md#memotioni2tretrieval)                                                            | Any2AnyRetrieval             | text, image  | eng                               |
    | [MemotionT2IRetrieval](./available_tasks/any2anyretrieval.md#memotiont2iretrieval)                                                            | Any2AnyRetrieval             | text, image  | eng                               |
    | [METI2IRetrieval](./available_tasks/any2anyretrieval.md#meti2iretrieval)                                                                      | Any2AnyRetrieval             | image        | eng                               |
    | [MSCOCOI2TRetrieval](./available_tasks/any2anyretrieval.md#mscocoi2tretrieval)                                                                | Any2AnyRetrieval             | text, image  | eng                               |
    | [MSCOCOT2IRetrieval](./available_tasks/any2anyretrieval.md#mscocot2iretrieval)                                                                | Any2AnyRetrieval             | text, image  | eng                               |
    | [NIGHTSI2IRetrieval](./available_tasks/any2anyretrieval.md#nightsi2iretrieval)                                                                | Any2AnyRetrieval             | image        | eng                               |
    | [OVENIT2ITRetrieval](./available_tasks/any2anyretrieval.md#ovenit2itretrieval)                                                                | Any2AnyRetrieval             | image, text  | eng                               |
    | [OVENIT2TRetrieval](./available_tasks/any2anyretrieval.md#ovenit2tretrieval)                                                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [ROxfordEasyI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordeasyi2iretrieval)                                                      | Any2AnyRetrieval             | image        | eng                               |
    | [ROxfordMediumI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordmediumi2iretrieval)                                                  | Any2AnyRetrieval             | image        | eng                               |
    | [ROxfordHardI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordhardi2iretrieval)                                                      | Any2AnyRetrieval             | image        | eng                               |
    | [RP2kI2IRetrieval](./available_tasks/any2anyretrieval.md#rp2ki2iretrieval)                                                                    | Any2AnyRetrieval             | image        | eng                               |
    | [RParisEasyI2IRetrieval](./available_tasks/any2anyretrieval.md#rpariseasyi2iretrieval)                                                        | Any2AnyRetrieval             | image        | eng                               |
    | [RParisMediumI2IRetrieval](./available_tasks/any2anyretrieval.md#rparismediumi2iretrieval)                                                    | Any2AnyRetrieval             | image        | eng                               |
    | [RParisHardI2IRetrieval](./available_tasks/any2anyretrieval.md#rparishardi2iretrieval)                                                        | Any2AnyRetrieval             | image        | eng                               |
    | [SciMMIRI2TRetrieval](./available_tasks/any2anyretrieval.md#scimmiri2tretrieval)                                                              | Any2AnyRetrieval             | text, image  | eng                               |
    | [SciMMIRT2IRetrieval](./available_tasks/any2anyretrieval.md#scimmirt2iretrieval)                                                              | Any2AnyRetrieval             | text, image  | eng                               |
    | [SketchyI2IRetrieval](./available_tasks/any2anyretrieval.md#sketchyi2iretrieval)                                                              | Any2AnyRetrieval             | image        | eng                               |
    | [SOPI2IRetrieval](./available_tasks/any2anyretrieval.md#sopi2iretrieval)                                                                      | Any2AnyRetrieval             | image        | eng                               |
    | [StanfordCarsI2IRetrieval](./available_tasks/any2anyretrieval.md#stanfordcarsi2iretrieval)                                                    | Any2AnyRetrieval             | image        | eng                               |
    | [TUBerlinT2IRetrieval](./available_tasks/any2anyretrieval.md#tuberlint2iretrieval)                                                            | Any2AnyRetrieval             | text, image  | eng                               |
    | [VidoreArxivQARetrieval](./available_tasks/documentunderstanding.md#vidorearxivqaretrieval)                                                   | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreDocVQARetrieval](./available_tasks/documentunderstanding.md#vidoredocvqaretrieval)                                                     | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreInfoVQARetrieval](./available_tasks/documentunderstanding.md#vidoreinfovqaretrieval)                                                   | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreTabfquadRetrieval](./available_tasks/documentunderstanding.md#vidoretabfquadretrieval)                                                 | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreTatdqaRetrieval](./available_tasks/documentunderstanding.md#vidoretatdqaretrieval)                                                     | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreShiftProjectRetrieval](./available_tasks/documentunderstanding.md#vidoreshiftprojectretrieval)                                         | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreSyntheticDocQAAIRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaairetrieval)                                 | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreSyntheticDocQAEnergyRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaenergyretrieval)                         | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreSyntheticDocQAGovernmentReportsRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqagovernmentreportsretrieval)   | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreSyntheticDocQAHealthcareIndustryRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqahealthcareindustryretrieval) | DocumentUnderstanding        | text, image  | eng                               |
    | [VisualNewsI2TRetrieval](./available_tasks/any2anyretrieval.md#visualnewsi2tretrieval)                                                        | Any2AnyRetrieval             | image, text  | eng                               |
    | [VisualNewsT2IRetrieval](./available_tasks/any2anyretrieval.md#visualnewst2iretrieval)                                                        | Any2AnyRetrieval             | image, text  | eng                               |
    | [VizWizIT2TRetrieval](./available_tasks/any2anyretrieval.md#vizwizit2tretrieval)                                                              | Any2AnyRetrieval             | text, image  | eng                               |
    | [VQA2IT2TRetrieval](./available_tasks/any2anyretrieval.md#vqa2it2tretrieval)                                                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [WebQAT2ITRetrieval](./available_tasks/any2anyretrieval.md#webqat2itretrieval)                                                                | Any2AnyRetrieval             | image, text  | eng                               |
    | [WebQAT2TRetrieval](./available_tasks/any2anyretrieval.md#webqat2tretrieval)                                                                  | Any2AnyRetrieval             | text         | eng                               |
    | [WITT2IRetrieval](./available_tasks/any2anymultilingualretrieval.md#witt2iretrieval)                                                          | Any2AnyMultilingualRetrieval | text, image  | ara, bul, dan, ell, eng, ... (11) |
    | [XFlickr30kCoT2IRetrieval](./available_tasks/any2anymultilingualretrieval.md#xflickr30kcot2iretrieval)                                        | Any2AnyMultilingualRetrieval | text, image  | deu, eng, ind, jpn, rus, ... (8)  |
    | [XM3600T2IRetrieval](./available_tasks/any2anymultilingualretrieval.md#xm3600t2iretrieval)                                                    | Any2AnyMultilingualRetrieval | text, image  | ara, ben, ces, dan, deu, ... (38) |
    | [VisualSTS17Eng](./available_tasks/visualsts(eng).md#visualsts17eng)                                                                          | VisualSTS(eng)               | image        | eng                               |
    | [VisualSTS-b-Eng](./available_tasks/visualsts(eng).md#visualsts-b-eng)                                                                        | VisualSTS(eng)               | image        | eng                               |
    | [VisualSTS17Multilingual](./available_tasks/visualsts(multi).md#visualsts17multilingual)                                                      | VisualSTS(multi)             | image        | ara, deu, eng, fra, ita, ... (9)  |
    | [VisualSTS-b-Multilingual](./available_tasks/visualsts(multi).md#visualsts-b-multilingual)                                                    | VisualSTS(multi)             | image        | cmn, deu, fra, ita, nld, ... (9)  |


??? quote "Citation"


    ```bibtex

    @article{xiao2025mieb,
      author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
      doi = {10.48550/ARXIV.2504.10471},
      journal = {arXiv preprint arXiv:2504.10471},
      publisher = {arXiv},
      title = {MIEB: Massive Image Embedding Benchmark},
      url = {https://arxiv.org/abs/2504.10471},
      year = {2025},
    }

    ```




####  MIEB(eng)

MIEB(eng) is a comprehensive image embeddings benchmark, spanning 8 task types, covering 125 tasks.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document undestanding, visual STS, and CV-centric tasks.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info "Tasks"

    | name                                                                                                                                          | type                   | modalities   | languages   |
    |:----------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|:-------------|:------------|
    | [Birdsnap](./available_tasks/imageclassification.md#birdsnap)                                                                                 | ImageClassification    | image        | eng         |
    | [Caltech101](./available_tasks/imageclassification.md#caltech101)                                                                             | ImageClassification    | image        | eng         |
    | [CIFAR10](./available_tasks/imageclassification.md#cifar10)                                                                                   | ImageClassification    | image        | eng         |
    | [CIFAR100](./available_tasks/imageclassification.md#cifar100)                                                                                 | ImageClassification    | image        | eng         |
    | [Country211](./available_tasks/imageclassification.md#country211)                                                                             | ImageClassification    | image        | eng         |
    | [DTD](./available_tasks/imageclassification.md#dtd)                                                                                           | ImageClassification    | image        | eng         |
    | [EuroSAT](./available_tasks/imageclassification.md#eurosat)                                                                                   | ImageClassification    | image        | eng         |
    | [FER2013](./available_tasks/imageclassification.md#fer2013)                                                                                   | ImageClassification    | image        | eng         |
    | [FGVCAircraft](./available_tasks/imageclassification.md#fgvcaircraft)                                                                         | ImageClassification    | image        | eng         |
    | [Food101Classification](./available_tasks/imageclassification.md#food101classification)                                                       | ImageClassification    | image        | eng         |
    | [GTSRB](./available_tasks/imageclassification.md#gtsrb)                                                                                       | ImageClassification    | image        | eng         |
    | [Imagenet1k](./available_tasks/imageclassification.md#imagenet1k)                                                                             | ImageClassification    | image        | eng         |
    | [MNIST](./available_tasks/imageclassification.md#mnist)                                                                                       | ImageClassification    | image        | eng         |
    | [OxfordFlowersClassification](./available_tasks/imageclassification.md#oxfordflowersclassification)                                           | ImageClassification    | image        | eng         |
    | [OxfordPets](./available_tasks/imageclassification.md#oxfordpets)                                                                             | ImageClassification    | image        | eng         |
    | [PatchCamelyon](./available_tasks/imageclassification.md#patchcamelyon)                                                                       | ImageClassification    | image        | eng         |
    | [RESISC45](./available_tasks/imageclassification.md#resisc45)                                                                                 | ImageClassification    | image        | eng         |
    | [StanfordCars](./available_tasks/imageclassification.md#stanfordcars)                                                                         | ImageClassification    | image        | eng         |
    | [STL10](./available_tasks/imageclassification.md#stl10)                                                                                       | ImageClassification    | image        | eng         |
    | [SUN397](./available_tasks/imageclassification.md#sun397)                                                                                     | ImageClassification    | image        | eng         |
    | [UCF101](./available_tasks/imageclassification.md#ucf101)                                                                                     | ImageClassification    | image        | eng         |
    | [VOC2007](./available_tasks/imageclassification.md#voc2007)                                                                                   | ImageClassification    | image        | eng         |
    | [CIFAR10Clustering](./available_tasks/imageclustering.md#cifar10clustering)                                                                   | ImageClustering        | image        | eng         |
    | [CIFAR100Clustering](./available_tasks/imageclustering.md#cifar100clustering)                                                                 | ImageClustering        | image        | eng         |
    | [ImageNetDog15Clustering](./available_tasks/imageclustering.md#imagenetdog15clustering)                                                       | ImageClustering        | image        | eng         |
    | [ImageNet10Clustering](./available_tasks/imageclustering.md#imagenet10clustering)                                                             | ImageClustering        | image        | eng         |
    | [TinyImageNetClustering](./available_tasks/imageclustering.md#tinyimagenetclustering)                                                         | ImageClustering        | image        | eng         |
    | [BirdsnapZeroShot](./available_tasks/zeroshotclassification.md#birdsnapzeroshot)                                                              | ZeroShotClassification | image, text  | eng         |
    | [Caltech101ZeroShot](./available_tasks/zeroshotclassification.md#caltech101zeroshot)                                                          | ZeroShotClassification | text, image  | eng         |
    | [CIFAR10ZeroShot](./available_tasks/zeroshotclassification.md#cifar10zeroshot)                                                                | ZeroShotClassification | text, image  | eng         |
    | [CIFAR100ZeroShot](./available_tasks/zeroshotclassification.md#cifar100zeroshot)                                                              | ZeroShotClassification | text, image  | eng         |
    | [CLEVRZeroShot](./available_tasks/zeroshotclassification.md#clevrzeroshot)                                                                    | ZeroShotClassification | text, image  | eng         |
    | [CLEVRCountZeroShot](./available_tasks/zeroshotclassification.md#clevrcountzeroshot)                                                          | ZeroShotClassification | text, image  | eng         |
    | [Country211ZeroShot](./available_tasks/zeroshotclassification.md#country211zeroshot)                                                          | ZeroShotClassification | image, text  | eng         |
    | [DTDZeroShot](./available_tasks/zeroshotclassification.md#dtdzeroshot)                                                                        | ZeroShotClassification | image, text  | eng         |
    | [EuroSATZeroShot](./available_tasks/zeroshotclassification.md#eurosatzeroshot)                                                                | ZeroShotClassification | image, text  | eng         |
    | [FER2013ZeroShot](./available_tasks/zeroshotclassification.md#fer2013zeroshot)                                                                | ZeroShotClassification | image, text  | eng         |
    | [FGVCAircraftZeroShot](./available_tasks/zeroshotclassification.md#fgvcaircraftzeroshot)                                                      | ZeroShotClassification | text, image  | eng         |
    | [Food101ZeroShot](./available_tasks/zeroshotclassification.md#food101zeroshot)                                                                | ZeroShotClassification | text, image  | eng         |
    | [GTSRBZeroShot](./available_tasks/zeroshotclassification.md#gtsrbzeroshot)                                                                    | ZeroShotClassification | image        | eng         |
    | [Imagenet1kZeroShot](./available_tasks/zeroshotclassification.md#imagenet1kzeroshot)                                                          | ZeroShotClassification | image, text  | eng         |
    | [MNISTZeroShot](./available_tasks/zeroshotclassification.md#mnistzeroshot)                                                                    | ZeroShotClassification | image, text  | eng         |
    | [OxfordPetsZeroShot](./available_tasks/zeroshotclassification.md#oxfordpetszeroshot)                                                          | ZeroShotClassification | text, image  | eng         |
    | [PatchCamelyonZeroShot](./available_tasks/zeroshotclassification.md#patchcamelyonzeroshot)                                                    | ZeroShotClassification | image, text  | eng         |
    | [RenderedSST2](./available_tasks/zeroshotclassification.md#renderedsst2)                                                                      | ZeroShotClassification | text, image  | eng         |
    | [RESISC45ZeroShot](./available_tasks/zeroshotclassification.md#resisc45zeroshot)                                                              | ZeroShotClassification | image, text  | eng         |
    | [StanfordCarsZeroShot](./available_tasks/zeroshotclassification.md#stanfordcarszeroshot)                                                      | ZeroShotClassification | image, text  | eng         |
    | [STL10ZeroShot](./available_tasks/zeroshotclassification.md#stl10zeroshot)                                                                    | ZeroShotClassification | image, text  | eng         |
    | [SUN397ZeroShot](./available_tasks/zeroshotclassification.md#sun397zeroshot)                                                                  | ZeroShotClassification | image, text  | eng         |
    | [UCF101ZeroShot](./available_tasks/zeroshotclassification.md#ucf101zeroshot)                                                                  | ZeroShotClassification | image, text  | eng         |
    | [BLINKIT2IMultiChoice](./available_tasks/visioncentricqa.md#blinkit2imultichoice)                                                             | VisionCentricQA        | text, image  | eng         |
    | [BLINKIT2TMultiChoice](./available_tasks/visioncentricqa.md#blinkit2tmultichoice)                                                             | VisionCentricQA        | text, image  | eng         |
    | [CVBenchCount](./available_tasks/visioncentricqa.md#cvbenchcount)                                                                             | VisionCentricQA        | image, text  | eng         |
    | [CVBenchRelation](./available_tasks/visioncentricqa.md#cvbenchrelation)                                                                       | VisionCentricQA        | text, image  | eng         |
    | [CVBenchDepth](./available_tasks/visioncentricqa.md#cvbenchdepth)                                                                             | VisionCentricQA        | text, image  | eng         |
    | [CVBenchDistance](./available_tasks/visioncentricqa.md#cvbenchdistance)                                                                       | VisionCentricQA        | text, image  | eng         |
    | [AROCocoOrder](./available_tasks/compositionality.md#arococoorder)                                                                            | Compositionality       | text, image  | eng         |
    | [AROFlickrOrder](./available_tasks/compositionality.md#aroflickrorder)                                                                        | Compositionality       | text, image  | eng         |
    | [AROVisualAttribution](./available_tasks/compositionality.md#arovisualattribution)                                                            | Compositionality       | text, image  | eng         |
    | [AROVisualRelation](./available_tasks/compositionality.md#arovisualrelation)                                                                  | Compositionality       | text, image  | eng         |
    | [SugarCrepe](./available_tasks/compositionality.md#sugarcrepe)                                                                                | Compositionality       | text, image  | eng         |
    | [Winoground](./available_tasks/compositionality.md#winoground)                                                                                | Compositionality       | text, image  | eng         |
    | [ImageCoDe](./available_tasks/compositionality.md#imagecode)                                                                                  | Compositionality       | text, image  | eng         |
    | [STS12VisualSTS](./available_tasks/visualsts(eng).md#sts12visualsts)                                                                          | VisualSTS(eng)         | image        | eng         |
    | [STS13VisualSTS](./available_tasks/visualsts(eng).md#sts13visualsts)                                                                          | VisualSTS(eng)         | image        | eng         |
    | [STS14VisualSTS](./available_tasks/visualsts(eng).md#sts14visualsts)                                                                          | VisualSTS(eng)         | image        | eng         |
    | [STS15VisualSTS](./available_tasks/visualsts(eng).md#sts15visualsts)                                                                          | VisualSTS(eng)         | image        | eng         |
    | [STS16VisualSTS](./available_tasks/visualsts(eng).md#sts16visualsts)                                                                          | VisualSTS(eng)         | image        | eng         |
    | [BLINKIT2IRetrieval](./available_tasks/any2anyretrieval.md#blinkit2iretrieval)                                                                | Any2AnyRetrieval       | text, image  | eng         |
    | [BLINKIT2TRetrieval](./available_tasks/any2anyretrieval.md#blinkit2tretrieval)                                                                | Any2AnyRetrieval       | text, image  | eng         |
    | [CIRRIT2IRetrieval](./available_tasks/any2anyretrieval.md#cirrit2iretrieval)                                                                  | Any2AnyRetrieval       | text, image  | eng         |
    | [CUB200I2IRetrieval](./available_tasks/any2anyretrieval.md#cub200i2iretrieval)                                                                | Any2AnyRetrieval       | image        | eng         |
    | [EDIST2ITRetrieval](./available_tasks/any2anyretrieval.md#edist2itretrieval)                                                                  | Any2AnyRetrieval       | text, image  | eng         |
    | [Fashion200kI2TRetrieval](./available_tasks/any2anyretrieval.md#fashion200ki2tretrieval)                                                      | Any2AnyRetrieval       | text, image  | eng         |
    | [Fashion200kT2IRetrieval](./available_tasks/any2anyretrieval.md#fashion200kt2iretrieval)                                                      | Any2AnyRetrieval       | text, image  | eng         |
    | [FashionIQIT2IRetrieval](./available_tasks/any2anyretrieval.md#fashioniqit2iretrieval)                                                        | Any2AnyRetrieval       | text, image  | eng         |
    | [Flickr30kI2TRetrieval](./available_tasks/any2anyretrieval.md#flickr30ki2tretrieval)                                                          | Any2AnyRetrieval       | text, image  | eng         |
    | [Flickr30kT2IRetrieval](./available_tasks/any2anyretrieval.md#flickr30kt2iretrieval)                                                          | Any2AnyRetrieval       | text, image  | eng         |
    | [FORBI2IRetrieval](./available_tasks/any2anyretrieval.md#forbi2iretrieval)                                                                    | Any2AnyRetrieval       | image        | eng         |
    | [GLDv2I2IRetrieval](./available_tasks/any2anyretrieval.md#gldv2i2iretrieval)                                                                  | Any2AnyRetrieval       | image        | eng         |
    | [GLDv2I2TRetrieval](./available_tasks/any2anyretrieval.md#gldv2i2tretrieval)                                                                  | Any2AnyRetrieval       | text, image  | eng         |
    | [HatefulMemesI2TRetrieval](./available_tasks/any2anyretrieval.md#hatefulmemesi2tretrieval)                                                    | Any2AnyRetrieval       | text, image  | eng         |
    | [HatefulMemesT2IRetrieval](./available_tasks/any2anyretrieval.md#hatefulmemest2iretrieval)                                                    | Any2AnyRetrieval       | text, image  | eng         |
    | [ImageCoDeT2IRetrieval](./available_tasks/any2anyretrieval.md#imagecodet2iretrieval)                                                          | Any2AnyRetrieval       | text, image  | eng         |
    | [InfoSeekIT2ITRetrieval](./available_tasks/any2anyretrieval.md#infoseekit2itretrieval)                                                        | Any2AnyRetrieval       | text, image  | eng         |
    | [InfoSeekIT2TRetrieval](./available_tasks/any2anyretrieval.md#infoseekit2tretrieval)                                                          | Any2AnyRetrieval       | text, image  | eng         |
    | [MemotionI2TRetrieval](./available_tasks/any2anyretrieval.md#memotioni2tretrieval)                                                            | Any2AnyRetrieval       | text, image  | eng         |
    | [MemotionT2IRetrieval](./available_tasks/any2anyretrieval.md#memotiont2iretrieval)                                                            | Any2AnyRetrieval       | text, image  | eng         |
    | [METI2IRetrieval](./available_tasks/any2anyretrieval.md#meti2iretrieval)                                                                      | Any2AnyRetrieval       | image        | eng         |
    | [MSCOCOI2TRetrieval](./available_tasks/any2anyretrieval.md#mscocoi2tretrieval)                                                                | Any2AnyRetrieval       | text, image  | eng         |
    | [MSCOCOT2IRetrieval](./available_tasks/any2anyretrieval.md#mscocot2iretrieval)                                                                | Any2AnyRetrieval       | text, image  | eng         |
    | [NIGHTSI2IRetrieval](./available_tasks/any2anyretrieval.md#nightsi2iretrieval)                                                                | Any2AnyRetrieval       | image        | eng         |
    | [OVENIT2ITRetrieval](./available_tasks/any2anyretrieval.md#ovenit2itretrieval)                                                                | Any2AnyRetrieval       | image, text  | eng         |
    | [OVENIT2TRetrieval](./available_tasks/any2anyretrieval.md#ovenit2tretrieval)                                                                  | Any2AnyRetrieval       | text, image  | eng         |
    | [ROxfordEasyI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordeasyi2iretrieval)                                                      | Any2AnyRetrieval       | image        | eng         |
    | [ROxfordMediumI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordmediumi2iretrieval)                                                  | Any2AnyRetrieval       | image        | eng         |
    | [ROxfordHardI2IRetrieval](./available_tasks/any2anyretrieval.md#roxfordhardi2iretrieval)                                                      | Any2AnyRetrieval       | image        | eng         |
    | [RP2kI2IRetrieval](./available_tasks/any2anyretrieval.md#rp2ki2iretrieval)                                                                    | Any2AnyRetrieval       | image        | eng         |
    | [RParisEasyI2IRetrieval](./available_tasks/any2anyretrieval.md#rpariseasyi2iretrieval)                                                        | Any2AnyRetrieval       | image        | eng         |
    | [RParisMediumI2IRetrieval](./available_tasks/any2anyretrieval.md#rparismediumi2iretrieval)                                                    | Any2AnyRetrieval       | image        | eng         |
    | [RParisHardI2IRetrieval](./available_tasks/any2anyretrieval.md#rparishardi2iretrieval)                                                        | Any2AnyRetrieval       | image        | eng         |
    | [SciMMIRI2TRetrieval](./available_tasks/any2anyretrieval.md#scimmiri2tretrieval)                                                              | Any2AnyRetrieval       | text, image  | eng         |
    | [SciMMIRT2IRetrieval](./available_tasks/any2anyretrieval.md#scimmirt2iretrieval)                                                              | Any2AnyRetrieval       | text, image  | eng         |
    | [SketchyI2IRetrieval](./available_tasks/any2anyretrieval.md#sketchyi2iretrieval)                                                              | Any2AnyRetrieval       | image        | eng         |
    | [SOPI2IRetrieval](./available_tasks/any2anyretrieval.md#sopi2iretrieval)                                                                      | Any2AnyRetrieval       | image        | eng         |
    | [StanfordCarsI2IRetrieval](./available_tasks/any2anyretrieval.md#stanfordcarsi2iretrieval)                                                    | Any2AnyRetrieval       | image        | eng         |
    | [TUBerlinT2IRetrieval](./available_tasks/any2anyretrieval.md#tuberlint2iretrieval)                                                            | Any2AnyRetrieval       | text, image  | eng         |
    | [VidoreArxivQARetrieval](./available_tasks/documentunderstanding.md#vidorearxivqaretrieval)                                                   | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreDocVQARetrieval](./available_tasks/documentunderstanding.md#vidoredocvqaretrieval)                                                     | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreInfoVQARetrieval](./available_tasks/documentunderstanding.md#vidoreinfovqaretrieval)                                                   | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreTabfquadRetrieval](./available_tasks/documentunderstanding.md#vidoretabfquadretrieval)                                                 | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreTatdqaRetrieval](./available_tasks/documentunderstanding.md#vidoretatdqaretrieval)                                                     | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreShiftProjectRetrieval](./available_tasks/documentunderstanding.md#vidoreshiftprojectretrieval)                                         | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreSyntheticDocQAAIRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaairetrieval)                                 | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreSyntheticDocQAEnergyRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaenergyretrieval)                         | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreSyntheticDocQAGovernmentReportsRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqagovernmentreportsretrieval)   | DocumentUnderstanding  | text, image  | eng         |
    | [VidoreSyntheticDocQAHealthcareIndustryRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqahealthcareindustryretrieval) | DocumentUnderstanding  | text, image  | eng         |
    | [VisualNewsI2TRetrieval](./available_tasks/any2anyretrieval.md#visualnewsi2tretrieval)                                                        | Any2AnyRetrieval       | image, text  | eng         |
    | [VisualNewsT2IRetrieval](./available_tasks/any2anyretrieval.md#visualnewst2iretrieval)                                                        | Any2AnyRetrieval       | image, text  | eng         |
    | [VizWizIT2TRetrieval](./available_tasks/any2anyretrieval.md#vizwizit2tretrieval)                                                              | Any2AnyRetrieval       | text, image  | eng         |
    | [VQA2IT2TRetrieval](./available_tasks/any2anyretrieval.md#vqa2it2tretrieval)                                                                  | Any2AnyRetrieval       | text, image  | eng         |
    | [WebQAT2ITRetrieval](./available_tasks/any2anyretrieval.md#webqat2itretrieval)                                                                | Any2AnyRetrieval       | image, text  | eng         |
    | [WebQAT2TRetrieval](./available_tasks/any2anyretrieval.md#webqat2tretrieval)                                                                  | Any2AnyRetrieval       | text         | eng         |
    | [VisualSTS17Eng](./available_tasks/visualsts(eng).md#visualsts17eng)                                                                          | VisualSTS(eng)         | image        | eng         |
    | [VisualSTS-b-Eng](./available_tasks/visualsts(eng).md#visualsts-b-eng)                                                                        | VisualSTS(eng)         | image        | eng         |


??? quote "Citation"


    ```bibtex

    @article{xiao2025mieb,
      author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
      doi = {10.48550/ARXIV.2504.10471},
      journal = {arXiv preprint arXiv:2504.10471},
      publisher = {arXiv},
      title = {MIEB: Massive Image Embedding Benchmark},
      url = {https://arxiv.org/abs/2504.10471},
      year = {2025},
    }

    ```




####  MIEB(lite)

MIEB(lite) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 51 tasks.
    This is a lite version of MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining
    relative rank of models.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info "Tasks"

    | name                                                                                                          | type                         | modalities   | languages                         |
    |:--------------------------------------------------------------------------------------------------------------|:-----------------------------|:-------------|:----------------------------------|
    | [Country211](./available_tasks/imageclassification.md#country211)                                             | ImageClassification          | image        | eng                               |
    | [DTD](./available_tasks/imageclassification.md#dtd)                                                           | ImageClassification          | image        | eng                               |
    | [EuroSAT](./available_tasks/imageclassification.md#eurosat)                                                   | ImageClassification          | image        | eng                               |
    | [GTSRB](./available_tasks/imageclassification.md#gtsrb)                                                       | ImageClassification          | image        | eng                               |
    | [OxfordPets](./available_tasks/imageclassification.md#oxfordpets)                                             | ImageClassification          | image        | eng                               |
    | [PatchCamelyon](./available_tasks/imageclassification.md#patchcamelyon)                                       | ImageClassification          | image        | eng                               |
    | [RESISC45](./available_tasks/imageclassification.md#resisc45)                                                 | ImageClassification          | image        | eng                               |
    | [SUN397](./available_tasks/imageclassification.md#sun397)                                                     | ImageClassification          | image        | eng                               |
    | [ImageNetDog15Clustering](./available_tasks/imageclustering.md#imagenetdog15clustering)                       | ImageClustering              | image        | eng                               |
    | [TinyImageNetClustering](./available_tasks/imageclustering.md#tinyimagenetclustering)                         | ImageClustering              | image        | eng                               |
    | [CIFAR100ZeroShot](./available_tasks/zeroshotclassification.md#cifar100zeroshot)                              | ZeroShotClassification       | text, image  | eng                               |
    | [Country211ZeroShot](./available_tasks/zeroshotclassification.md#country211zeroshot)                          | ZeroShotClassification       | image, text  | eng                               |
    | [FER2013ZeroShot](./available_tasks/zeroshotclassification.md#fer2013zeroshot)                                | ZeroShotClassification       | image, text  | eng                               |
    | [FGVCAircraftZeroShot](./available_tasks/zeroshotclassification.md#fgvcaircraftzeroshot)                      | ZeroShotClassification       | text, image  | eng                               |
    | [Food101ZeroShot](./available_tasks/zeroshotclassification.md#food101zeroshot)                                | ZeroShotClassification       | text, image  | eng                               |
    | [OxfordPetsZeroShot](./available_tasks/zeroshotclassification.md#oxfordpetszeroshot)                          | ZeroShotClassification       | text, image  | eng                               |
    | [StanfordCarsZeroShot](./available_tasks/zeroshotclassification.md#stanfordcarszeroshot)                      | ZeroShotClassification       | image, text  | eng                               |
    | [BLINKIT2IMultiChoice](./available_tasks/visioncentricqa.md#blinkit2imultichoice)                             | VisionCentricQA              | text, image  | eng                               |
    | [CVBenchCount](./available_tasks/visioncentricqa.md#cvbenchcount)                                             | VisionCentricQA              | image, text  | eng                               |
    | [CVBenchRelation](./available_tasks/visioncentricqa.md#cvbenchrelation)                                       | VisionCentricQA              | text, image  | eng                               |
    | [CVBenchDepth](./available_tasks/visioncentricqa.md#cvbenchdepth)                                             | VisionCentricQA              | text, image  | eng                               |
    | [CVBenchDistance](./available_tasks/visioncentricqa.md#cvbenchdistance)                                       | VisionCentricQA              | text, image  | eng                               |
    | [AROCocoOrder](./available_tasks/compositionality.md#arococoorder)                                            | Compositionality             | text, image  | eng                               |
    | [AROFlickrOrder](./available_tasks/compositionality.md#aroflickrorder)                                        | Compositionality             | text, image  | eng                               |
    | [AROVisualAttribution](./available_tasks/compositionality.md#arovisualattribution)                            | Compositionality             | text, image  | eng                               |
    | [AROVisualRelation](./available_tasks/compositionality.md#arovisualrelation)                                  | Compositionality             | text, image  | eng                               |
    | [Winoground](./available_tasks/compositionality.md#winoground)                                                | Compositionality             | text, image  | eng                               |
    | [ImageCoDe](./available_tasks/compositionality.md#imagecode)                                                  | Compositionality             | text, image  | eng                               |
    | [STS13VisualSTS](./available_tasks/visualsts(eng).md#sts13visualsts)                                          | VisualSTS(eng)               | image        | eng                               |
    | [STS15VisualSTS](./available_tasks/visualsts(eng).md#sts15visualsts)                                          | VisualSTS(eng)               | image        | eng                               |
    | [VisualSTS17Multilingual](./available_tasks/visualsts(multi).md#visualsts17multilingual)                      | VisualSTS(multi)             | image        | ara, deu, eng, fra, ita, ... (9)  |
    | [VisualSTS-b-Multilingual](./available_tasks/visualsts(multi).md#visualsts-b-multilingual)                    | VisualSTS(multi)             | image        | cmn, deu, fra, ita, nld, ... (9)  |
    | [CIRRIT2IRetrieval](./available_tasks/any2anyretrieval.md#cirrit2iretrieval)                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [CUB200I2IRetrieval](./available_tasks/any2anyretrieval.md#cub200i2iretrieval)                                | Any2AnyRetrieval             | image        | eng                               |
    | [Fashion200kI2TRetrieval](./available_tasks/any2anyretrieval.md#fashion200ki2tretrieval)                      | Any2AnyRetrieval             | text, image  | eng                               |
    | [HatefulMemesI2TRetrieval](./available_tasks/any2anyretrieval.md#hatefulmemesi2tretrieval)                    | Any2AnyRetrieval             | text, image  | eng                               |
    | [InfoSeekIT2TRetrieval](./available_tasks/any2anyretrieval.md#infoseekit2tretrieval)                          | Any2AnyRetrieval             | text, image  | eng                               |
    | [NIGHTSI2IRetrieval](./available_tasks/any2anyretrieval.md#nightsi2iretrieval)                                | Any2AnyRetrieval             | image        | eng                               |
    | [OVENIT2TRetrieval](./available_tasks/any2anyretrieval.md#ovenit2tretrieval)                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [RP2kI2IRetrieval](./available_tasks/any2anyretrieval.md#rp2ki2iretrieval)                                    | Any2AnyRetrieval             | image        | eng                               |
    | [VidoreDocVQARetrieval](./available_tasks/documentunderstanding.md#vidoredocvqaretrieval)                     | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreInfoVQARetrieval](./available_tasks/documentunderstanding.md#vidoreinfovqaretrieval)                   | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreTabfquadRetrieval](./available_tasks/documentunderstanding.md#vidoretabfquadretrieval)                 | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreTatdqaRetrieval](./available_tasks/documentunderstanding.md#vidoretatdqaretrieval)                     | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreShiftProjectRetrieval](./available_tasks/documentunderstanding.md#vidoreshiftprojectretrieval)         | DocumentUnderstanding        | text, image  | eng                               |
    | [VidoreSyntheticDocQAAIRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaairetrieval) | DocumentUnderstanding        | text, image  | eng                               |
    | [VisualNewsI2TRetrieval](./available_tasks/any2anyretrieval.md#visualnewsi2tretrieval)                        | Any2AnyRetrieval             | image, text  | eng                               |
    | [VQA2IT2TRetrieval](./available_tasks/any2anyretrieval.md#vqa2it2tretrieval)                                  | Any2AnyRetrieval             | text, image  | eng                               |
    | [WebQAT2ITRetrieval](./available_tasks/any2anyretrieval.md#webqat2itretrieval)                                | Any2AnyRetrieval             | image, text  | eng                               |
    | [WITT2IRetrieval](./available_tasks/any2anymultilingualretrieval.md#witt2iretrieval)                          | Any2AnyMultilingualRetrieval | text, image  | ara, bul, dan, ell, eng, ... (11) |
    | [XM3600T2IRetrieval](./available_tasks/any2anymultilingualretrieval.md#xm3600t2iretrieval)                    | Any2AnyMultilingualRetrieval | text, image  | ara, ben, ces, dan, deu, ... (38) |


??? quote "Citation"


    ```bibtex

    @article{xiao2025mieb,
      author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
      doi = {10.48550/ARXIV.2504.10471},
      journal = {arXiv preprint arXiv:2504.10471},
      publisher = {arXiv},
      title = {MIEB: Massive Image Embedding Benchmark},
      url = {https://arxiv.org/abs/2504.10471},
      year = {2025},
    }

    ```




####  MINERSBitextMining

Bitext Mining texts from the MINERS benchmark, a benchmark designed to evaluate the
    ability of multilingual LMs in semantic retrieval tasks,
    including bitext mining and classification via retrieval-augmented contexts.


[Learn more →](https://arxiv.org/pdf/2406.07424)



??? info "Tasks"

    | name                                                                                         | type         | modalities   | languages                          |
    |:---------------------------------------------------------------------------------------------|:-------------|:-------------|:-----------------------------------|
    | [BUCC](./available_tasks/bitextmining.md#bucc)                                               | BitextMining | text         | cmn, deu, eng, fra, rus            |
    | [LinceMTBitextMining](./available_tasks/bitextmining.md#lincemtbitextmining)                 | BitextMining | text         | eng, hin                           |
    | [NollySentiBitextMining](./available_tasks/bitextmining.md#nollysentibitextmining)           | BitextMining | text         | eng, hau, ibo, pcm, yor            |
    | [NusaXBitextMining](./available_tasks/bitextmining.md#nusaxbitextmining)                     | BitextMining | text         | ace, ban, bbc, bjn, bug, ... (12)  |
    | [NusaTranslationBitextMining](./available_tasks/bitextmining.md#nusatranslationbitextmining) | BitextMining | text         | abs, bbc, bew, bhp, ind, ... (12)  |
    | [PhincBitextMining](./available_tasks/bitextmining.md#phincbitextmining)                     | BitextMining | text         | eng, hin                           |
    | [Tatoeba](./available_tasks/bitextmining.md#tatoeba)                                         | BitextMining | text         | afr, amh, ang, ara, arq, ... (113) |


??? quote "Citation"


    ```bibtex

    @article{winata2024miners,
      author = {Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
      journal = {arXiv preprint arXiv:2406.07424},
      title = {MINERS: Multilingual Language Models as Semantic Retrievers},
      year = {2024},
    }

    ```




####  MTEB(Code, v1)

A massive code embedding benchmark covering retrieval tasks in a miriad of popular programming languages.

??? info "Tasks"

    | name                                                                                    | type      | modalities   | languages                                  |
    |:----------------------------------------------------------------------------------------|:----------|:-------------|:-------------------------------------------|
    | [AppsRetrieval](./available_tasks/retrieval.md#appsretrieval)                           | Retrieval | text         | eng, python                                |
    | [CodeEditSearchRetrieval](./available_tasks/retrieval.md#codeeditsearchretrieval)       | Retrieval | text         | c, c++, go, java, javascript, ... (13)     |
    | [CodeFeedbackMT](./available_tasks/retrieval.md#codefeedbackmt)                         | Retrieval | text         | eng                                        |
    | [CodeFeedbackST](./available_tasks/retrieval.md#codefeedbackst)                         | Retrieval | text         | eng                                        |
    | [CodeSearchNetCCRetrieval](./available_tasks/retrieval.md#codesearchnetccretrieval)     | Retrieval | text         | go, java, javascript, php, python, ... (6) |
    | [CodeSearchNetRetrieval](./available_tasks/retrieval.md#codesearchnetretrieval)         | Retrieval | text         | go, java, javascript, php, python, ... (6) |
    | [CodeTransOceanContest](./available_tasks/retrieval.md#codetransoceancontest)           | Retrieval | text         | c++, python                                |
    | [CodeTransOceanDL](./available_tasks/retrieval.md#codetransoceandl)                     | Retrieval | text         | python                                     |
    | [CosQA](./available_tasks/retrieval.md#cosqa)                                           | Retrieval | text         | eng, python                                |
    | [COIRCodeSearchNetRetrieval](./available_tasks/retrieval.md#coircodesearchnetretrieval) | Retrieval | text         | go, java, javascript, php, python, ... (6) |
    | [StackOverflowQA](./available_tasks/retrieval.md#stackoverflowqa)                       | Retrieval | text         | eng                                        |
    | [SyntheticText2SQL](./available_tasks/retrieval.md#synthetictext2sql)                   | Retrieval | text         | eng, sql                                   |


??? quote "Citation"


    ```bibtex
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      doi = {10.48550/arXiv.2502.13595},
      journal = {arXiv preprint arXiv:2502.13595},
      publisher = {arXiv},
      title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2502.13595},
      year = {2025},
    }
    ```




####  MTEB(Europe, v1)

A regional geopolitical text embedding benchmark targetting embedding performance on European languages.

??? info "Tasks"

    | name                                                                                                                         | type                     | modalities   | languages                          |
    |:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------|:-------------|:-----------------------------------|
    | [BornholmBitextMining](./available_tasks/bitextmining.md#bornholmbitextmining)                                               | BitextMining             | text         | dan                                |
    | [BibleNLPBitextMining](./available_tasks/bitextmining.md#biblenlpbitextmining)                                               | BitextMining             | text         | aai, aak, aau, aaz, abt, ... (829) |
    | [BUCC.v2](./available_tasks/bitextmining.md#bucc.v2)                                                                         | BitextMining             | text         | cmn, deu, eng, fra, rus            |
    | [DiaBlaBitextMining](./available_tasks/bitextmining.md#diablabitextmining)                                                   | BitextMining             | text         | eng, fra                           |
    | [FloresBitextMining](./available_tasks/bitextmining.md#floresbitextmining)                                                   | BitextMining             | text         | ace, acm, acq, aeb, afr, ... (196) |
    | [NorwegianCourtsBitextMining](./available_tasks/bitextmining.md#norwegiancourtsbitextmining)                                 | BitextMining             | text         | nno, nob                           |
    | [NTREXBitextMining](./available_tasks/bitextmining.md#ntrexbitextmining)                                                     | BitextMining             | text         | afr, amh, arb, aze, bak, ... (119) |
    | [BulgarianStoreReviewSentimentClassfication](./available_tasks/classification.md#bulgarianstorereviewsentimentclassfication) | Classification           | text         | bul                                |
    | [CzechProductReviewSentimentClassification](./available_tasks/classification.md#czechproductreviewsentimentclassification)   | Classification           | text         | ces                                |
    | [GreekLegalCodeClassification](./available_tasks/classification.md#greeklegalcodeclassification)                             | Classification           | text         | ell                                |
    | [DBpediaClassification](./available_tasks/classification.md#dbpediaclassification)                                           | Classification           | text         | eng                                |
    | [FinancialPhrasebankClassification](./available_tasks/classification.md#financialphrasebankclassification)                   | Classification           | text         | eng                                |
    | [PoemSentimentClassification](./available_tasks/classification.md#poemsentimentclassification)                               | Classification           | text         | eng                                |
    | [ToxicChatClassification](./available_tasks/classification.md#toxicchatclassification)                                       | Classification           | text         | eng                                |
    | [ToxicConversationsClassification](./available_tasks/classification.md#toxicconversationsclassification)                     | Classification           | text         | eng                                |
    | [EstonianValenceClassification](./available_tasks/classification.md#estonianvalenceclassification)                           | Classification           | text         | est                                |
    | [ItaCaseholdClassification](./available_tasks/classification.md#itacaseholdclassification)                                   | Classification           | text         | ita                                |
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification)                 | Classification           | text         | deu, eng, jpn                      |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)                           | Classification           | text         | afr, amh, ara, aze, ben, ... (50)  |
    | [MultiHateClassification](./available_tasks/classification.md#multihateclassification)                                       | Classification           | text         | ara, cmn, deu, eng, fra, ... (11)  |
    | [NordicLangClassification](./available_tasks/classification.md#nordiclangclassification)                                     | Classification           | text         | dan, fao, isl, nno, nob, ... (6)   |
    | [ScalaClassification](./available_tasks/classification.md#scalaclassification)                                               | Classification           | text         | dan, nno, nob, swe                 |
    | [SwissJudgementClassification](./available_tasks/classification.md#swissjudgementclassification)                             | Classification           | text         | deu, fra, ita                      |
    | [TweetSentimentClassification](./available_tasks/classification.md#tweetsentimentclassification)                             | Classification           | text         | ara, deu, eng, fra, hin, ... (8)   |
    | [CBD](./available_tasks/classification.md#cbd)                                                                               | Classification           | text         | pol                                |
    | [PolEmo2.0-OUT](./available_tasks/classification.md#polemo2.0-out)                                                           | Classification           | text         | pol                                |
    | [CSFDSKMovieReviewSentimentClassification](./available_tasks/classification.md#csfdskmoviereviewsentimentclassification)     | Classification           | text         | slk                                |
    | [DalajClassification](./available_tasks/classification.md#dalajclassification)                                               | Classification           | text         | swe                                |
    | [WikiCitiesClustering](./available_tasks/clustering.md#wikicitiesclustering)                                                 | Clustering               | text         | eng                                |
    | [RomaniBibleClustering](./available_tasks/clustering.md#romanibibleclustering)                                               | Clustering               | text         | rom                                |
    | [BigPatentClustering.v2](./available_tasks/clustering.md#bigpatentclustering.v2)                                             | Clustering               | text         | eng                                |
    | [BiorxivClusteringP2P.v2](./available_tasks/clustering.md#biorxivclusteringp2p.v2)                                           | Clustering               | text         | eng                                |
    | [AlloProfClusteringS2S.v2](./available_tasks/clustering.md#alloprofclusterings2s.v2)                                         | Clustering               | text         | fra                                |
    | [HALClusteringS2S.v2](./available_tasks/clustering.md#halclusterings2s.v2)                                                   | Clustering               | text         | fra                                |
    | [SIB200ClusteringS2S](./available_tasks/clustering.md#sib200clusterings2s)                                                   | Clustering               | text         | ace, acm, acq, aeb, afr, ... (197) |
    | [WikiClusteringP2P.v2](./available_tasks/clustering.md#wikiclusteringp2p.v2)                                                 | Clustering               | text         | bos, cat, ces, dan, eus, ... (14)  |
    | [StackOverflowQA](./available_tasks/retrieval.md#stackoverflowqa)                                                            | Retrieval                | text         | eng                                |
    | [TwitterHjerneRetrieval](./available_tasks/retrieval.md#twitterhjerneretrieval)                                              | Retrieval                | text         | dan                                |
    | [LegalQuAD](./available_tasks/retrieval.md#legalquad)                                                                        | Retrieval                | text         | deu                                |
    | [ArguAna](./available_tasks/retrieval.md#arguana)                                                                            | Retrieval                | text         | eng                                |
    | [HagridRetrieval](./available_tasks/retrieval.md#hagridretrieval)                                                            | Retrieval                | text         | eng                                |
    | [LegalBenchCorporateLobbying](./available_tasks/retrieval.md#legalbenchcorporatelobbying)                                    | Retrieval                | text         | eng                                |
    | [LEMBPasskeyRetrieval](./available_tasks/retrieval.md#lembpasskeyretrieval)                                                  | Retrieval                | text         | eng                                |
    | [SCIDOCS](./available_tasks/retrieval.md#scidocs)                                                                            | Retrieval                | text         | eng                                |
    | [SpartQA](./available_tasks/retrieval.md#spartqa)                                                                            | Retrieval                | text         | eng                                |
    | [TempReasonL1](./available_tasks/retrieval.md#tempreasonl1)                                                                  | Retrieval                | text         | eng                                |
    | [WinoGrande](./available_tasks/retrieval.md#winogrande)                                                                      | Retrieval                | text         | eng                                |
    | [AlloprofRetrieval](./available_tasks/retrieval.md#alloprofretrieval)                                                        | Retrieval                | text         | fra                                |
    | [BelebeleRetrieval](./available_tasks/retrieval.md#belebeleretrieval)                                                        | Retrieval                | text         | acm, afr, als, amh, apc, ... (115) |
    | [StatcanDialogueDatasetRetrieval](./available_tasks/retrieval.md#statcandialoguedatasetretrieval)                            | Retrieval                | text         | eng, fra                           |
    | [WikipediaRetrievalMultilingual](./available_tasks/retrieval.md#wikipediaretrievalmultilingual)                              | Retrieval                | text         | ben, bul, ces, dan, deu, ... (16)  |
    | [Core17InstructionRetrieval](./available_tasks/instructionreranking.md#core17instructionretrieval)                           | InstructionReranking     | text         | eng                                |
    | [News21InstructionRetrieval](./available_tasks/instructionreranking.md#news21instructionretrieval)                           | InstructionReranking     | text         | eng                                |
    | [Robust04InstructionRetrieval](./available_tasks/instructionreranking.md#robust04instructionretrieval)                       | InstructionReranking     | text         | eng                                |
    | [MalteseNewsClassification](./available_tasks/multilabelclassification.md#maltesenewsclassification)                         | MultilabelClassification | text         | mlt                                |
    | [MultiEURLEXMultilabelClassification](./available_tasks/multilabelclassification.md#multieurlexmultilabelclassification)     | MultilabelClassification | text         | bul, ces, dan, deu, ell, ... (23)  |
    | [CTKFactsNLI](./available_tasks/pairclassification.md#ctkfactsnli)                                                           | PairClassification       | text         | ces                                |
    | [SprintDuplicateQuestions](./available_tasks/pairclassification.md#sprintduplicatequestions)                                 | PairClassification       | text         | eng                                |
    | [OpusparcusPC](./available_tasks/pairclassification.md#opusparcuspc)                                                         | PairClassification       | text         | deu, eng, fin, fra, rus, ... (6)   |
    | [RTE3](./available_tasks/pairclassification.md#rte3)                                                                         | PairClassification       | text         | deu, eng, fra, ita                 |
    | [XNLI](./available_tasks/pairclassification.md#xnli)                                                                         | PairClassification       | text         | ara, bul, deu, ell, eng, ... (14)  |
    | [PSC](./available_tasks/pairclassification.md#psc)                                                                           | PairClassification       | text         | pol                                |
    | [WebLINXCandidatesReranking](./available_tasks/reranking.md#weblinxcandidatesreranking)                                      | Reranking                | text         | eng                                |
    | [AlloprofReranking](./available_tasks/reranking.md#alloprofreranking)                                                        | Reranking                | text         | fra                                |
    | [WikipediaRerankingMultilingual](./available_tasks/reranking.md#wikipediarerankingmultilingual)                              | Reranking                | text         | ben, bul, ces, dan, deu, ... (18)  |
    | [SICK-R](./available_tasks/sts.md#sick-r)                                                                                    | STS                      | text         | eng                                |
    | [STS12](./available_tasks/sts.md#sts12)                                                                                      | STS                      | text         | eng                                |
    | [STS14](./available_tasks/sts.md#sts14)                                                                                      | STS                      | text         | eng                                |
    | [STS15](./available_tasks/sts.md#sts15)                                                                                      | STS                      | text         | eng                                |
    | [STSBenchmark](./available_tasks/sts.md#stsbenchmark)                                                                        | STS                      | text         | eng                                |
    | [FinParaSTS](./available_tasks/sts.md#finparasts)                                                                            | STS                      | text         | fin                                |
    | [STS17](./available_tasks/sts.md#sts17)                                                                                      | STS                      | text         | ara, deu, eng, fra, ita, ... (9)   |
    | [SICK-R-PL](./available_tasks/sts.md#sick-r-pl)                                                                              | STS                      | text         | pol                                |
    | [STSES](./available_tasks/sts.md#stses)                                                                                      | STS                      | text         | spa                                |


??? quote "Citation"


    ```bibtex
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      doi = {10.48550/arXiv.2502.13595},
      journal = {arXiv preprint arXiv:2502.13595},
      publisher = {arXiv},
      title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2502.13595},
      year = {2025},
    }
    ```




####  MTEB(Indic, v1)

A regional geopolitical text embedding benchmark targetting embedding performance on Indic languages.

??? info "Tasks"

    | name                                                                                                     | type               | modalities   | languages                          |
    |:---------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:-----------------------------------|
    | [IN22ConvBitextMining](./available_tasks/bitextmining.md#in22convbitextmining)                           | BitextMining       | text         | asm, ben, brx, doi, eng, ... (23)  |
    | [IN22GenBitextMining](./available_tasks/bitextmining.md#in22genbitextmining)                             | BitextMining       | text         | asm, ben, brx, doi, eng, ... (23)  |
    | [IndicGenBenchFloresBitextMining](./available_tasks/bitextmining.md#indicgenbenchfloresbitextmining)     | BitextMining       | text         | asm, awa, ben, bgc, bho, ... (30)  |
    | [LinceMTBitextMining](./available_tasks/bitextmining.md#lincemtbitextmining)                             | BitextMining       | text         | eng, hin                           |
    | [SIB200ClusteringS2S](./available_tasks/clustering.md#sib200clusterings2s)                               | Clustering         | text         | ace, acm, acq, aeb, afr, ... (197) |
    | [BengaliSentimentAnalysis](./available_tasks/classification.md#bengalisentimentanalysis)                 | Classification     | text         | ben                                |
    | [GujaratiNewsClassification](./available_tasks/classification.md#gujaratinewsclassification)             | Classification     | text         | guj                                |
    | [HindiDiscourseClassification](./available_tasks/classification.md#hindidiscourseclassification)         | Classification     | text         | hin                                |
    | [SentimentAnalysisHindi](./available_tasks/classification.md#sentimentanalysishindi)                     | Classification     | text         | hin                                |
    | [MalayalamNewsClassification](./available_tasks/classification.md#malayalamnewsclassification)           | Classification     | text         | mal                                |
    | [IndicLangClassification](./available_tasks/classification.md#indiclangclassification)                   | Classification     | text         | asm, ben, brx, doi, gom, ... (22)  |
    | [MTOPIntentClassification](./available_tasks/classification.md#mtopintentclassification)                 | Classification     | text         | deu, eng, fra, hin, spa, ... (6)   |
    | [MultiHateClassification](./available_tasks/classification.md#multihateclassification)                   | Classification     | text         | ara, cmn, deu, eng, fra, ... (11)  |
    | [TweetSentimentClassification](./available_tasks/classification.md#tweetsentimentclassification)         | Classification     | text         | ara, deu, eng, fra, hin, ... (8)   |
    | [NepaliNewsClassification](./available_tasks/classification.md#nepalinewsclassification)                 | Classification     | text         | nep                                |
    | [PunjabiNewsClassification](./available_tasks/classification.md#punjabinewsclassification)               | Classification     | text         | pan                                |
    | [SanskritShlokasClassification](./available_tasks/classification.md#sanskritshlokasclassification)       | Classification     | text         | san                                |
    | [UrduRomanSentimentClassification](./available_tasks/classification.md#urduromansentimentclassification) | Classification     | text         | urd                                |
    | [XNLI](./available_tasks/pairclassification.md#xnli)                                                     | PairClassification | text         | ara, bul, deu, ell, eng, ... (14)  |
    | [BelebeleRetrieval](./available_tasks/retrieval.md#belebeleretrieval)                                    | Retrieval          | text         | acm, afr, als, amh, apc, ... (115) |
    | [XQuADRetrieval](./available_tasks/retrieval.md#xquadretrieval)                                          | Retrieval          | text         | arb, deu, ell, eng, hin, ... (12)  |
    | [WikipediaRerankingMultilingual](./available_tasks/reranking.md#wikipediarerankingmultilingual)          | Reranking          | text         | ben, bul, ces, dan, deu, ... (18)  |
    | [IndicCrosslingualSTS](./available_tasks/sts.md#indiccrosslingualsts)                                    | STS                | text         | asm, ben, eng, guj, hin, ... (13)  |


??? quote "Citation"


    ```bibtex
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      doi = {10.48550/arXiv.2502.13595},
      journal = {arXiv preprint arXiv:2502.13595},
      publisher = {arXiv},
      title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2502.13595},
      year = {2025},
    }
    ```




####  MTEB(Law, v1)

A benchmark of retrieval tasks in the legal domain.

??? info "Tasks"

    | name                                                                                          | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [AILACasedocs](./available_tasks/retrieval.md#ailacasedocs)                                   | Retrieval | text         | eng         |
    | [AILAStatutes](./available_tasks/retrieval.md#ailastatutes)                                   | Retrieval | text         | eng         |
    | [LegalSummarization](./available_tasks/retrieval.md#legalsummarization)                       | Retrieval | text         | eng         |
    | [GerDaLIRSmall](./available_tasks/retrieval.md#gerdalirsmall)                                 | Retrieval | text         | deu         |
    | [LeCaRDv2](./available_tasks/retrieval.md#lecardv2)                                           | Retrieval | text         | zho         |
    | [LegalBenchConsumerContractsQA](./available_tasks/retrieval.md#legalbenchconsumercontractsqa) | Retrieval | text         | eng         |
    | [LegalBenchCorporateLobbying](./available_tasks/retrieval.md#legalbenchcorporatelobbying)     | Retrieval | text         | eng         |
    | [LegalQuAD](./available_tasks/retrieval.md#legalquad)                                         | Retrieval | text         | deu         |


####  MTEB(Medical, v1)

A curated set of MTEB tasks designed to evaluate systems in the context of medical information retrieval.

??? info "Tasks"

    | name                                                                               | type       | modalities   | languages                        |
    |:-----------------------------------------------------------------------------------|:-----------|:-------------|:---------------------------------|
    | [CUREv1](./available_tasks/retrieval.md#curev1)                                    | Retrieval  | text         | eng, fra, spa                    |
    | [NFCorpus](./available_tasks/retrieval.md#nfcorpus)                                | Retrieval  | text         | eng                              |
    | [TRECCOVID](./available_tasks/retrieval.md#treccovid)                              | Retrieval  | text         | eng                              |
    | [TRECCOVID-PL](./available_tasks/retrieval.md#treccovid-pl)                        | Retrieval  | text         | pol                              |
    | [SciFact](./available_tasks/retrieval.md#scifact)                                  | Retrieval  | text         | eng                              |
    | [SciFact-PL](./available_tasks/retrieval.md#scifact-pl)                            | Retrieval  | text         | pol                              |
    | [MedicalQARetrieval](./available_tasks/retrieval.md#medicalqaretrieval)            | Retrieval  | text         | eng                              |
    | [PublicHealthQA](./available_tasks/retrieval.md#publichealthqa)                    | Retrieval  | text         | ara, eng, fra, kor, rus, ... (8) |
    | [MedrxivClusteringP2P.v2](./available_tasks/clustering.md#medrxivclusteringp2p.v2) | Clustering | text         | eng                              |
    | [MedrxivClusteringS2S.v2](./available_tasks/clustering.md#medrxivclusterings2s.v2) | Clustering | text         | eng                              |
    | [CmedqaRetrieval](./available_tasks/retrieval.md#cmedqaretrieval)                  | Retrieval  | text         | cmn                              |
    | [CMedQAv2-reranking](./available_tasks/reranking.md#cmedqav2-reranking)            | Reranking  | text         | cmn                              |


####  MTEB(Multilingual, v1)

A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages. This benhcmark has been replaced by MTEB(Multilingual, v2) as one of the datasets (SNLHierarchicalClustering) included in v1 was removed from the Hugging Face Hub.

[Learn more →](https://arxiv.org/abs/2502.13595)



??? info "Tasks"

    | name                                                                                                                         | type                     | modalities   | languages                          |
    |:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------|:-------------|:-----------------------------------|
    | [BornholmBitextMining](./available_tasks/bitextmining.md#bornholmbitextmining)                                               | BitextMining             | text         | dan                                |
    | [BibleNLPBitextMining](./available_tasks/bitextmining.md#biblenlpbitextmining)                                               | BitextMining             | text         | aai, aak, aau, aaz, abt, ... (829) |
    | [BUCC.v2](./available_tasks/bitextmining.md#bucc.v2)                                                                         | BitextMining             | text         | cmn, deu, eng, fra, rus            |
    | [DiaBlaBitextMining](./available_tasks/bitextmining.md#diablabitextmining)                                                   | BitextMining             | text         | eng, fra                           |
    | [FloresBitextMining](./available_tasks/bitextmining.md#floresbitextmining)                                                   | BitextMining             | text         | ace, acm, acq, aeb, afr, ... (196) |
    | [IN22GenBitextMining](./available_tasks/bitextmining.md#in22genbitextmining)                                                 | BitextMining             | text         | asm, ben, brx, doi, eng, ... (23)  |
    | [IndicGenBenchFloresBitextMining](./available_tasks/bitextmining.md#indicgenbenchfloresbitextmining)                         | BitextMining             | text         | asm, awa, ben, bgc, bho, ... (30)  |
    | [NollySentiBitextMining](./available_tasks/bitextmining.md#nollysentibitextmining)                                           | BitextMining             | text         | eng, hau, ibo, pcm, yor            |
    | [NorwegianCourtsBitextMining](./available_tasks/bitextmining.md#norwegiancourtsbitextmining)                                 | BitextMining             | text         | nno, nob                           |
    | [NTREXBitextMining](./available_tasks/bitextmining.md#ntrexbitextmining)                                                     | BitextMining             | text         | afr, amh, arb, aze, bak, ... (119) |
    | [NusaTranslationBitextMining](./available_tasks/bitextmining.md#nusatranslationbitextmining)                                 | BitextMining             | text         | abs, bbc, bew, bhp, ind, ... (12)  |
    | [NusaXBitextMining](./available_tasks/bitextmining.md#nusaxbitextmining)                                                     | BitextMining             | text         | ace, ban, bbc, bjn, bug, ... (12)  |
    | [Tatoeba](./available_tasks/bitextmining.md#tatoeba)                                                                         | BitextMining             | text         | afr, amh, ang, ara, arq, ... (113) |
    | [BulgarianStoreReviewSentimentClassfication](./available_tasks/classification.md#bulgarianstorereviewsentimentclassfication) | Classification           | text         | bul                                |
    | [CzechProductReviewSentimentClassification](./available_tasks/classification.md#czechproductreviewsentimentclassification)   | Classification           | text         | ces                                |
    | [GreekLegalCodeClassification](./available_tasks/classification.md#greeklegalcodeclassification)                             | Classification           | text         | ell                                |
    | [DBpediaClassification](./available_tasks/classification.md#dbpediaclassification)                                           | Classification           | text         | eng                                |
    | [FinancialPhrasebankClassification](./available_tasks/classification.md#financialphrasebankclassification)                   | Classification           | text         | eng                                |
    | [PoemSentimentClassification](./available_tasks/classification.md#poemsentimentclassification)                               | Classification           | text         | eng                                |
    | [ToxicConversationsClassification](./available_tasks/classification.md#toxicconversationsclassification)                     | Classification           | text         | eng                                |
    | [TweetTopicSingleClassification](./available_tasks/classification.md#tweettopicsingleclassification)                         | Classification           | text         | eng                                |
    | [EstonianValenceClassification](./available_tasks/classification.md#estonianvalenceclassification)                           | Classification           | text         | est                                |
    | [FilipinoShopeeReviewsClassification](./available_tasks/classification.md#filipinoshopeereviewsclassification)               | Classification           | text         | fil                                |
    | [GujaratiNewsClassification](./available_tasks/classification.md#gujaratinewsclassification)                                 | Classification           | text         | guj                                |
    | [SentimentAnalysisHindi](./available_tasks/classification.md#sentimentanalysishindi)                                         | Classification           | text         | hin                                |
    | [IndonesianIdClickbaitClassification](./available_tasks/classification.md#indonesianidclickbaitclassification)               | Classification           | text         | ind                                |
    | [ItaCaseholdClassification](./available_tasks/classification.md#itacaseholdclassification)                                   | Classification           | text         | ita                                |
    | [KorSarcasmClassification](./available_tasks/classification.md#korsarcasmclassification)                                     | Classification           | text         | kor                                |
    | [KurdishSentimentClassification](./available_tasks/classification.md#kurdishsentimentclassification)                         | Classification           | text         | kur                                |
    | [MacedonianTweetSentimentClassification](./available_tasks/classification.md#macedoniantweetsentimentclassification)         | Classification           | text         | mkd                                |
    | [AfriSentiClassification](./available_tasks/classification.md#afrisenticlassification)                                       | Classification           | text         | amh, arq, ary, hau, ibo, ... (12)  |
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification)                 | Classification           | text         | deu, eng, jpn                      |
    | [CataloniaTweetClassification](./available_tasks/classification.md#cataloniatweetclassification)                             | Classification           | text         | cat, spa                           |
    | [CyrillicTurkicLangClassification](./available_tasks/classification.md#cyrillicturkiclangclassification)                     | Classification           | text         | bak, chv, kaz, kir, krc, ... (9)   |
    | [IndicLangClassification](./available_tasks/classification.md#indiclangclassification)                                       | Classification           | text         | asm, ben, brx, doi, gom, ... (22)  |
    | [MasakhaNEWSClassification](./available_tasks/classification.md#masakhanewsclassification)                                   | Classification           | text         | amh, eng, fra, hau, ibo, ... (16)  |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                               | Classification           | text         | afr, amh, ara, aze, ben, ... (50)  |
    | [MultiHateClassification](./available_tasks/classification.md#multihateclassification)                                       | Classification           | text         | ara, cmn, deu, eng, fra, ... (11)  |
    | [NordicLangClassification](./available_tasks/classification.md#nordiclangclassification)                                     | Classification           | text         | dan, fao, isl, nno, nob, ... (6)   |
    | [NusaParagraphEmotionClassification](./available_tasks/classification.md#nusaparagraphemotionclassification)                 | Classification           | text         | bbc, bew, bug, jav, mad, ... (10)  |
    | [NusaX-senti](./available_tasks/classification.md#nusax-senti)                                                               | Classification           | text         | ace, ban, bbc, bjn, bug, ... (12)  |
    | [ScalaClassification](./available_tasks/classification.md#scalaclassification)                                               | Classification           | text         | dan, nno, nob, swe                 |
    | [SwissJudgementClassification](./available_tasks/classification.md#swissjudgementclassification)                             | Classification           | text         | deu, fra, ita                      |
    | [NepaliNewsClassification](./available_tasks/classification.md#nepalinewsclassification)                                     | Classification           | text         | nep                                |
    | [OdiaNewsClassification](./available_tasks/classification.md#odianewsclassification)                                         | Classification           | text         | ory                                |
    | [PunjabiNewsClassification](./available_tasks/classification.md#punjabinewsclassification)                                   | Classification           | text         | pan                                |
    | [PolEmo2.0-OUT](./available_tasks/classification.md#polemo2.0-out)                                                           | Classification           | text         | pol                                |
    | [PAC](./available_tasks/classification.md#pac)                                                                               | Classification           | text         | pol                                |
    | [SinhalaNewsClassification](./available_tasks/classification.md#sinhalanewsclassification)                                   | Classification           | text         | sin                                |
    | [CSFDSKMovieReviewSentimentClassification](./available_tasks/classification.md#csfdskmoviereviewsentimentclassification)     | Classification           | text         | slk                                |
    | [SiswatiNewsClassification](./available_tasks/classification.md#siswatinewsclassification)                                   | Classification           | text         | ssw                                |
    | [SlovakMovieReviewSentimentClassification](./available_tasks/classification.md#slovakmoviereviewsentimentclassification)     | Classification           | text         | slk                                |
    | [SwahiliNewsClassification](./available_tasks/classification.md#swahilinewsclassification)                                   | Classification           | text         | swa                                |
    | [DalajClassification](./available_tasks/classification.md#dalajclassification)                                               | Classification           | text         | swe                                |
    | [TswanaNewsClassification](./available_tasks/classification.md#tswananewsclassification)                                     | Classification           | text         | tsn                                |
    | [IsiZuluNewsClassification](./available_tasks/classification.md#isizulunewsclassification)                                   | Classification           | text         | zul                                |
    | [WikiCitiesClustering](./available_tasks/clustering.md#wikicitiesclustering)                                                 | Clustering               | text         | eng                                |
    | [MasakhaNEWSClusteringS2S](./available_tasks/clustering.md#masakhanewsclusterings2s)                                         | Clustering               | text         | amh, eng, fra, hau, ibo, ... (16)  |
    | [RomaniBibleClustering](./available_tasks/clustering.md#romanibibleclustering)                                               | Clustering               | text         | rom                                |
    | [ArXivHierarchicalClusteringP2P](./available_tasks/clustering.md#arxivhierarchicalclusteringp2p)                             | Clustering               | text         | eng                                |
    | [ArXivHierarchicalClusteringS2S](./available_tasks/clustering.md#arxivhierarchicalclusterings2s)                             | Clustering               | text         | eng                                |
    | [BigPatentClustering.v2](./available_tasks/clustering.md#bigpatentclustering.v2)                                             | Clustering               | text         | eng                                |
    | [BiorxivClusteringP2P.v2](./available_tasks/clustering.md#biorxivclusteringp2p.v2)                                           | Clustering               | text         | eng                                |
    | [MedrxivClusteringP2P.v2](./available_tasks/clustering.md#medrxivclusteringp2p.v2)                                           | Clustering               | text         | eng                                |
    | [StackExchangeClustering.v2](./available_tasks/clustering.md#stackexchangeclustering.v2)                                     | Clustering               | text         | eng                                |
    | [AlloProfClusteringS2S.v2](./available_tasks/clustering.md#alloprofclusterings2s.v2)                                         | Clustering               | text         | fra                                |
    | [HALClusteringS2S.v2](./available_tasks/clustering.md#halclusterings2s.v2)                                                   | Clustering               | text         | fra                                |
    | [SIB200ClusteringS2S](./available_tasks/clustering.md#sib200clusterings2s)                                                   | Clustering               | text         | ace, acm, acq, aeb, afr, ... (197) |
    | [WikiClusteringP2P.v2](./available_tasks/clustering.md#wikiclusteringp2p.v2)                                                 | Clustering               | text         | bos, cat, ces, dan, eus, ... (14)  |
    | [PlscClusteringP2P.v2](./available_tasks/clustering.md#plscclusteringp2p.v2)                                                 | Clustering               | text         | pol                                |
    | [SwednClusteringP2P](./available_tasks/clustering.md#swednclusteringp2p)                                                     | Clustering               | text         | swe                                |
    | [CLSClusteringP2P.v2](./available_tasks/clustering.md#clsclusteringp2p.v2)                                                   | Clustering               | text         | cmn                                |
    | [StackOverflowQA](./available_tasks/retrieval.md#stackoverflowqa)                                                            | Retrieval                | text         | eng                                |
    | [TwitterHjerneRetrieval](./available_tasks/retrieval.md#twitterhjerneretrieval)                                              | Retrieval                | text         | dan                                |
    | [AILAStatutes](./available_tasks/retrieval.md#ailastatutes)                                                                  | Retrieval                | text         | eng                                |
    | [ArguAna](./available_tasks/retrieval.md#arguana)                                                                            | Retrieval                | text         | eng                                |
    | [HagridRetrieval](./available_tasks/retrieval.md#hagridretrieval)                                                            | Retrieval                | text         | eng                                |
    | [LegalBenchCorporateLobbying](./available_tasks/retrieval.md#legalbenchcorporatelobbying)                                    | Retrieval                | text         | eng                                |
    | [LEMBPasskeyRetrieval](./available_tasks/retrieval.md#lembpasskeyretrieval)                                                  | Retrieval                | text         | eng                                |
    | [SCIDOCS](./available_tasks/retrieval.md#scidocs)                                                                            | Retrieval                | text         | eng                                |
    | [SpartQA](./available_tasks/retrieval.md#spartqa)                                                                            | Retrieval                | text         | eng                                |
    | [TempReasonL1](./available_tasks/retrieval.md#tempreasonl1)                                                                  | Retrieval                | text         | eng                                |
    | [TRECCOVID](./available_tasks/retrieval.md#treccovid)                                                                        | Retrieval                | text         | eng                                |
    | [WinoGrande](./available_tasks/retrieval.md#winogrande)                                                                      | Retrieval                | text         | eng                                |
    | [BelebeleRetrieval](./available_tasks/retrieval.md#belebeleretrieval)                                                        | Retrieval                | text         | acm, afr, als, amh, apc, ... (115) |
    | [MLQARetrieval](./available_tasks/retrieval.md#mlqaretrieval)                                                                | Retrieval                | text         | ara, deu, eng, hin, spa, ... (7)   |
    | [StatcanDialogueDatasetRetrieval](./available_tasks/retrieval.md#statcandialoguedatasetretrieval)                            | Retrieval                | text         | eng, fra                           |
    | [WikipediaRetrievalMultilingual](./available_tasks/retrieval.md#wikipediaretrievalmultilingual)                              | Retrieval                | text         | ben, bul, ces, dan, deu, ... (16)  |
    | [CovidRetrieval](./available_tasks/retrieval.md#covidretrieval)                                                              | Retrieval                | text         | cmn                                |
    | [Core17InstructionRetrieval](./available_tasks/instructionreranking.md#core17instructionretrieval)                           | InstructionReranking     | text         | eng                                |
    | [News21InstructionRetrieval](./available_tasks/instructionreranking.md#news21instructionretrieval)                           | InstructionReranking     | text         | eng                                |
    | [Robust04InstructionRetrieval](./available_tasks/instructionreranking.md#robust04instructionretrieval)                       | InstructionReranking     | text         | eng                                |
    | [KorHateSpeechMLClassification](./available_tasks/multilabelclassification.md#korhatespeechmlclassification)                 | MultilabelClassification | text         | kor                                |
    | [MalteseNewsClassification](./available_tasks/multilabelclassification.md#maltesenewsclassification)                         | MultilabelClassification | text         | mlt                                |
    | [MultiEURLEXMultilabelClassification](./available_tasks/multilabelclassification.md#multieurlexmultilabelclassification)     | MultilabelClassification | text         | bul, ces, dan, deu, ell, ... (23)  |
    | [BrazilianToxicTweetsClassification](./available_tasks/multilabelclassification.md#braziliantoxictweetsclassification)       | MultilabelClassification | text         | por                                |
    | [CEDRClassification](./available_tasks/multilabelclassification.md#cedrclassification)                                       | MultilabelClassification | text         | rus                                |
    | [CTKFactsNLI](./available_tasks/pairclassification.md#ctkfactsnli)                                                           | PairClassification       | text         | ces                                |
    | [SprintDuplicateQuestions](./available_tasks/pairclassification.md#sprintduplicatequestions)                                 | PairClassification       | text         | eng                                |
    | [TwitterURLCorpus](./available_tasks/pairclassification.md#twitterurlcorpus)                                                 | PairClassification       | text         | eng                                |
    | [ArmenianParaphrasePC](./available_tasks/pairclassification.md#armenianparaphrasepc)                                         | PairClassification       | text         | hye                                |
    | [indonli](./available_tasks/pairclassification.md#indonli)                                                                   | PairClassification       | text         | ind                                |
    | [OpusparcusPC](./available_tasks/pairclassification.md#opusparcuspc)                                                         | PairClassification       | text         | deu, eng, fin, fra, rus, ... (6)   |
    | [PawsXPairClassification](./available_tasks/pairclassification.md#pawsxpairclassification)                                   | PairClassification       | text         | cmn, deu, eng, fra, jpn, ... (7)   |
    | [RTE3](./available_tasks/pairclassification.md#rte3)                                                                         | PairClassification       | text         | deu, eng, fra, ita                 |
    | [XNLI](./available_tasks/pairclassification.md#xnli)                                                                         | PairClassification       | text         | ara, bul, deu, ell, eng, ... (14)  |
    | [PpcPC](./available_tasks/pairclassification.md#ppcpc)                                                                       | PairClassification       | text         | pol                                |
    | [TERRa](./available_tasks/pairclassification.md#terra)                                                                       | PairClassification       | text         | rus                                |
    | [WebLINXCandidatesReranking](./available_tasks/reranking.md#weblinxcandidatesreranking)                                      | Reranking                | text         | eng                                |
    | [AlloprofReranking](./available_tasks/reranking.md#alloprofreranking)                                                        | Reranking                | text         | fra                                |
    | [VoyageMMarcoReranking](./available_tasks/reranking.md#voyagemmarcoreranking)                                                | Reranking                | text         | jpn                                |
    | [WikipediaRerankingMultilingual](./available_tasks/reranking.md#wikipediarerankingmultilingual)                              | Reranking                | text         | ben, bul, ces, dan, deu, ... (18)  |
    | [RuBQReranking](./available_tasks/reranking.md#rubqreranking)                                                                | Reranking                | text         | rus                                |
    | [T2Reranking](./available_tasks/reranking.md#t2reranking)                                                                    | Reranking                | text         | cmn                                |
    | [GermanSTSBenchmark](./available_tasks/sts.md#germanstsbenchmark)                                                            | STS                      | text         | deu                                |
    | [SICK-R](./available_tasks/sts.md#sick-r)                                                                                    | STS                      | text         | eng                                |
    | [STS12](./available_tasks/sts.md#sts12)                                                                                      | STS                      | text         | eng                                |
    | [STS13](./available_tasks/sts.md#sts13)                                                                                      | STS                      | text         | eng                                |
    | [STS14](./available_tasks/sts.md#sts14)                                                                                      | STS                      | text         | eng                                |
    | [STS15](./available_tasks/sts.md#sts15)                                                                                      | STS                      | text         | eng                                |
    | [STSBenchmark](./available_tasks/sts.md#stsbenchmark)                                                                        | STS                      | text         | eng                                |
    | [FaroeseSTS](./available_tasks/sts.md#faroesests)                                                                            | STS                      | text         | fao                                |
    | [FinParaSTS](./available_tasks/sts.md#finparasts)                                                                            | STS                      | text         | fin                                |
    | [JSICK](./available_tasks/sts.md#jsick)                                                                                      | STS                      | text         | jpn                                |
    | [IndicCrosslingualSTS](./available_tasks/sts.md#indiccrosslingualsts)                                                        | STS                      | text         | asm, ben, eng, guj, hin, ... (13)  |
    | [SemRel24STS](./available_tasks/sts.md#semrel24sts)                                                                          | STS                      | text         | afr, amh, arb, arq, ary, ... (12)  |
    | [STS17](./available_tasks/sts.md#sts17)                                                                                      | STS                      | text         | ara, deu, eng, fra, ita, ... (9)   |
    | [STS22.v2](./available_tasks/sts.md#sts22.v2)                                                                                | STS                      | text         | ara, cmn, deu, eng, fra, ... (10)  |
    | [STSES](./available_tasks/sts.md#stses)                                                                                      | STS                      | text         | spa                                |
    | [STSB](./available_tasks/sts.md#stsb)                                                                                        | STS                      | text         | cmn                                |
    | [MIRACLRetrievalHardNegatives](./available_tasks/retrieval.md#miraclretrievalhardnegatives)                                  | Retrieval                | text         | ara, ben, deu, eng, fas, ... (18)  |
    | [SNLHierarchicalClusteringP2P](./available_tasks/clustering.md#snlhierarchicalclusteringp2p)                                 | Clustering               | text         | nob                                |


??? quote "Citation"


    ```bibtex
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      doi = {10.48550/arXiv.2502.13595},
      journal = {arXiv preprint arXiv:2502.13595},
      publisher = {arXiv},
      title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2502.13595},
      year = {2025},
    }
    ```




####  MTEB(Multilingual, v2)

A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages.

[Learn more →](https://arxiv.org/abs/2502.13595)



??? info "Tasks"

    | name                                                                                                                         | type                     | modalities   | languages                          |
    |:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------|:-------------|:-----------------------------------|
    | [BornholmBitextMining](./available_tasks/bitextmining.md#bornholmbitextmining)                                               | BitextMining             | text         | dan                                |
    | [BibleNLPBitextMining](./available_tasks/bitextmining.md#biblenlpbitextmining)                                               | BitextMining             | text         | aai, aak, aau, aaz, abt, ... (829) |
    | [BUCC.v2](./available_tasks/bitextmining.md#bucc.v2)                                                                         | BitextMining             | text         | cmn, deu, eng, fra, rus            |
    | [DiaBlaBitextMining](./available_tasks/bitextmining.md#diablabitextmining)                                                   | BitextMining             | text         | eng, fra                           |
    | [FloresBitextMining](./available_tasks/bitextmining.md#floresbitextmining)                                                   | BitextMining             | text         | ace, acm, acq, aeb, afr, ... (196) |
    | [IN22GenBitextMining](./available_tasks/bitextmining.md#in22genbitextmining)                                                 | BitextMining             | text         | asm, ben, brx, doi, eng, ... (23)  |
    | [IndicGenBenchFloresBitextMining](./available_tasks/bitextmining.md#indicgenbenchfloresbitextmining)                         | BitextMining             | text         | asm, awa, ben, bgc, bho, ... (30)  |
    | [NollySentiBitextMining](./available_tasks/bitextmining.md#nollysentibitextmining)                                           | BitextMining             | text         | eng, hau, ibo, pcm, yor            |
    | [NorwegianCourtsBitextMining](./available_tasks/bitextmining.md#norwegiancourtsbitextmining)                                 | BitextMining             | text         | nno, nob                           |
    | [NTREXBitextMining](./available_tasks/bitextmining.md#ntrexbitextmining)                                                     | BitextMining             | text         | afr, amh, arb, aze, bak, ... (119) |
    | [NusaTranslationBitextMining](./available_tasks/bitextmining.md#nusatranslationbitextmining)                                 | BitextMining             | text         | abs, bbc, bew, bhp, ind, ... (12)  |
    | [NusaXBitextMining](./available_tasks/bitextmining.md#nusaxbitextmining)                                                     | BitextMining             | text         | ace, ban, bbc, bjn, bug, ... (12)  |
    | [Tatoeba](./available_tasks/bitextmining.md#tatoeba)                                                                         | BitextMining             | text         | afr, amh, ang, ara, arq, ... (113) |
    | [BulgarianStoreReviewSentimentClassfication](./available_tasks/classification.md#bulgarianstorereviewsentimentclassfication) | Classification           | text         | bul                                |
    | [CzechProductReviewSentimentClassification](./available_tasks/classification.md#czechproductreviewsentimentclassification)   | Classification           | text         | ces                                |
    | [GreekLegalCodeClassification](./available_tasks/classification.md#greeklegalcodeclassification)                             | Classification           | text         | ell                                |
    | [DBpediaClassification](./available_tasks/classification.md#dbpediaclassification)                                           | Classification           | text         | eng                                |
    | [FinancialPhrasebankClassification](./available_tasks/classification.md#financialphrasebankclassification)                   | Classification           | text         | eng                                |
    | [PoemSentimentClassification](./available_tasks/classification.md#poemsentimentclassification)                               | Classification           | text         | eng                                |
    | [ToxicConversationsClassification](./available_tasks/classification.md#toxicconversationsclassification)                     | Classification           | text         | eng                                |
    | [TweetTopicSingleClassification](./available_tasks/classification.md#tweettopicsingleclassification)                         | Classification           | text         | eng                                |
    | [EstonianValenceClassification](./available_tasks/classification.md#estonianvalenceclassification)                           | Classification           | text         | est                                |
    | [FilipinoShopeeReviewsClassification](./available_tasks/classification.md#filipinoshopeereviewsclassification)               | Classification           | text         | fil                                |
    | [GujaratiNewsClassification](./available_tasks/classification.md#gujaratinewsclassification)                                 | Classification           | text         | guj                                |
    | [SentimentAnalysisHindi](./available_tasks/classification.md#sentimentanalysishindi)                                         | Classification           | text         | hin                                |
    | [IndonesianIdClickbaitClassification](./available_tasks/classification.md#indonesianidclickbaitclassification)               | Classification           | text         | ind                                |
    | [ItaCaseholdClassification](./available_tasks/classification.md#itacaseholdclassification)                                   | Classification           | text         | ita                                |
    | [KorSarcasmClassification](./available_tasks/classification.md#korsarcasmclassification)                                     | Classification           | text         | kor                                |
    | [KurdishSentimentClassification](./available_tasks/classification.md#kurdishsentimentclassification)                         | Classification           | text         | kur                                |
    | [MacedonianTweetSentimentClassification](./available_tasks/classification.md#macedoniantweetsentimentclassification)         | Classification           | text         | mkd                                |
    | [AfriSentiClassification](./available_tasks/classification.md#afrisenticlassification)                                       | Classification           | text         | amh, arq, ary, hau, ibo, ... (12)  |
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification)                 | Classification           | text         | deu, eng, jpn                      |
    | [CataloniaTweetClassification](./available_tasks/classification.md#cataloniatweetclassification)                             | Classification           | text         | cat, spa                           |
    | [CyrillicTurkicLangClassification](./available_tasks/classification.md#cyrillicturkiclangclassification)                     | Classification           | text         | bak, chv, kaz, kir, krc, ... (9)   |
    | [IndicLangClassification](./available_tasks/classification.md#indiclangclassification)                                       | Classification           | text         | asm, ben, brx, doi, gom, ... (22)  |
    | [MasakhaNEWSClassification](./available_tasks/classification.md#masakhanewsclassification)                                   | Classification           | text         | amh, eng, fra, hau, ibo, ... (16)  |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                               | Classification           | text         | afr, amh, ara, aze, ben, ... (50)  |
    | [MultiHateClassification](./available_tasks/classification.md#multihateclassification)                                       | Classification           | text         | ara, cmn, deu, eng, fra, ... (11)  |
    | [NordicLangClassification](./available_tasks/classification.md#nordiclangclassification)                                     | Classification           | text         | dan, fao, isl, nno, nob, ... (6)   |
    | [NusaParagraphEmotionClassification](./available_tasks/classification.md#nusaparagraphemotionclassification)                 | Classification           | text         | bbc, bew, bug, jav, mad, ... (10)  |
    | [NusaX-senti](./available_tasks/classification.md#nusax-senti)                                                               | Classification           | text         | ace, ban, bbc, bjn, bug, ... (12)  |
    | [ScalaClassification](./available_tasks/classification.md#scalaclassification)                                               | Classification           | text         | dan, nno, nob, swe                 |
    | [SwissJudgementClassification](./available_tasks/classification.md#swissjudgementclassification)                             | Classification           | text         | deu, fra, ita                      |
    | [NepaliNewsClassification](./available_tasks/classification.md#nepalinewsclassification)                                     | Classification           | text         | nep                                |
    | [OdiaNewsClassification](./available_tasks/classification.md#odianewsclassification)                                         | Classification           | text         | ory                                |
    | [PunjabiNewsClassification](./available_tasks/classification.md#punjabinewsclassification)                                   | Classification           | text         | pan                                |
    | [PolEmo2.0-OUT](./available_tasks/classification.md#polemo2.0-out)                                                           | Classification           | text         | pol                                |
    | [PAC](./available_tasks/classification.md#pac)                                                                               | Classification           | text         | pol                                |
    | [SinhalaNewsClassification](./available_tasks/classification.md#sinhalanewsclassification)                                   | Classification           | text         | sin                                |
    | [CSFDSKMovieReviewSentimentClassification](./available_tasks/classification.md#csfdskmoviereviewsentimentclassification)     | Classification           | text         | slk                                |
    | [SiswatiNewsClassification](./available_tasks/classification.md#siswatinewsclassification)                                   | Classification           | text         | ssw                                |
    | [SlovakMovieReviewSentimentClassification](./available_tasks/classification.md#slovakmoviereviewsentimentclassification)     | Classification           | text         | slk                                |
    | [SwahiliNewsClassification](./available_tasks/classification.md#swahilinewsclassification)                                   | Classification           | text         | swa                                |
    | [DalajClassification](./available_tasks/classification.md#dalajclassification)                                               | Classification           | text         | swe                                |
    | [TswanaNewsClassification](./available_tasks/classification.md#tswananewsclassification)                                     | Classification           | text         | tsn                                |
    | [IsiZuluNewsClassification](./available_tasks/classification.md#isizulunewsclassification)                                   | Classification           | text         | zul                                |
    | [WikiCitiesClustering](./available_tasks/clustering.md#wikicitiesclustering)                                                 | Clustering               | text         | eng                                |
    | [MasakhaNEWSClusteringS2S](./available_tasks/clustering.md#masakhanewsclusterings2s)                                         | Clustering               | text         | amh, eng, fra, hau, ibo, ... (16)  |
    | [RomaniBibleClustering](./available_tasks/clustering.md#romanibibleclustering)                                               | Clustering               | text         | rom                                |
    | [ArXivHierarchicalClusteringP2P](./available_tasks/clustering.md#arxivhierarchicalclusteringp2p)                             | Clustering               | text         | eng                                |
    | [ArXivHierarchicalClusteringS2S](./available_tasks/clustering.md#arxivhierarchicalclusterings2s)                             | Clustering               | text         | eng                                |
    | [BigPatentClustering.v2](./available_tasks/clustering.md#bigpatentclustering.v2)                                             | Clustering               | text         | eng                                |
    | [BiorxivClusteringP2P.v2](./available_tasks/clustering.md#biorxivclusteringp2p.v2)                                           | Clustering               | text         | eng                                |
    | [MedrxivClusteringP2P.v2](./available_tasks/clustering.md#medrxivclusteringp2p.v2)                                           | Clustering               | text         | eng                                |
    | [StackExchangeClustering.v2](./available_tasks/clustering.md#stackexchangeclustering.v2)                                     | Clustering               | text         | eng                                |
    | [AlloProfClusteringS2S.v2](./available_tasks/clustering.md#alloprofclusterings2s.v2)                                         | Clustering               | text         | fra                                |
    | [HALClusteringS2S.v2](./available_tasks/clustering.md#halclusterings2s.v2)                                                   | Clustering               | text         | fra                                |
    | [SIB200ClusteringS2S](./available_tasks/clustering.md#sib200clusterings2s)                                                   | Clustering               | text         | ace, acm, acq, aeb, afr, ... (197) |
    | [WikiClusteringP2P.v2](./available_tasks/clustering.md#wikiclusteringp2p.v2)                                                 | Clustering               | text         | bos, cat, ces, dan, eus, ... (14)  |
    | [PlscClusteringP2P.v2](./available_tasks/clustering.md#plscclusteringp2p.v2)                                                 | Clustering               | text         | pol                                |
    | [SwednClusteringP2P](./available_tasks/clustering.md#swednclusteringp2p)                                                     | Clustering               | text         | swe                                |
    | [CLSClusteringP2P.v2](./available_tasks/clustering.md#clsclusteringp2p.v2)                                                   | Clustering               | text         | cmn                                |
    | [StackOverflowQA](./available_tasks/retrieval.md#stackoverflowqa)                                                            | Retrieval                | text         | eng                                |
    | [TwitterHjerneRetrieval](./available_tasks/retrieval.md#twitterhjerneretrieval)                                              | Retrieval                | text         | dan                                |
    | [AILAStatutes](./available_tasks/retrieval.md#ailastatutes)                                                                  | Retrieval                | text         | eng                                |
    | [ArguAna](./available_tasks/retrieval.md#arguana)                                                                            | Retrieval                | text         | eng                                |
    | [HagridRetrieval](./available_tasks/retrieval.md#hagridretrieval)                                                            | Retrieval                | text         | eng                                |
    | [LegalBenchCorporateLobbying](./available_tasks/retrieval.md#legalbenchcorporatelobbying)                                    | Retrieval                | text         | eng                                |
    | [LEMBPasskeyRetrieval](./available_tasks/retrieval.md#lembpasskeyretrieval)                                                  | Retrieval                | text         | eng                                |
    | [SCIDOCS](./available_tasks/retrieval.md#scidocs)                                                                            | Retrieval                | text         | eng                                |
    | [SpartQA](./available_tasks/retrieval.md#spartqa)                                                                            | Retrieval                | text         | eng                                |
    | [TempReasonL1](./available_tasks/retrieval.md#tempreasonl1)                                                                  | Retrieval                | text         | eng                                |
    | [TRECCOVID](./available_tasks/retrieval.md#treccovid)                                                                        | Retrieval                | text         | eng                                |
    | [WinoGrande](./available_tasks/retrieval.md#winogrande)                                                                      | Retrieval                | text         | eng                                |
    | [BelebeleRetrieval](./available_tasks/retrieval.md#belebeleretrieval)                                                        | Retrieval                | text         | acm, afr, als, amh, apc, ... (115) |
    | [MLQARetrieval](./available_tasks/retrieval.md#mlqaretrieval)                                                                | Retrieval                | text         | ara, deu, eng, hin, spa, ... (7)   |
    | [StatcanDialogueDatasetRetrieval](./available_tasks/retrieval.md#statcandialoguedatasetretrieval)                            | Retrieval                | text         | eng, fra                           |
    | [WikipediaRetrievalMultilingual](./available_tasks/retrieval.md#wikipediaretrievalmultilingual)                              | Retrieval                | text         | ben, bul, ces, dan, deu, ... (16)  |
    | [CovidRetrieval](./available_tasks/retrieval.md#covidretrieval)                                                              | Retrieval                | text         | cmn                                |
    | [Core17InstructionRetrieval](./available_tasks/instructionreranking.md#core17instructionretrieval)                           | InstructionReranking     | text         | eng                                |
    | [News21InstructionRetrieval](./available_tasks/instructionreranking.md#news21instructionretrieval)                           | InstructionReranking     | text         | eng                                |
    | [Robust04InstructionRetrieval](./available_tasks/instructionreranking.md#robust04instructionretrieval)                       | InstructionReranking     | text         | eng                                |
    | [KorHateSpeechMLClassification](./available_tasks/multilabelclassification.md#korhatespeechmlclassification)                 | MultilabelClassification | text         | kor                                |
    | [MalteseNewsClassification](./available_tasks/multilabelclassification.md#maltesenewsclassification)                         | MultilabelClassification | text         | mlt                                |
    | [MultiEURLEXMultilabelClassification](./available_tasks/multilabelclassification.md#multieurlexmultilabelclassification)     | MultilabelClassification | text         | bul, ces, dan, deu, ell, ... (23)  |
    | [BrazilianToxicTweetsClassification](./available_tasks/multilabelclassification.md#braziliantoxictweetsclassification)       | MultilabelClassification | text         | por                                |
    | [CEDRClassification](./available_tasks/multilabelclassification.md#cedrclassification)                                       | MultilabelClassification | text         | rus                                |
    | [CTKFactsNLI](./available_tasks/pairclassification.md#ctkfactsnli)                                                           | PairClassification       | text         | ces                                |
    | [SprintDuplicateQuestions](./available_tasks/pairclassification.md#sprintduplicatequestions)                                 | PairClassification       | text         | eng                                |
    | [TwitterURLCorpus](./available_tasks/pairclassification.md#twitterurlcorpus)                                                 | PairClassification       | text         | eng                                |
    | [ArmenianParaphrasePC](./available_tasks/pairclassification.md#armenianparaphrasepc)                                         | PairClassification       | text         | hye                                |
    | [indonli](./available_tasks/pairclassification.md#indonli)                                                                   | PairClassification       | text         | ind                                |
    | [OpusparcusPC](./available_tasks/pairclassification.md#opusparcuspc)                                                         | PairClassification       | text         | deu, eng, fin, fra, rus, ... (6)   |
    | [PawsXPairClassification](./available_tasks/pairclassification.md#pawsxpairclassification)                                   | PairClassification       | text         | cmn, deu, eng, fra, jpn, ... (7)   |
    | [RTE3](./available_tasks/pairclassification.md#rte3)                                                                         | PairClassification       | text         | deu, eng, fra, ita                 |
    | [XNLI](./available_tasks/pairclassification.md#xnli)                                                                         | PairClassification       | text         | ara, bul, deu, ell, eng, ... (14)  |
    | [PpcPC](./available_tasks/pairclassification.md#ppcpc)                                                                       | PairClassification       | text         | pol                                |
    | [TERRa](./available_tasks/pairclassification.md#terra)                                                                       | PairClassification       | text         | rus                                |
    | [WebLINXCandidatesReranking](./available_tasks/reranking.md#weblinxcandidatesreranking)                                      | Reranking                | text         | eng                                |
    | [AlloprofReranking](./available_tasks/reranking.md#alloprofreranking)                                                        | Reranking                | text         | fra                                |
    | [VoyageMMarcoReranking](./available_tasks/reranking.md#voyagemmarcoreranking)                                                | Reranking                | text         | jpn                                |
    | [WikipediaRerankingMultilingual](./available_tasks/reranking.md#wikipediarerankingmultilingual)                              | Reranking                | text         | ben, bul, ces, dan, deu, ... (18)  |
    | [RuBQReranking](./available_tasks/reranking.md#rubqreranking)                                                                | Reranking                | text         | rus                                |
    | [T2Reranking](./available_tasks/reranking.md#t2reranking)                                                                    | Reranking                | text         | cmn                                |
    | [GermanSTSBenchmark](./available_tasks/sts.md#germanstsbenchmark)                                                            | STS                      | text         | deu                                |
    | [SICK-R](./available_tasks/sts.md#sick-r)                                                                                    | STS                      | text         | eng                                |
    | [STS12](./available_tasks/sts.md#sts12)                                                                                      | STS                      | text         | eng                                |
    | [STS13](./available_tasks/sts.md#sts13)                                                                                      | STS                      | text         | eng                                |
    | [STS14](./available_tasks/sts.md#sts14)                                                                                      | STS                      | text         | eng                                |
    | [STS15](./available_tasks/sts.md#sts15)                                                                                      | STS                      | text         | eng                                |
    | [STSBenchmark](./available_tasks/sts.md#stsbenchmark)                                                                        | STS                      | text         | eng                                |
    | [FaroeseSTS](./available_tasks/sts.md#faroesests)                                                                            | STS                      | text         | fao                                |
    | [FinParaSTS](./available_tasks/sts.md#finparasts)                                                                            | STS                      | text         | fin                                |
    | [JSICK](./available_tasks/sts.md#jsick)                                                                                      | STS                      | text         | jpn                                |
    | [IndicCrosslingualSTS](./available_tasks/sts.md#indiccrosslingualsts)                                                        | STS                      | text         | asm, ben, eng, guj, hin, ... (13)  |
    | [SemRel24STS](./available_tasks/sts.md#semrel24sts)                                                                          | STS                      | text         | afr, amh, arb, arq, ary, ... (12)  |
    | [STS17](./available_tasks/sts.md#sts17)                                                                                      | STS                      | text         | ara, deu, eng, fra, ita, ... (9)   |
    | [STS22.v2](./available_tasks/sts.md#sts22.v2)                                                                                | STS                      | text         | ara, cmn, deu, eng, fra, ... (10)  |
    | [STSES](./available_tasks/sts.md#stses)                                                                                      | STS                      | text         | spa                                |
    | [STSB](./available_tasks/sts.md#stsb)                                                                                        | STS                      | text         | cmn                                |
    | [MIRACLRetrievalHardNegatives](./available_tasks/retrieval.md#miraclretrievalhardnegatives)                                  | Retrieval                | text         | ara, ben, deu, eng, fas, ... (18)  |


??? quote "Citation"


    ```bibtex
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      doi = {10.48550/arXiv.2502.13595},
      journal = {arXiv preprint arXiv:2502.13595},
      publisher = {arXiv},
      title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2502.13595},
      year = {2025},
    }
    ```




####  MTEB(Scandinavian, v1)

A curated selection of tasks coverering the Scandinavian languages; Danish, Swedish and Norwegian, including Bokmål and Nynorsk.

[Learn more →](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)



??? info "Tasks"

    | name                                                                                                               | type           | modalities   | languages                         |
    |:-------------------------------------------------------------------------------------------------------------------|:---------------|:-------------|:----------------------------------|
    | [BornholmBitextMining](./available_tasks/bitextmining.md#bornholmbitextmining)                                     | BitextMining   | text         | dan                               |
    | [NorwegianCourtsBitextMining](./available_tasks/bitextmining.md#norwegiancourtsbitextmining)                       | BitextMining   | text         | nno, nob                          |
    | [AngryTweetsClassification](./available_tasks/classification.md#angrytweetsclassification)                         | Classification | text         | dan                               |
    | [DanishPoliticalCommentsClassification](./available_tasks/classification.md#danishpoliticalcommentsclassification) | Classification | text         | dan                               |
    | [DalajClassification](./available_tasks/classification.md#dalajclassification)                                     | Classification | text         | swe                               |
    | [DKHateClassification](./available_tasks/classification.md#dkhateclassification)                                   | Classification | text         | dan                               |
    | [LccSentimentClassification](./available_tasks/classification.md#lccsentimentclassification)                       | Classification | text         | dan                               |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                     | Classification | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)                 | Classification | text         | afr, amh, ara, aze, ben, ... (50) |
    | [NordicLangClassification](./available_tasks/classification.md#nordiclangclassification)                           | Classification | text         | dan, fao, isl, nno, nob, ... (6)  |
    | [NoRecClassification](./available_tasks/classification.md#norecclassification)                                     | Classification | text         | nob                               |
    | [NorwegianParliamentClassification](./available_tasks/classification.md#norwegianparliamentclassification)         | Classification | text         | nob                               |
    | [ScalaClassification](./available_tasks/classification.md#scalaclassification)                                     | Classification | text         | dan, nno, nob, swe                |
    | [SwedishSentimentClassification](./available_tasks/classification.md#swedishsentimentclassification)               | Classification | text         | swe                               |
    | [SweRecClassification](./available_tasks/classification.md#swerecclassification)                                   | Classification | text         | swe                               |
    | [DanFeverRetrieval](./available_tasks/retrieval.md#danfeverretrieval)                                              | Retrieval      | text         | dan                               |
    | [NorQuadRetrieval](./available_tasks/retrieval.md#norquadretrieval)                                                | Retrieval      | text         | nob                               |
    | [SNLRetrieval](./available_tasks/retrieval.md#snlretrieval)                                                        | Retrieval      | text         | nob                               |
    | [SwednRetrieval](./available_tasks/retrieval.md#swednretrieval)                                                    | Retrieval      | text         | swe                               |
    | [SweFaqRetrieval](./available_tasks/retrieval.md#swefaqretrieval)                                                  | Retrieval      | text         | swe                               |
    | [TV2Nordretrieval](./available_tasks/retrieval.md#tv2nordretrieval)                                                | Retrieval      | text         | dan                               |
    | [TwitterHjerneRetrieval](./available_tasks/retrieval.md#twitterhjerneretrieval)                                    | Retrieval      | text         | dan                               |
    | [SNLHierarchicalClusteringS2S](./available_tasks/clustering.md#snlhierarchicalclusterings2s)                       | Clustering     | text         | nob                               |
    | [SNLHierarchicalClusteringP2P](./available_tasks/clustering.md#snlhierarchicalclusteringp2p)                       | Clustering     | text         | nob                               |
    | [SwednClusteringP2P](./available_tasks/clustering.md#swednclusteringp2p)                                           | Clustering     | text         | swe                               |
    | [SwednClusteringS2S](./available_tasks/clustering.md#swednclusterings2s)                                           | Clustering     | text         | swe                               |
    | [VGHierarchicalClusteringS2S](./available_tasks/clustering.md#vghierarchicalclusterings2s)                         | Clustering     | text         | nob                               |
    | [VGHierarchicalClusteringP2P](./available_tasks/clustering.md#vghierarchicalclusteringp2p)                         | Clustering     | text         | nob                               |


??? quote "Citation"


    ```bibtex

    @inproceedings{enevoldsen2024scandinavian,
      author = {Enevoldsen, Kenneth and Kardos, M{\'a}rton and Muennighoff, Niklas and Nielbo, Kristoffer},
      booktitle = {Advances in Neural Information Processing Systems},
      title = {The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding},
      url = {https://nips.cc/virtual/2024/poster/97869},
      year = {2024},
    }

    ```




####  MTEB(cmn, v1)

The Chinese Massive Text Embedding Benchmark (C-MTEB) is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets.

[Learn more →](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB)



??? info "Tasks"

    | name                                                                               | type               | modalities   | languages   |
    |:-----------------------------------------------------------------------------------|:-------------------|:-------------|:------------|
    | [T2Retrieval](./available_tasks/retrieval.md#t2retrieval)                          | Retrieval          | text         | cmn         |
    | [MMarcoRetrieval](./available_tasks/retrieval.md#mmarcoretrieval)                  | Retrieval          | text         | cmn         |
    | [DuRetrieval](./available_tasks/retrieval.md#duretrieval)                          | Retrieval          | text         | cmn         |
    | [CovidRetrieval](./available_tasks/retrieval.md#covidretrieval)                    | Retrieval          | text         | cmn         |
    | [CmedqaRetrieval](./available_tasks/retrieval.md#cmedqaretrieval)                  | Retrieval          | text         | cmn         |
    | [EcomRetrieval](./available_tasks/retrieval.md#ecomretrieval)                      | Retrieval          | text         | cmn         |
    | [MedicalRetrieval](./available_tasks/retrieval.md#medicalretrieval)                | Retrieval          | text         | cmn         |
    | [VideoRetrieval](./available_tasks/retrieval.md#videoretrieval)                    | Retrieval          | text         | cmn         |
    | [T2Reranking](./available_tasks/reranking.md#t2reranking)                          | Reranking          | text         | cmn         |
    | [MMarcoReranking](./available_tasks/reranking.md#mmarcoreranking)                  | Reranking          | text         | cmn         |
    | [CMedQAv1-reranking](./available_tasks/reranking.md#cmedqav1-reranking)            | Reranking          | text         | cmn         |
    | [CMedQAv2-reranking](./available_tasks/reranking.md#cmedqav2-reranking)            | Reranking          | text         | cmn         |
    | [Ocnli](./available_tasks/pairclassification.md#ocnli)                             | PairClassification | text         | cmn         |
    | [Cmnli](./available_tasks/pairclassification.md#cmnli)                             | PairClassification | text         | cmn         |
    | [CLSClusteringS2S](./available_tasks/clustering.md#clsclusterings2s)               | Clustering         | text         | cmn         |
    | [CLSClusteringP2P](./available_tasks/clustering.md#clsclusteringp2p)               | Clustering         | text         | cmn         |
    | [ThuNewsClusteringS2S](./available_tasks/clustering.md#thunewsclusterings2s)       | Clustering         | text         | cmn         |
    | [ThuNewsClusteringP2P](./available_tasks/clustering.md#thunewsclusteringp2p)       | Clustering         | text         | cmn         |
    | [LCQMC](./available_tasks/sts.md#lcqmc)                                            | STS                | text         | cmn         |
    | [PAWSX](./available_tasks/sts.md#pawsx)                                            | STS                | text         | cmn         |
    | [AFQMC](./available_tasks/sts.md#afqmc)                                            | STS                | text         | cmn         |
    | [QBQTC](./available_tasks/sts.md#qbqtc)                                            | STS                | text         | cmn         |
    | [TNews](./available_tasks/classification.md#tnews)                                 | Classification     | text         | cmn         |
    | [IFlyTek](./available_tasks/classification.md#iflytek)                             | Classification     | text         | cmn         |
    | [Waimai](./available_tasks/classification.md#waimai)                               | Classification     | text         | cmn         |
    | [OnlineShopping](./available_tasks/classification.md#onlineshopping)               | Classification     | text         | cmn         |
    | [JDReview](./available_tasks/classification.md#jdreview)                           | Classification     | text         | cmn         |
    | [MultilingualSentiment](./available_tasks/classification.md#multilingualsentiment) | Classification     | text         | cmn         |
    | [ATEC](./available_tasks/sts.md#atec)                                              | STS                | text         | cmn         |
    | [BQ](./available_tasks/sts.md#bq)                                                  | STS                | text         | cmn         |
    | [STSB](./available_tasks/sts.md#stsb)                                              | STS                | text         | cmn         |
    | [MultilingualSentiment](./available_tasks/classification.md#multilingualsentiment) | Classification     | text         | cmn         |


??? quote "Citation"


    ```bibtex

    @misc{c-pack,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      year = {2023},
    }

    ```




####  MTEB(deu, v1)

A benchmark for text-embedding performance in German.

[Learn more →](https://arxiv.org/html/2401.02709v1)



??? info "Tasks"

    | name                                                                                                         | type               | modalities   | languages                         |
    |:-------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification) | Classification     | text         | deu, eng, jpn                     |
    | [AmazonReviewsClassification](./available_tasks/classification.md#amazonreviewsclassification)               | Classification     | text         | cmn, deu, eng, fra, jpn, ... (6)  |
    | [MTOPDomainClassification](./available_tasks/classification.md#mtopdomainclassification)                     | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [MTOPIntentClassification](./available_tasks/classification.md#mtopintentclassification)                     | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)               | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)           | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [BlurbsClusteringP2P](./available_tasks/clustering.md#blurbsclusteringp2p)                                   | Clustering         | text         | deu                               |
    | [BlurbsClusteringS2S](./available_tasks/clustering.md#blurbsclusterings2s)                                   | Clustering         | text         | deu                               |
    | [TenKGnadClusteringP2P](./available_tasks/clustering.md#tenkgnadclusteringp2p)                               | Clustering         | text         | deu                               |
    | [TenKGnadClusteringS2S](./available_tasks/clustering.md#tenkgnadclusterings2s)                               | Clustering         | text         | deu                               |
    | [FalseFriendsGermanEnglish](./available_tasks/pairclassification.md#falsefriendsgermanenglish)               | PairClassification | text         | deu                               |
    | [PawsXPairClassification](./available_tasks/pairclassification.md#pawsxpairclassification)                   | PairClassification | text         | cmn, deu, eng, fra, jpn, ... (7)  |
    | [MIRACLReranking](./available_tasks/reranking.md#miraclreranking)                                            | Reranking          | text         | ara, ben, deu, eng, fas, ... (18) |
    | [GermanQuAD-Retrieval](./available_tasks/retrieval.md#germanquad-retrieval)                                  | Retrieval          | text         | deu                               |
    | [GermanDPR](./available_tasks/retrieval.md#germandpr)                                                        | Retrieval          | text         | deu                               |
    | [XMarket](./available_tasks/retrieval.md#xmarket)                                                            | Retrieval          | text         | deu, eng, spa                     |
    | [GerDaLIR](./available_tasks/retrieval.md#gerdalir)                                                          | Retrieval          | text         | deu                               |
    | [GermanSTSBenchmark](./available_tasks/sts.md#germanstsbenchmark)                                            | STS                | text         | deu                               |
    | [STS22](./available_tasks/sts.md#sts22)                                                                      | STS                | text         | ara, cmn, deu, eng, fra, ... (10) |


??? quote "Citation"


    ```bibtex

    @misc{wehrli2024germantextembeddingclustering,
      archiveprefix = {arXiv},
      author = {Silvan Wehrli and Bert Arnrich and Christopher Irrgang},
      eprint = {2401.02709},
      primaryclass = {cs.CL},
      title = {German Text Embedding Clustering Benchmark},
      url = {https://arxiv.org/abs/2401.02709},
      year = {2024},
    }

    ```




####  MTEB(eng, v1)

The original English benchmark by Muennighoff et al., (2023).
This page is an adaptation of the [old MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard_legacy).
We recommend that you use [MTEB(eng, v2)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v2%29) instead, as it uses updated versions of the task, making it notably faster to run and resolving [a known bug](https://github.com/embeddings-benchmark/mteb/issues/1156) in existing tasks. This benchmark also removes datasets common for fine-tuning, such as MSMARCO, which makes model performance scores more comparable. However, generally, both benchmarks provide similar estimates.


??? info "Tasks"

    | name                                                                                                                 | type               | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [AmazonPolarityClassification](./available_tasks/classification.md#amazonpolarityclassification)                     | Classification     | text         | eng                               |
    | [AmazonReviewsClassification](./available_tasks/classification.md#amazonreviewsclassification)                       | Classification     | text         | cmn, deu, eng, fra, jpn, ... (6)  |
    | [ArguAna](./available_tasks/retrieval.md#arguana)                                                                    | Retrieval          | text         | eng                               |
    | [ArxivClusteringP2P](./available_tasks/clustering.md#arxivclusteringp2p)                                             | Clustering         | text         | eng                               |
    | [ArxivClusteringS2S](./available_tasks/clustering.md#arxivclusterings2s)                                             | Clustering         | text         | eng                               |
    | [AskUbuntuDupQuestions](./available_tasks/reranking.md#askubuntudupquestions)                                        | Reranking          | text         | eng                               |
    | [BIOSSES](./available_tasks/sts.md#biosses)                                                                          | STS                | text         | eng                               |
    | [Banking77Classification](./available_tasks/classification.md#banking77classification)                               | Classification     | text         | eng                               |
    | [BiorxivClusteringP2P](./available_tasks/clustering.md#biorxivclusteringp2p)                                         | Clustering         | text         | eng                               |
    | [BiorxivClusteringS2S](./available_tasks/clustering.md#biorxivclusterings2s)                                         | Clustering         | text         | eng                               |
    | [CQADupstackRetrieval](./available_tasks/retrieval.md#cqadupstackretrieval)                                          | Retrieval          | text         | eng                               |
    | [ClimateFEVER](./available_tasks/retrieval.md#climatefever)                                                          | Retrieval          | text         | eng                               |
    | [DBPedia](./available_tasks/retrieval.md#dbpedia)                                                                    | Retrieval          | text         | eng                               |
    | [EmotionClassification](./available_tasks/classification.md#emotionclassification)                                   | Classification     | text         | eng                               |
    | [FEVER](./available_tasks/retrieval.md#fever)                                                                        | Retrieval          | text         | eng                               |
    | [FiQA2018](./available_tasks/retrieval.md#fiqa2018)                                                                  | Retrieval          | text         | eng                               |
    | [HotpotQA](./available_tasks/retrieval.md#hotpotqa)                                                                  | Retrieval          | text         | eng                               |
    | [ImdbClassification](./available_tasks/classification.md#imdbclassification)                                         | Classification     | text         | eng                               |
    | [MTOPDomainClassification](./available_tasks/classification.md#mtopdomainclassification)                             | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [MTOPIntentClassification](./available_tasks/classification.md#mtopintentclassification)                             | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                       | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)                   | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MedrxivClusteringP2P](./available_tasks/clustering.md#medrxivclusteringp2p)                                         | Clustering         | text         | eng                               |
    | [MedrxivClusteringS2S](./available_tasks/clustering.md#medrxivclusterings2s)                                         | Clustering         | text         | eng                               |
    | [MindSmallReranking](./available_tasks/reranking.md#mindsmallreranking)                                              | Reranking          | text         | eng                               |
    | [NFCorpus](./available_tasks/retrieval.md#nfcorpus)                                                                  | Retrieval          | text         | eng                               |
    | [NQ](./available_tasks/retrieval.md#nq)                                                                              | Retrieval          | text         | eng                               |
    | [QuoraRetrieval](./available_tasks/retrieval.md#quoraretrieval)                                                      | Retrieval          | text         | eng                               |
    | [RedditClustering](./available_tasks/clustering.md#redditclustering)                                                 | Clustering         | text         | eng                               |
    | [RedditClusteringP2P](./available_tasks/clustering.md#redditclusteringp2p)                                           | Clustering         | text         | eng                               |
    | [SCIDOCS](./available_tasks/retrieval.md#scidocs)                                                                    | Retrieval          | text         | eng                               |
    | [SICK-R](./available_tasks/sts.md#sick-r)                                                                            | STS                | text         | eng                               |
    | [STS12](./available_tasks/sts.md#sts12)                                                                              | STS                | text         | eng                               |
    | [STS13](./available_tasks/sts.md#sts13)                                                                              | STS                | text         | eng                               |
    | [STS14](./available_tasks/sts.md#sts14)                                                                              | STS                | text         | eng                               |
    | [STS15](./available_tasks/sts.md#sts15)                                                                              | STS                | text         | eng                               |
    | [STS16](./available_tasks/sts.md#sts16)                                                                              | STS                | text         | eng                               |
    | [STSBenchmark](./available_tasks/sts.md#stsbenchmark)                                                                | STS                | text         | eng                               |
    | [SciDocsRR](./available_tasks/reranking.md#scidocsrr)                                                                | Reranking          | text         | eng                               |
    | [SciFact](./available_tasks/retrieval.md#scifact)                                                                    | Retrieval          | text         | eng                               |
    | [SprintDuplicateQuestions](./available_tasks/pairclassification.md#sprintduplicatequestions)                         | PairClassification | text         | eng                               |
    | [StackExchangeClustering](./available_tasks/clustering.md#stackexchangeclustering)                                   | Clustering         | text         | eng                               |
    | [StackExchangeClusteringP2P](./available_tasks/clustering.md#stackexchangeclusteringp2p)                             | Clustering         | text         | eng                               |
    | [StackOverflowDupQuestions](./available_tasks/reranking.md#stackoverflowdupquestions)                                | Reranking          | text         | eng                               |
    | [SummEval](./available_tasks/summarization.md#summeval)                                                              | Summarization      | text         | eng                               |
    | [TRECCOVID](./available_tasks/retrieval.md#treccovid)                                                                | Retrieval          | text         | eng                               |
    | [Touche2020](./available_tasks/retrieval.md#touche2020)                                                              | Retrieval          | text         | eng                               |
    | [ToxicConversationsClassification](./available_tasks/classification.md#toxicconversationsclassification)             | Classification     | text         | eng                               |
    | [TweetSentimentExtractionClassification](./available_tasks/classification.md#tweetsentimentextractionclassification) | Classification     | text         | eng                               |
    | [TwentyNewsgroupsClustering](./available_tasks/clustering.md#twentynewsgroupsclustering)                             | Clustering         | text         | eng                               |
    | [TwitterSemEval2015](./available_tasks/pairclassification.md#twittersemeval2015)                                     | PairClassification | text         | eng                               |
    | [TwitterURLCorpus](./available_tasks/pairclassification.md#twitterurlcorpus)                                         | PairClassification | text         | eng                               |
    | [MSMARCO](./available_tasks/retrieval.md#msmarco)                                                                    | Retrieval          | text         | eng                               |
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification)         | Classification     | text         | deu, eng, jpn                     |
    | [STS17](./available_tasks/sts.md#sts17)                                                                              | STS                | text         | ara, deu, eng, fra, ita, ... (9)  |
    | [STS22](./available_tasks/sts.md#sts22)                                                                              | STS                | text         | ara, cmn, deu, eng, fra, ... (10) |


??? quote "Citation"


    ```bibtex

    @article{muennighoff2022mteb,
      author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
      doi = {10.48550/ARXIV.2210.07316},
      journal = {arXiv preprint arXiv:2210.07316},
      publisher = {arXiv},
      title = {MTEB: Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2210.07316},
      year = {2022},
    }

    ```




####  MTEB(eng, v2)

The new English Massive Text Embedding Benchmark.
This benchmark was created to account for the fact that many models have now been finetuned
to tasks in the original MTEB, and contains tasks that are not as frequently used for model training.
This way the new benchmark and leaderboard can give our users a more realistic expectation of models' generalization performance.

The original MTEB leaderboard is available under the [MTEB(eng, v1)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v1%29) tab.


??? info "Tasks"

    | name                                                                                                                 | type               | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [ArguAna](./available_tasks/retrieval.md#arguana)                                                                    | Retrieval          | text         | eng                               |
    | [ArXivHierarchicalClusteringP2P](./available_tasks/clustering.md#arxivhierarchicalclusteringp2p)                     | Clustering         | text         | eng                               |
    | [ArXivHierarchicalClusteringS2S](./available_tasks/clustering.md#arxivhierarchicalclusterings2s)                     | Clustering         | text         | eng                               |
    | [AskUbuntuDupQuestions](./available_tasks/reranking.md#askubuntudupquestions)                                        | Reranking          | text         | eng                               |
    | [BIOSSES](./available_tasks/sts.md#biosses)                                                                          | STS                | text         | eng                               |
    | [Banking77Classification](./available_tasks/classification.md#banking77classification)                               | Classification     | text         | eng                               |
    | [BiorxivClusteringP2P.v2](./available_tasks/clustering.md#biorxivclusteringp2p.v2)                                   | Clustering         | text         | eng                               |
    | [CQADupstackGamingRetrieval](./available_tasks/retrieval.md#cqadupstackgamingretrieval)                              | Retrieval          | text         | eng                               |
    | [CQADupstackUnixRetrieval](./available_tasks/retrieval.md#cqadupstackunixretrieval)                                  | Retrieval          | text         | eng                               |
    | [ClimateFEVERHardNegatives](./available_tasks/retrieval.md#climatefeverhardnegatives)                                | Retrieval          | text         | eng                               |
    | [FEVERHardNegatives](./available_tasks/retrieval.md#feverhardnegatives)                                              | Retrieval          | text         | eng                               |
    | [FiQA2018](./available_tasks/retrieval.md#fiqa2018)                                                                  | Retrieval          | text         | eng                               |
    | [HotpotQAHardNegatives](./available_tasks/retrieval.md#hotpotqahardnegatives)                                        | Retrieval          | text         | eng                               |
    | [ImdbClassification](./available_tasks/classification.md#imdbclassification)                                         | Classification     | text         | eng                               |
    | [MTOPDomainClassification](./available_tasks/classification.md#mtopdomainclassification)                             | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                       | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)                   | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MedrxivClusteringP2P.v2](./available_tasks/clustering.md#medrxivclusteringp2p.v2)                                   | Clustering         | text         | eng                               |
    | [MedrxivClusteringS2S.v2](./available_tasks/clustering.md#medrxivclusterings2s.v2)                                   | Clustering         | text         | eng                               |
    | [MindSmallReranking](./available_tasks/reranking.md#mindsmallreranking)                                              | Reranking          | text         | eng                               |
    | [SCIDOCS](./available_tasks/retrieval.md#scidocs)                                                                    | Retrieval          | text         | eng                               |
    | [SICK-R](./available_tasks/sts.md#sick-r)                                                                            | STS                | text         | eng                               |
    | [STS12](./available_tasks/sts.md#sts12)                                                                              | STS                | text         | eng                               |
    | [STS13](./available_tasks/sts.md#sts13)                                                                              | STS                | text         | eng                               |
    | [STS14](./available_tasks/sts.md#sts14)                                                                              | STS                | text         | eng                               |
    | [STS15](./available_tasks/sts.md#sts15)                                                                              | STS                | text         | eng                               |
    | [STSBenchmark](./available_tasks/sts.md#stsbenchmark)                                                                | STS                | text         | eng                               |
    | [SprintDuplicateQuestions](./available_tasks/pairclassification.md#sprintduplicatequestions)                         | PairClassification | text         | eng                               |
    | [StackExchangeClustering.v2](./available_tasks/clustering.md#stackexchangeclustering.v2)                             | Clustering         | text         | eng                               |
    | [StackExchangeClusteringP2P.v2](./available_tasks/clustering.md#stackexchangeclusteringp2p.v2)                       | Clustering         | text         | eng                               |
    | [TRECCOVID](./available_tasks/retrieval.md#treccovid)                                                                | Retrieval          | text         | eng                               |
    | [Touche2020Retrieval.v3](./available_tasks/retrieval.md#touche2020retrieval.v3)                                      | Retrieval          | text         | eng                               |
    | [ToxicConversationsClassification](./available_tasks/classification.md#toxicconversationsclassification)             | Classification     | text         | eng                               |
    | [TweetSentimentExtractionClassification](./available_tasks/classification.md#tweetsentimentextractionclassification) | Classification     | text         | eng                               |
    | [TwentyNewsgroupsClustering.v2](./available_tasks/clustering.md#twentynewsgroupsclustering.v2)                       | Clustering         | text         | eng                               |
    | [TwitterSemEval2015](./available_tasks/pairclassification.md#twittersemeval2015)                                     | PairClassification | text         | eng                               |
    | [TwitterURLCorpus](./available_tasks/pairclassification.md#twitterurlcorpus)                                         | PairClassification | text         | eng                               |
    | [SummEvalSummarization.v2](./available_tasks/summarization.md#summevalsummarization.v2)                              | Summarization      | text         | eng                               |
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification)         | Classification     | text         | deu, eng, jpn                     |
    | [STS17](./available_tasks/sts.md#sts17)                                                                              | STS                | text         | ara, deu, eng, fra, ita, ... (9)  |
    | [STS22.v2](./available_tasks/sts.md#sts22.v2)                                                                        | STS                | text         | ara, cmn, deu, eng, fra, ... (10) |


??? quote "Citation"


    ```bibtex
    @article{enevoldsen2025mmtebmassivemultilingualtext,
      author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
      doi = {10.48550/arXiv.2502.13595},
      journal = {arXiv preprint arXiv:2502.13595},
      publisher = {arXiv},
      title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2502.13595},
      year = {2025},
    }
    ```




####  MTEB(fas, v1)

The Persian Massive Text Embedding Benchmark (FaMTEB) is a comprehensive benchmark for Persian text embeddings covering 7 tasks and 60+ datasets.

[Learn more →](https://arxiv.org/abs/2502.11571)



??? info "Tasks"

    | name                                                                                                                             | type               | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [PersianFoodSentimentClassification](./available_tasks/classification.md#persianfoodsentimentclassification)                     | Classification     | text         | fas                               |
    | [SynPerChatbotConvSAClassification](./available_tasks/classification.md#synperchatbotconvsaclassification)                       | Classification     | text         | fas                               |
    | [SynPerChatbotConvSAToneChatbotClassification](./available_tasks/classification.md#synperchatbotconvsatonechatbotclassification) | Classification     | text         | fas                               |
    | [SynPerChatbotConvSAToneUserClassification](./available_tasks/classification.md#synperchatbotconvsatoneuserclassification)       | Classification     | text         | fas                               |
    | [SynPerChatbotSatisfactionLevelClassification](./available_tasks/classification.md#synperchatbotsatisfactionlevelclassification) | Classification     | text         | fas                               |
    | [SynPerChatbotRAGToneChatbotClassification](./available_tasks/classification.md#synperchatbotragtonechatbotclassification)       | Classification     | text         | fas                               |
    | [SynPerChatbotRAGToneUserClassification](./available_tasks/classification.md#synperchatbotragtoneuserclassification)             | Classification     | text         | fas                               |
    | [SynPerChatbotToneChatbotClassification](./available_tasks/classification.md#synperchatbottonechatbotclassification)             | Classification     | text         | fas                               |
    | [SynPerChatbotToneUserClassification](./available_tasks/classification.md#synperchatbottoneuserclassification)                   | Classification     | text         | fas                               |
    | [SynPerTextToneClassification](./available_tasks/classification.md#synpertexttoneclassification)                                 | Classification     | text         | fas                               |
    | [SIDClassification](./available_tasks/classification.md#sidclassification)                                                       | Classification     | text         | fas                               |
    | [DeepSentiPers](./available_tasks/classification.md#deepsentipers)                                                               | Classification     | text         | fas                               |
    | [PersianTextEmotion](./available_tasks/classification.md#persiantextemotion)                                                     | Classification     | text         | fas                               |
    | [SentimentDKSF](./available_tasks/classification.md#sentimentdksf)                                                               | Classification     | text         | fas                               |
    | [NLPTwitterAnalysisClassification](./available_tasks/classification.md#nlptwitteranalysisclassification)                         | Classification     | text         | fas                               |
    | [DigikalamagClassification](./available_tasks/classification.md#digikalamagclassification)                                       | Classification     | text         | fas                               |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                                   | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)                               | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [BeytooteClustering](./available_tasks/clustering.md#beytooteclustering)                                                         | Clustering         | text         | fas                               |
    | [DigikalamagClustering](./available_tasks/clustering.md#digikalamagclustering)                                                   | Clustering         | text         | fas                               |
    | [HamshahriClustring](./available_tasks/clustering.md#hamshahriclustring)                                                         | Clustering         | text         | fas                               |
    | [NLPTwitterAnalysisClustering](./available_tasks/clustering.md#nlptwitteranalysisclustering)                                     | Clustering         | text         | fas                               |
    | [SIDClustring](./available_tasks/clustering.md#sidclustring)                                                                     | Clustering         | text         | fas                               |
    | [FarsTail](./available_tasks/pairclassification.md#farstail)                                                                     | PairClassification | text         | fas                               |
    | [CExaPPC](./available_tasks/pairclassification.md#cexappc)                                                                       | PairClassification | text         | fas                               |
    | [SynPerChatbotRAGFAQPC](./available_tasks/pairclassification.md#synperchatbotragfaqpc)                                           | PairClassification | text         | fas                               |
    | [FarsiParaphraseDetection](./available_tasks/pairclassification.md#farsiparaphrasedetection)                                     | PairClassification | text         | fas                               |
    | [SynPerTextKeywordsPC](./available_tasks/pairclassification.md#synpertextkeywordspc)                                             | PairClassification | text         | fas                               |
    | [SynPerQAPC](./available_tasks/pairclassification.md#synperqapc)                                                                 | PairClassification | text         | fas                               |
    | [ParsinluEntail](./available_tasks/pairclassification.md#parsinluentail)                                                         | PairClassification | text         | fas                               |
    | [ParsinluQueryParaphPC](./available_tasks/pairclassification.md#parsinluqueryparaphpc)                                           | PairClassification | text         | fas                               |
    | [MIRACLReranking](./available_tasks/reranking.md#miraclreranking)                                                                | Reranking          | text         | ara, ben, deu, eng, fas, ... (18) |
    | [WikipediaRerankingMultilingual](./available_tasks/reranking.md#wikipediarerankingmultilingual)                                  | Reranking          | text         | ben, bul, ces, dan, deu, ... (18) |
    | [SynPerQARetrieval](./available_tasks/retrieval.md#synperqaretrieval)                                                            | Retrieval          | text         | fas                               |
    | [SynPerChatbotTopicsRetrieval](./available_tasks/retrieval.md#synperchatbottopicsretrieval)                                      | Retrieval          | text         | fas                               |
    | [SynPerChatbotRAGTopicsRetrieval](./available_tasks/retrieval.md#synperchatbotragtopicsretrieval)                                | Retrieval          | text         | fas                               |
    | [SynPerChatbotRAGFAQRetrieval](./available_tasks/retrieval.md#synperchatbotragfaqretrieval)                                      | Retrieval          | text         | fas                               |
    | [PersianWebDocumentRetrieval](./available_tasks/retrieval.md#persianwebdocumentretrieval)                                        | Retrieval          | text         | fas                               |
    | [WikipediaRetrievalMultilingual](./available_tasks/retrieval.md#wikipediaretrievalmultilingual)                                  | Retrieval          | text         | ben, bul, ces, dan, deu, ... (16) |
    | [MIRACLRetrieval](./available_tasks/retrieval.md#miraclretrieval)                                                                | Retrieval          | text         | ara, ben, deu, eng, fas, ... (18) |
    | [ClimateFEVER-Fa](./available_tasks/retrieval.md#climatefever-fa)                                                                | Retrieval          | text         | fas                               |
    | [DBPedia-Fa](./available_tasks/retrieval.md#dbpedia-fa)                                                                          | Retrieval          | text         | fas                               |
    | [HotpotQA-Fa](./available_tasks/retrieval.md#hotpotqa-fa)                                                                        | Retrieval          | text         | fas                               |
    | [MSMARCO-Fa](./available_tasks/retrieval.md#msmarco-fa)                                                                          | Retrieval          | text         | fas                               |
    | [NQ-Fa](./available_tasks/retrieval.md#nq-fa)                                                                                    | Retrieval          | text         | fas                               |
    | [ArguAna-Fa](./available_tasks/retrieval.md#arguana-fa)                                                                          | Retrieval          | text         | fas                               |
    | [CQADupstackRetrieval-Fa](./available_tasks/retrieval.md#cqadupstackretrieval-fa)                                                | Retrieval          | text         | fas                               |
    | [FiQA2018-Fa](./available_tasks/retrieval.md#fiqa2018-fa)                                                                        | Retrieval          | text         | fas                               |
    | [NFCorpus-Fa](./available_tasks/retrieval.md#nfcorpus-fa)                                                                        | Retrieval          | text         | fas                               |
    | [QuoraRetrieval-Fa](./available_tasks/retrieval.md#quoraretrieval-fa)                                                            | Retrieval          | text         | fas                               |
    | [SCIDOCS-Fa](./available_tasks/retrieval.md#scidocs-fa)                                                                          | Retrieval          | text         | fas                               |
    | [SciFact-Fa](./available_tasks/retrieval.md#scifact-fa)                                                                          | Retrieval          | text         | fas                               |
    | [TRECCOVID-Fa](./available_tasks/retrieval.md#treccovid-fa)                                                                      | Retrieval          | text         | fas                               |
    | [Touche2020-Fa](./available_tasks/retrieval.md#touche2020-fa)                                                                    | Retrieval          | text         | fas                               |
    | [Farsick](./available_tasks/sts.md#farsick)                                                                                      | STS                | text         | fas                               |
    | [SynPerSTS](./available_tasks/sts.md#synpersts)                                                                                  | STS                | text         | fas                               |
    | [Query2Query](./available_tasks/sts.md#query2query)                                                                              | STS                | text         | fas                               |
    | [SAMSumFa](./available_tasks/bitextmining.md#samsumfa)                                                                           | BitextMining       | text         | fas                               |
    | [SynPerChatbotSumSRetrieval](./available_tasks/bitextmining.md#synperchatbotsumsretrieval)                                       | BitextMining       | text         | fas                               |
    | [SynPerChatbotRAGSumSRetrieval](./available_tasks/bitextmining.md#synperchatbotragsumsretrieval)                                 | BitextMining       | text         | fas                               |


??? quote "Citation"


    ```bibtex

    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }

    ```




####  MTEB(fas, v2)

The Persian Massive Text Embedding Benchmark (FaMTEB) is a comprehensive benchmark for Persian text embeddings covering 7 tasks and 50+ datasets. In version 2, we have optimized large datasets to make them more manageable and accessible, removed low-quality datasets, and added higher-quality data to improve the overall benchmark. For more details on the improvements, see the main PR comment: [main PR](https://github.com/embeddings-benchmark/mteb/pull/3157).

[Learn more →](https://arxiv.org/abs/2502.11571)



??? info "Tasks"

    | name                                                                                                                             | type               | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [PersianFoodSentimentClassification](./available_tasks/classification.md#persianfoodsentimentclassification)                     | Classification     | text         | fas                               |
    | [SynPerChatbotConvSAClassification](./available_tasks/classification.md#synperchatbotconvsaclassification)                       | Classification     | text         | fas                               |
    | [SynPerChatbotConvSAToneChatbotClassification](./available_tasks/classification.md#synperchatbotconvsatonechatbotclassification) | Classification     | text         | fas                               |
    | [SynPerChatbotConvSAToneUserClassification](./available_tasks/classification.md#synperchatbotconvsatoneuserclassification)       | Classification     | text         | fas                               |
    | [SynPerChatbotSatisfactionLevelClassification](./available_tasks/classification.md#synperchatbotsatisfactionlevelclassification) | Classification     | text         | fas                               |
    | [SynPerTextToneClassification.v3](./available_tasks/classification.md#synpertexttoneclassification.v3)                           | Classification     | text         | fas                               |
    | [SIDClassification.v2](./available_tasks/classification.md#sidclassification.v2)                                                 | Classification     | text         | fas                               |
    | [DeepSentiPers.v2](./available_tasks/classification.md#deepsentipers.v2)                                                         | Classification     | text         | fas                               |
    | [PersianTextEmotion.v2](./available_tasks/classification.md#persiantextemotion.v2)                                               | Classification     | text         | fas                               |
    | [NLPTwitterAnalysisClassification.v2](./available_tasks/classification.md#nlptwitteranalysisclassification.v2)                   | Classification     | text         | fas                               |
    | [DigikalamagClassification](./available_tasks/classification.md#digikalamagclassification)                                       | Classification     | text         | fas                               |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)                                   | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)                               | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [StyleClassification](./available_tasks/classification.md#styleclassification)                                                   | Classification     | text         | fas                               |
    | [PerShopDomainClassification](./available_tasks/classification.md#pershopdomainclassification)                                   | Classification     | text         | fas                               |
    | [PerShopIntentClassification](./available_tasks/classification.md#pershopintentclassification)                                   | Classification     | text         | fas                               |
    | [BeytooteClustering](./available_tasks/clustering.md#beytooteclustering)                                                         | Clustering         | text         | fas                               |
    | [DigikalamagClustering](./available_tasks/clustering.md#digikalamagclustering)                                                   | Clustering         | text         | fas                               |
    | [HamshahriClustring](./available_tasks/clustering.md#hamshahriclustring)                                                         | Clustering         | text         | fas                               |
    | [NLPTwitterAnalysisClustering](./available_tasks/clustering.md#nlptwitteranalysisclustering)                                     | Clustering         | text         | fas                               |
    | [SIDClustring](./available_tasks/clustering.md#sidclustring)                                                                     | Clustering         | text         | fas                               |
    | [FarsTail](./available_tasks/pairclassification.md#farstail)                                                                     | PairClassification | text         | fas                               |
    | [SynPerChatbotRAGFAQPC](./available_tasks/pairclassification.md#synperchatbotragfaqpc)                                           | PairClassification | text         | fas                               |
    | [FarsiParaphraseDetection](./available_tasks/pairclassification.md#farsiparaphrasedetection)                                     | PairClassification | text         | fas                               |
    | [SynPerTextKeywordsPC](./available_tasks/pairclassification.md#synpertextkeywordspc)                                             | PairClassification | text         | fas                               |
    | [SynPerQAPC](./available_tasks/pairclassification.md#synperqapc)                                                                 | PairClassification | text         | fas                               |
    | [ParsinluEntail](./available_tasks/pairclassification.md#parsinluentail)                                                         | PairClassification | text         | fas                               |
    | [ParsinluQueryParaphPC](./available_tasks/pairclassification.md#parsinluqueryparaphpc)                                           | PairClassification | text         | fas                               |
    | [MIRACLReranking](./available_tasks/reranking.md#miraclreranking)                                                                | Reranking          | text         | ara, ben, deu, eng, fas, ... (18) |
    | [WikipediaRerankingMultilingual](./available_tasks/reranking.md#wikipediarerankingmultilingual)                                  | Reranking          | text         | ben, bul, ces, dan, deu, ... (18) |
    | [SynPerQARetrieval](./available_tasks/retrieval.md#synperqaretrieval)                                                            | Retrieval          | text         | fas                               |
    | [SynPerChatbotRAGFAQRetrieval](./available_tasks/retrieval.md#synperchatbotragfaqretrieval)                                      | Retrieval          | text         | fas                               |
    | [PersianWebDocumentRetrieval](./available_tasks/retrieval.md#persianwebdocumentretrieval)                                        | Retrieval          | text         | fas                               |
    | [WikipediaRetrievalMultilingual](./available_tasks/retrieval.md#wikipediaretrievalmultilingual)                                  | Retrieval          | text         | ben, bul, ces, dan, deu, ... (16) |
    | [MIRACLRetrievalHardNegatives](./available_tasks/retrieval.md#miraclretrievalhardnegatives)                                      | Retrieval          | text         | ara, ben, deu, eng, fas, ... (18) |
    | [HotpotQA-FaHardNegatives](./available_tasks/retrieval.md#hotpotqa-fahardnegatives)                                              | Retrieval          | text         | fas                               |
    | [MSMARCO-FaHardNegatives](./available_tasks/retrieval.md#msmarco-fahardnegatives)                                                | Retrieval          | text         | fas                               |
    | [NQ-FaHardNegatives](./available_tasks/retrieval.md#nq-fahardnegatives)                                                          | Retrieval          | text         | fas                               |
    | [ArguAna-Fa.v2](./available_tasks/retrieval.md#arguana-fa.v2)                                                                    | Retrieval          | text         | fas                               |
    | [FiQA2018-Fa.v2](./available_tasks/retrieval.md#fiqa2018-fa.v2)                                                                  | Retrieval          | text         | fas                               |
    | [QuoraRetrieval-Fa.v2](./available_tasks/retrieval.md#quoraretrieval-fa.v2)                                                      | Retrieval          | text         | fas                               |
    | [SCIDOCS-Fa.v2](./available_tasks/retrieval.md#scidocs-fa.v2)                                                                    | Retrieval          | text         | fas                               |
    | [SciFact-Fa.v2](./available_tasks/retrieval.md#scifact-fa.v2)                                                                    | Retrieval          | text         | fas                               |
    | [TRECCOVID-Fa.v2](./available_tasks/retrieval.md#treccovid-fa.v2)                                                                | Retrieval          | text         | fas                               |
    | [FEVER-FaHardNegatives](./available_tasks/retrieval.md#fever-fahardnegatives)                                                    | Retrieval          | text         | fas                               |
    | [NeuCLIR2023RetrievalHardNegatives](./available_tasks/retrieval.md#neuclir2023retrievalhardnegatives)                            | Retrieval          | text         | fas, rus, zho                     |
    | [WebFAQRetrieval](./available_tasks/retrieval.md#webfaqretrieval)                                                                | Retrieval          | text         | ara, aze, ben, bul, cat, ... (51) |
    | [Farsick](./available_tasks/sts.md#farsick)                                                                                      | STS                | text         | fas                               |
    | [SynPerSTS](./available_tasks/sts.md#synpersts)                                                                                  | STS                | text         | fas                               |
    | [SAMSumFa](./available_tasks/bitextmining.md#samsumfa)                                                                           | BitextMining       | text         | fas                               |
    | [SynPerChatbotSumSRetrieval](./available_tasks/bitextmining.md#synperchatbotsumsretrieval)                                       | BitextMining       | text         | fas                               |
    | [SynPerChatbotRAGSumSRetrieval](./available_tasks/bitextmining.md#synperchatbotragsumsretrieval)                                 | BitextMining       | text         | fas                               |


??? quote "Citation"


    ```bibtex

    @article{zinvandi2025famteb,
      author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
      journal = {arXiv preprint arXiv:2502.11571},
      title = {Famteb: Massive text embedding benchmark in persian language},
      year = {2025},
    }

    ```




####  MTEB(fra, v1)

MTEB-French, a French expansion of the original benchmark with high-quality native French datasets.

[Learn more →](https://arxiv.org/abs/2405.20468)



??? info "Tasks"

    | name                                                                                               | type               | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [AmazonReviewsClassification](./available_tasks/classification.md#amazonreviewsclassification)     | Classification     | text         | cmn, deu, eng, fra, jpn, ... (6)  |
    | [MasakhaNEWSClassification](./available_tasks/classification.md#masakhanewsclassification)         | Classification     | text         | amh, eng, fra, hau, ibo, ... (16) |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)     | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification) | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MTOPDomainClassification](./available_tasks/classification.md#mtopdomainclassification)           | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [MTOPIntentClassification](./available_tasks/classification.md#mtopintentclassification)           | Classification     | text         | deu, eng, fra, hin, spa, ... (6)  |
    | [AlloProfClusteringP2P](./available_tasks/clustering.md#alloprofclusteringp2p)                     | Clustering         | text         | fra                               |
    | [AlloProfClusteringS2S](./available_tasks/clustering.md#alloprofclusterings2s)                     | Clustering         | text         | fra                               |
    | [HALClusteringS2S](./available_tasks/clustering.md#halclusterings2s)                               | Clustering         | text         | fra                               |
    | [MasakhaNEWSClusteringP2P](./available_tasks/clustering.md#masakhanewsclusteringp2p)               | Clustering         | text         | amh, eng, fra, hau, ibo, ... (16) |
    | [MasakhaNEWSClusteringS2S](./available_tasks/clustering.md#masakhanewsclusterings2s)               | Clustering         | text         | amh, eng, fra, hau, ibo, ... (16) |
    | [MLSUMClusteringP2P](./available_tasks/clustering.md#mlsumclusteringp2p)                           | Clustering         | text         | deu, fra, rus, spa                |
    | [MLSUMClusteringS2S](./available_tasks/clustering.md#mlsumclusterings2s)                           | Clustering         | text         | deu, fra, rus, spa                |
    | [PawsXPairClassification](./available_tasks/pairclassification.md#pawsxpairclassification)         | PairClassification | text         | cmn, deu, eng, fra, jpn, ... (7)  |
    | [AlloprofReranking](./available_tasks/reranking.md#alloprofreranking)                              | Reranking          | text         | fra                               |
    | [SyntecReranking](./available_tasks/reranking.md#syntecreranking)                                  | Reranking          | text         | fra                               |
    | [AlloprofRetrieval](./available_tasks/retrieval.md#alloprofretrieval)                              | Retrieval          | text         | fra                               |
    | [BSARDRetrieval](./available_tasks/retrieval.md#bsardretrieval)                                    | Retrieval          | text         | fra                               |
    | [MintakaRetrieval](./available_tasks/retrieval.md#mintakaretrieval)                                | Retrieval          | text         | ara, deu, fra, hin, ita, ... (8)  |
    | [SyntecRetrieval](./available_tasks/retrieval.md#syntecretrieval)                                  | Retrieval          | text         | fra                               |
    | [XPQARetrieval](./available_tasks/retrieval.md#xpqaretrieval)                                      | Retrieval          | text         | ara, cmn, deu, eng, fra, ... (13) |
    | [SICKFr](./available_tasks/sts.md#sickfr)                                                          | STS                | text         | fra                               |
    | [STSBenchmarkMultilingualSTS](./available_tasks/sts.md#stsbenchmarkmultilingualsts)                | STS                | text         | cmn, deu, eng, fra, ita, ... (10) |
    | [SummEvalFr](./available_tasks/summarization.md#summevalfr)                                        | Summarization      | text         | fra                               |
    | [STS22](./available_tasks/sts.md#sts22)                                                            | STS                | text         | ara, cmn, deu, eng, fra, ... (10) |


??? quote "Citation"


    ```bibtex

    @misc{ciancone2024mtebfrenchresourcesfrenchsentence,
      archiveprefix = {arXiv},
      author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      eprint = {2405.20468},
      primaryclass = {cs.CL},
      title = {MTEB-French: Resources for French Sentence Embedding Evaluation and Analysis},
      url = {https://arxiv.org/abs/2405.20468},
      year = {2024},
    }

    ```




####  MTEB(jpn, v1)

JMTEB is a benchmark for evaluating Japanese text embedding models.

[Learn more →](https://github.com/sbintuitions/JMTEB)



??? info "Tasks"

    | name                                                                                                         | type               | modalities   | languages                         |
    |:-------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [LivedoorNewsClustering.v2](./available_tasks/clustering.md#livedoornewsclustering.v2)                       | Clustering         | text         | jpn                               |
    | [MewsC16JaClustering](./available_tasks/clustering.md#mewsc16jaclustering)                                   | Clustering         | text         | jpn                               |
    | [AmazonReviewsClassification](./available_tasks/classification.md#amazonreviewsclassification)               | Classification     | text         | cmn, deu, eng, fra, jpn, ... (6)  |
    | [AmazonCounterfactualClassification](./available_tasks/classification.md#amazoncounterfactualclassification) | Classification     | text         | deu, eng, jpn                     |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)               | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)           | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [JSTS](./available_tasks/sts.md#jsts)                                                                        | STS                | text         | jpn                               |
    | [JSICK](./available_tasks/sts.md#jsick)                                                                      | STS                | text         | jpn                               |
    | [PawsXPairClassification](./available_tasks/pairclassification.md#pawsxpairclassification)                   | PairClassification | text         | cmn, deu, eng, fra, jpn, ... (7)  |
    | [JaqketRetrieval](./available_tasks/retrieval.md#jaqketretrieval)                                            | Retrieval          | text         | jpn                               |
    | [MrTidyRetrieval](./available_tasks/retrieval.md#mrtidyretrieval)                                            | Retrieval          | text         | ara, ben, eng, fin, ind, ... (11) |
    | [JaGovFaqsRetrieval](./available_tasks/retrieval.md#jagovfaqsretrieval)                                      | Retrieval          | text         | jpn                               |
    | [NLPJournalTitleAbsRetrieval](./available_tasks/retrieval.md#nlpjournaltitleabsretrieval)                    | Retrieval          | text         | jpn                               |
    | [NLPJournalAbsIntroRetrieval](./available_tasks/retrieval.md#nlpjournalabsintroretrieval)                    | Retrieval          | text         | jpn                               |
    | [NLPJournalTitleIntroRetrieval](./available_tasks/retrieval.md#nlpjournaltitleintroretrieval)                | Retrieval          | text         | jpn                               |
    | [ESCIReranking](./available_tasks/reranking.md#escireranking)                                                | Reranking          | text         | eng, jpn, spa                     |


####  MTEB(kor, v1)

A benchmark and leaderboard for evaluation of text embedding in Korean.

??? info "Tasks"

    | name                                                              | type           | modalities   | languages                         |
    |:------------------------------------------------------------------|:---------------|:-------------|:----------------------------------|
    | [KLUE-TC](./available_tasks/classification.md#klue-tc)            | Classification | text         | kor                               |
    | [MIRACLReranking](./available_tasks/reranking.md#miraclreranking) | Reranking      | text         | ara, ben, deu, eng, fas, ... (18) |
    | [MIRACLRetrieval](./available_tasks/retrieval.md#miraclretrieval) | Retrieval      | text         | ara, ben, deu, eng, fas, ... (18) |
    | [Ko-StrategyQA](./available_tasks/retrieval.md#ko-strategyqa)     | Retrieval      | text         | kor                               |
    | [KLUE-STS](./available_tasks/sts.md#klue-sts)                     | STS            | text         | kor                               |
    | [KorSTS](./available_tasks/sts.md#korsts)                         | STS            | text         | kor                               |


####  MTEB(pol, v1)

Polish Massive Text Embedding Benchmark (PL-MTEB), a comprehensive benchmark for text embeddings in Polish. The PL-MTEB consists of 28 diverse NLP
tasks from 5 task types. With tasks adapted based on previously used datasets by the Polish
NLP community. In addition, a new PLSC (Polish Library of Science Corpus) dataset was created
consisting of titles and abstracts of scientific publications in Polish, which was used as the basis for
two novel clustering tasks.

[Learn more →](https://arxiv.org/abs/2405.10138)



??? info "Tasks"

    | name                                                                                               | type               | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------------|:-------------------|:-------------|:----------------------------------|
    | [AllegroReviews](./available_tasks/classification.md#allegroreviews)                               | Classification     | text         | pol                               |
    | [CBD](./available_tasks/classification.md#cbd)                                                     | Classification     | text         | pol                               |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)     | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification) | Classification     | text         | afr, amh, ara, aze, ben, ... (50) |
    | [PolEmo2.0-IN](./available_tasks/classification.md#polemo2.0-in)                                   | Classification     | text         | pol                               |
    | [PolEmo2.0-OUT](./available_tasks/classification.md#polemo2.0-out)                                 | Classification     | text         | pol                               |
    | [PAC](./available_tasks/classification.md#pac)                                                     | Classification     | text         | pol                               |
    | [EightTagsClustering](./available_tasks/clustering.md#eighttagsclustering)                         | Clustering         | text         | pol                               |
    | [PlscClusteringS2S](./available_tasks/clustering.md#plscclusterings2s)                             | Clustering         | text         | pol                               |
    | [PlscClusteringP2P](./available_tasks/clustering.md#plscclusteringp2p)                             | Clustering         | text         | pol                               |
    | [CDSC-E](./available_tasks/pairclassification.md#cdsc-e)                                           | PairClassification | text         | pol                               |
    | [PpcPC](./available_tasks/pairclassification.md#ppcpc)                                             | PairClassification | text         | pol                               |
    | [PSC](./available_tasks/pairclassification.md#psc)                                                 | PairClassification | text         | pol                               |
    | [SICK-E-PL](./available_tasks/pairclassification.md#sick-e-pl)                                     | PairClassification | text         | pol                               |
    | [CDSC-R](./available_tasks/sts.md#cdsc-r)                                                          | STS                | text         | pol                               |
    | [SICK-R-PL](./available_tasks/sts.md#sick-r-pl)                                                    | STS                | text         | pol                               |
    | [STS22](./available_tasks/sts.md#sts22)                                                            | STS                | text         | ara, cmn, deu, eng, fra, ... (10) |


??? quote "Citation"


    ```bibtex

    @article{poswiata2024plmteb,
      author = {Rafał Poświata and Sławomir Dadas and Michał Perełkiewicz},
      journal = {arXiv preprint arXiv:2405.10138},
      title = {PL-MTEB: Polish Massive Text Embedding Benchmark},
      year = {2024},
    }

    ```




####  MTEB(rus, v1)

A Russian version of the Massive Text Embedding Benchmark with a number of novel Russian tasks in all task categories of the original MTEB.

[Learn more →](https://aclanthology.org/2023.eacl-main.148/)



??? info "Tasks"

    | name                                                                                                         | type                     | modalities   | languages                         |
    |:-------------------------------------------------------------------------------------------------------------|:-------------------------|:-------------|:----------------------------------|
    | [GeoreviewClassification](./available_tasks/classification.md#georeviewclassification)                       | Classification           | text         | rus                               |
    | [HeadlineClassification](./available_tasks/classification.md#headlineclassification)                         | Classification           | text         | rus                               |
    | [InappropriatenessClassification](./available_tasks/classification.md#inappropriatenessclassification)       | Classification           | text         | rus                               |
    | [KinopoiskClassification](./available_tasks/classification.md#kinopoiskclassification)                       | Classification           | text         | rus                               |
    | [MassiveIntentClassification](./available_tasks/classification.md#massiveintentclassification)               | Classification           | text         | afr, amh, ara, aze, ben, ... (50) |
    | [MassiveScenarioClassification](./available_tasks/classification.md#massivescenarioclassification)           | Classification           | text         | afr, amh, ara, aze, ben, ... (50) |
    | [RuReviewsClassification](./available_tasks/classification.md#rureviewsclassification)                       | Classification           | text         | rus                               |
    | [RuSciBenchGRNTIClassification](./available_tasks/classification.md#ruscibenchgrnticlassification)           | Classification           | text         | rus                               |
    | [RuSciBenchOECDClassification](./available_tasks/classification.md#ruscibenchoecdclassification)             | Classification           | text         | rus                               |
    | [GeoreviewClusteringP2P](./available_tasks/clustering.md#georeviewclusteringp2p)                             | Clustering               | text         | rus                               |
    | [RuSciBenchGRNTIClusteringP2P](./available_tasks/clustering.md#ruscibenchgrnticlusteringp2p)                 | Clustering               | text         | rus                               |
    | [RuSciBenchOECDClusteringP2P](./available_tasks/clustering.md#ruscibenchoecdclusteringp2p)                   | Clustering               | text         | rus                               |
    | [CEDRClassification](./available_tasks/multilabelclassification.md#cedrclassification)                       | MultilabelClassification | text         | rus                               |
    | [SensitiveTopicsClassification](./available_tasks/multilabelclassification.md#sensitivetopicsclassification) | MultilabelClassification | text         | rus                               |
    | [TERRa](./available_tasks/pairclassification.md#terra)                                                       | PairClassification       | text         | rus                               |
    | [MIRACLReranking](./available_tasks/reranking.md#miraclreranking)                                            | Reranking                | text         | ara, ben, deu, eng, fas, ... (18) |
    | [RuBQReranking](./available_tasks/reranking.md#rubqreranking)                                                | Reranking                | text         | rus                               |
    | [MIRACLRetrieval](./available_tasks/retrieval.md#miraclretrieval)                                            | Retrieval                | text         | ara, ben, deu, eng, fas, ... (18) |
    | [RiaNewsRetrieval](./available_tasks/retrieval.md#rianewsretrieval)                                          | Retrieval                | text         | rus                               |
    | [RuBQRetrieval](./available_tasks/retrieval.md#rubqretrieval)                                                | Retrieval                | text         | rus                               |
    | [RUParaPhraserSTS](./available_tasks/sts.md#ruparaphrasersts)                                                | STS                      | text         | rus                               |
    | [STS22](./available_tasks/sts.md#sts22)                                                                      | STS                      | text         | ara, cmn, deu, eng, fra, ... (10) |
    | [RuSTSBenchmarkSTS](./available_tasks/sts.md#rustsbenchmarksts)                                              | STS                      | text         | rus                               |


??? quote "Citation"


    ```bibtex

    @misc{snegirev2024russianfocusedembeddersexplorationrumteb,
      archiveprefix = {arXiv},
      author = {Artem Snegirev and Maria Tikhonova and Anna Maksimova and Alena Fenogenova and Alexander Abramov},
      eprint = {2408.12503},
      primaryclass = {cs.CL},
      title = {The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design},
      url = {https://arxiv.org/abs/2408.12503},
      year = {2024},
    }

    ```




####  NanoBEIR

A benchmark to evaluate with subsets of BEIR datasets to use less computational power

[Learn more →](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6)



??? info "Tasks"

    | name                                                                                  | type      | modalities   | languages   |
    |:--------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [NanoArguAnaRetrieval](./available_tasks/retrieval.md#nanoarguanaretrieval)           | Retrieval | text         | eng         |
    | [NanoClimateFeverRetrieval](./available_tasks/retrieval.md#nanoclimatefeverretrieval) | Retrieval | text         | eng         |
    | [NanoDBPediaRetrieval](./available_tasks/retrieval.md#nanodbpediaretrieval)           | Retrieval | text         | eng         |
    | [NanoFEVERRetrieval](./available_tasks/retrieval.md#nanofeverretrieval)               | Retrieval | text         | eng         |
    | [NanoFiQA2018Retrieval](./available_tasks/retrieval.md#nanofiqa2018retrieval)         | Retrieval | text         | eng         |
    | [NanoHotpotQARetrieval](./available_tasks/retrieval.md#nanohotpotqaretrieval)         | Retrieval | text         | eng         |
    | [NanoMSMARCORetrieval](./available_tasks/retrieval.md#nanomsmarcoretrieval)           | Retrieval | text         | eng         |
    | [NanoNFCorpusRetrieval](./available_tasks/retrieval.md#nanonfcorpusretrieval)         | Retrieval | text         | eng         |
    | [NanoNQRetrieval](./available_tasks/retrieval.md#nanonqretrieval)                     | Retrieval | text         | eng         |
    | [NanoQuoraRetrieval](./available_tasks/retrieval.md#nanoquoraretrieval)               | Retrieval | text         | eng         |
    | [NanoSCIDOCSRetrieval](./available_tasks/retrieval.md#nanoscidocsretrieval)           | Retrieval | text         | eng         |
    | [NanoSciFactRetrieval](./available_tasks/retrieval.md#nanoscifactretrieval)           | Retrieval | text         | eng         |
    | [NanoTouche2020Retrieval](./available_tasks/retrieval.md#nanotouche2020retrieval)     | Retrieval | text         | eng         |


####  R2MED

R2MED: First Reasoning-Driven Medical Retrieval Benchmark.
    R2MED is a high-quality, high-resolution information retrieval (IR) dataset designed for medical scenarios.
    It contains 876 queries with three retrieval tasks, five medical scenarios, and twelve body systems.


[Learn more →](https://r2med.github.io/)



??? info "Tasks"

    | name                                                                                          | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [R2MEDBiologyRetrieval](./available_tasks/retrieval.md#r2medbiologyretrieval)                 | Retrieval | text         | eng         |
    | [R2MEDBioinformaticsRetrieval](./available_tasks/retrieval.md#r2medbioinformaticsretrieval)   | Retrieval | text         | eng         |
    | [R2MEDMedicalSciencesRetrieval](./available_tasks/retrieval.md#r2medmedicalsciencesretrieval) | Retrieval | text         | eng         |
    | [R2MEDMedXpertQAExamRetrieval](./available_tasks/retrieval.md#r2medmedxpertqaexamretrieval)   | Retrieval | text         | eng         |
    | [R2MEDMedQADiagRetrieval](./available_tasks/retrieval.md#r2medmedqadiagretrieval)             | Retrieval | text         | eng         |
    | [R2MEDPMCTreatmentRetrieval](./available_tasks/retrieval.md#r2medpmctreatmentretrieval)       | Retrieval | text         | eng         |
    | [R2MEDPMCClinicalRetrieval](./available_tasks/retrieval.md#r2medpmcclinicalretrieval)         | Retrieval | text         | eng         |
    | [R2MEDIIYiClinicalRetrieval](./available_tasks/retrieval.md#r2mediiyiclinicalretrieval)       | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }

    ```




####  RAR-b

A benchmark to evaluate reasoning capabilities of retrievers.

[Learn more →](https://arxiv.org/abs/2404.06347)



??? info "Tasks"

    | name                                                                      | type      | modalities   | languages   |
    |:--------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [ARCChallenge](./available_tasks/retrieval.md#arcchallenge)               | Retrieval | text         | eng         |
    | [AlphaNLI](./available_tasks/retrieval.md#alphanli)                       | Retrieval | text         | eng         |
    | [HellaSwag](./available_tasks/retrieval.md#hellaswag)                     | Retrieval | text         | eng         |
    | [WinoGrande](./available_tasks/retrieval.md#winogrande)                   | Retrieval | text         | eng         |
    | [PIQA](./available_tasks/retrieval.md#piqa)                               | Retrieval | text         | eng         |
    | [SIQA](./available_tasks/retrieval.md#siqa)                               | Retrieval | text         | eng         |
    | [Quail](./available_tasks/retrieval.md#quail)                             | Retrieval | text         | eng         |
    | [SpartQA](./available_tasks/retrieval.md#spartqa)                         | Retrieval | text         | eng         |
    | [TempReasonL1](./available_tasks/retrieval.md#tempreasonl1)               | Retrieval | text         | eng         |
    | [TempReasonL2Pure](./available_tasks/retrieval.md#tempreasonl2pure)       | Retrieval | text         | eng         |
    | [TempReasonL2Fact](./available_tasks/retrieval.md#tempreasonl2fact)       | Retrieval | text         | eng         |
    | [TempReasonL2Context](./available_tasks/retrieval.md#tempreasonl2context) | Retrieval | text         | eng         |
    | [TempReasonL3Pure](./available_tasks/retrieval.md#tempreasonl3pure)       | Retrieval | text         | eng         |
    | [TempReasonL3Fact](./available_tasks/retrieval.md#tempreasonl3fact)       | Retrieval | text         | eng         |
    | [TempReasonL3Context](./available_tasks/retrieval.md#tempreasonl3context) | Retrieval | text         | eng         |
    | [RARbCode](./available_tasks/retrieval.md#rarbcode)                       | Retrieval | text         | eng         |
    | [RARbMath](./available_tasks/retrieval.md#rarbmath)                       | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex

    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Al Moubayed, Noura},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }

    ```




####  RTEB(Code, beta)

RTEB Code is a subset of RTEB containing retrieval tasks specifically focused on programming and code domains including algorithmic problems, data science tasks, code evaluation, SQL retrieval, and multilingual code retrieval. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                            | type      | modalities   | languages                   |
    |:--------------------------------------------------------------------------------|:----------|:-------------|:----------------------------|
    | [AppsRetrieval](./available_tasks/retrieval.md#appsretrieval)                   | Retrieval | text         | eng, python                 |
    | [DS1000Retrieval](./available_tasks/retrieval.md#ds1000retrieval)               | Retrieval | text         | eng, python                 |
    | [HumanEvalRetrieval](./available_tasks/retrieval.md#humanevalretrieval)         | Retrieval | text         | eng, python                 |
    | [MBPPRetrieval](./available_tasks/retrieval.md#mbppretrieval)                   | Retrieval | text         | eng, python                 |
    | [WikiSQLRetrieval](./available_tasks/retrieval.md#wikisqlretrieval)             | Retrieval | text         | eng, sql                    |
    | [FreshStackRetrieval](./available_tasks/retrieval.md#freshstackretrieval)       | Retrieval | text         | eng, go, javascript, python |
    | [Code1Retrieval](./available_tasks/retrieval.md#code1retrieval)                 | Retrieval | text         | eng                         |
    | [JapaneseCode1Retrieval](./available_tasks/retrieval.md#japanesecode1retrieval) | Retrieval | text         | jpn                         |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(Health, beta)

RTEB Healthcare is a subset of RTEB containing retrieval tasks specifically focused on healthcare and medical domains including medical Q&A, healthcare information retrieval, cross-lingual medical retrieval, and multilingual medical consultation. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                                      | type      | modalities   | languages     |
    |:------------------------------------------------------------------------------------------|:----------|:-------------|:--------------|
    | [ChatDoctorRetrieval](./available_tasks/retrieval.md#chatdoctorretrieval)                 | Retrieval | text         | eng           |
    | [CUREv1](./available_tasks/retrieval.md#curev1)                                           | Retrieval | text         | eng, fra, spa |
    | [EnglishHealthcare1Retrieval](./available_tasks/retrieval.md#englishhealthcare1retrieval) | Retrieval | text         | eng           |
    | [GermanHealthcare1Retrieval](./available_tasks/retrieval.md#germanhealthcare1retrieval)   | Retrieval | text         | deu           |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(Law, beta)

RTEB Legal is a subset of RTEB containing retrieval tasks specifically focused on legal domain including case documents, statutes, legal summarization, and multilingual legal Q&A. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                              | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [AILACasedocs](./available_tasks/retrieval.md#ailacasedocs)                       | Retrieval | text         | eng         |
    | [AILAStatutes](./available_tasks/retrieval.md#ailastatutes)                       | Retrieval | text         | eng         |
    | [LegalSummarization](./available_tasks/retrieval.md#legalsummarization)           | Retrieval | text         | eng         |
    | [LegalQuAD](./available_tasks/retrieval.md#legalquad)                             | Retrieval | text         | deu         |
    | [FrenchLegal1Retrieval](./available_tasks/retrieval.md#frenchlegal1retrieval)     | Retrieval | text         | fra         |
    | [GermanLegal1Retrieval](./available_tasks/retrieval.md#germanlegal1retrieval)     | Retrieval | text         | deu         |
    | [JapaneseLegal1Retrieval](./available_tasks/retrieval.md#japaneselegal1retrieval) | Retrieval | text         | jpn         |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(beta)

RTEB (ReTrieval Embedding Benchmark) is a comprehensive benchmark for evaluating text retrieval models across multiple specialized domains including legal, finance, code, and healthcare. It contains diverse retrieval tasks designed to test models' ability to understand domain-specific terminology and retrieve relevant documents in specialized contexts across multiple languages. The dataset includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                                        | type      | modalities   | languages                         |
    |:--------------------------------------------------------------------------------------------|:----------|:-------------|:----------------------------------|
    | [AILACasedocs](./available_tasks/retrieval.md#ailacasedocs)                                 | Retrieval | text         | eng                               |
    | [AILAStatutes](./available_tasks/retrieval.md#ailastatutes)                                 | Retrieval | text         | eng                               |
    | [LegalSummarization](./available_tasks/retrieval.md#legalsummarization)                     | Retrieval | text         | eng                               |
    | [LegalQuAD](./available_tasks/retrieval.md#legalquad)                                       | Retrieval | text         | deu                               |
    | [FinanceBenchRetrieval](./available_tasks/retrieval.md#financebenchretrieval)               | Retrieval | text         | eng                               |
    | [HC3FinanceRetrieval](./available_tasks/retrieval.md#hc3financeretrieval)                   | Retrieval | text         | eng                               |
    | [FinQARetrieval](./available_tasks/retrieval.md#finqaretrieval)                             | Retrieval | text         | eng                               |
    | [AppsRetrieval](./available_tasks/retrieval.md#appsretrieval)                               | Retrieval | text         | eng, python                       |
    | [DS1000Retrieval](./available_tasks/retrieval.md#ds1000retrieval)                           | Retrieval | text         | eng, python                       |
    | [HumanEvalRetrieval](./available_tasks/retrieval.md#humanevalretrieval)                     | Retrieval | text         | eng, python                       |
    | [MBPPRetrieval](./available_tasks/retrieval.md#mbppretrieval)                               | Retrieval | text         | eng, python                       |
    | [WikiSQLRetrieval](./available_tasks/retrieval.md#wikisqlretrieval)                         | Retrieval | text         | eng, sql                          |
    | [FreshStackRetrieval](./available_tasks/retrieval.md#freshstackretrieval)                   | Retrieval | text         | eng, go, javascript, python       |
    | [ChatDoctorRetrieval](./available_tasks/retrieval.md#chatdoctorretrieval)                   | Retrieval | text         | eng                               |
    | [CUREv1](./available_tasks/retrieval.md#curev1)                                             | Retrieval | text         | eng, fra, spa                     |
    | [MIRACLRetrievalHardNegatives](./available_tasks/retrieval.md#miraclretrievalhardnegatives) | Retrieval | text         | ara, ben, deu, eng, fas, ... (18) |
    | [Code1Retrieval](./available_tasks/retrieval.md#code1retrieval)                             | Retrieval | text         | eng                               |
    | [JapaneseCode1Retrieval](./available_tasks/retrieval.md#japanesecode1retrieval)             | Retrieval | text         | jpn                               |
    | [EnglishFinance1Retrieval](./available_tasks/retrieval.md#englishfinance1retrieval)         | Retrieval | text         | eng                               |
    | [EnglishFinance2Retrieval](./available_tasks/retrieval.md#englishfinance2retrieval)         | Retrieval | text         | eng                               |
    | [EnglishFinance3Retrieval](./available_tasks/retrieval.md#englishfinance3retrieval)         | Retrieval | text         | eng                               |
    | [EnglishFinance4Retrieval](./available_tasks/retrieval.md#englishfinance4retrieval)         | Retrieval | text         | eng                               |
    | [EnglishHealthcare1Retrieval](./available_tasks/retrieval.md#englishhealthcare1retrieval)   | Retrieval | text         | eng                               |
    | [French1Retrieval](./available_tasks/retrieval.md#french1retrieval)                         | Retrieval | text         | fra                               |
    | [FrenchLegal1Retrieval](./available_tasks/retrieval.md#frenchlegal1retrieval)               | Retrieval | text         | fra                               |
    | [German1Retrieval](./available_tasks/retrieval.md#german1retrieval)                         | Retrieval | text         | deu                               |
    | [GermanHealthcare1Retrieval](./available_tasks/retrieval.md#germanhealthcare1retrieval)     | Retrieval | text         | deu                               |
    | [GermanLegal1Retrieval](./available_tasks/retrieval.md#germanlegal1retrieval)               | Retrieval | text         | deu                               |
    | [JapaneseLegal1Retrieval](./available_tasks/retrieval.md#japaneselegal1retrieval)           | Retrieval | text         | jpn                               |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(deu, beta)

RTEB German is a subset of RTEB containing retrieval tasks in German across legal, healthcare, and business domains. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                                    | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [LegalQuAD](./available_tasks/retrieval.md#legalquad)                                   | Retrieval | text         | deu         |
    | [German1Retrieval](./available_tasks/retrieval.md#german1retrieval)                     | Retrieval | text         | deu         |
    | [GermanHealthcare1Retrieval](./available_tasks/retrieval.md#germanhealthcare1retrieval) | Retrieval | text         | deu         |
    | [GermanLegal1Retrieval](./available_tasks/retrieval.md#germanlegal1retrieval)           | Retrieval | text         | deu         |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(eng, beta)

RTEB English is a subset of RTEB containing retrieval tasks in English across legal, finance, code, and healthcare domains. Includes diverse tasks covering specialized domains such as healthcare and finance. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                                      | type      | modalities   | languages                   |
    |:------------------------------------------------------------------------------------------|:----------|:-------------|:----------------------------|
    | [AILACasedocs](./available_tasks/retrieval.md#ailacasedocs)                               | Retrieval | text         | eng                         |
    | [AILAStatutes](./available_tasks/retrieval.md#ailastatutes)                               | Retrieval | text         | eng                         |
    | [LegalSummarization](./available_tasks/retrieval.md#legalsummarization)                   | Retrieval | text         | eng                         |
    | [FinanceBenchRetrieval](./available_tasks/retrieval.md#financebenchretrieval)             | Retrieval | text         | eng                         |
    | [HC3FinanceRetrieval](./available_tasks/retrieval.md#hc3financeretrieval)                 | Retrieval | text         | eng                         |
    | [FinQARetrieval](./available_tasks/retrieval.md#finqaretrieval)                           | Retrieval | text         | eng                         |
    | [AppsRetrieval](./available_tasks/retrieval.md#appsretrieval)                             | Retrieval | text         | eng, python                 |
    | [DS1000Retrieval](./available_tasks/retrieval.md#ds1000retrieval)                         | Retrieval | text         | eng, python                 |
    | [HumanEvalRetrieval](./available_tasks/retrieval.md#humanevalretrieval)                   | Retrieval | text         | eng, python                 |
    | [MBPPRetrieval](./available_tasks/retrieval.md#mbppretrieval)                             | Retrieval | text         | eng, python                 |
    | [WikiSQLRetrieval](./available_tasks/retrieval.md#wikisqlretrieval)                       | Retrieval | text         | eng, sql                    |
    | [FreshStackRetrieval](./available_tasks/retrieval.md#freshstackretrieval)                 | Retrieval | text         | eng, go, javascript, python |
    | [ChatDoctorRetrieval](./available_tasks/retrieval.md#chatdoctorretrieval)                 | Retrieval | text         | eng                         |
    | [CUREv1](./available_tasks/retrieval.md#curev1)                                           | Retrieval | text         | eng, fra, spa               |
    | [Code1Retrieval](./available_tasks/retrieval.md#code1retrieval)                           | Retrieval | text         | eng                         |
    | [EnglishFinance1Retrieval](./available_tasks/retrieval.md#englishfinance1retrieval)       | Retrieval | text         | eng                         |
    | [EnglishFinance2Retrieval](./available_tasks/retrieval.md#englishfinance2retrieval)       | Retrieval | text         | eng                         |
    | [EnglishFinance3Retrieval](./available_tasks/retrieval.md#englishfinance3retrieval)       | Retrieval | text         | eng                         |
    | [EnglishFinance4Retrieval](./available_tasks/retrieval.md#englishfinance4retrieval)       | Retrieval | text         | eng                         |
    | [EnglishHealthcare1Retrieval](./available_tasks/retrieval.md#englishhealthcare1retrieval) | Retrieval | text         | eng                         |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(fin, beta)

RTEB Finance is a subset of RTEB  containing retrieval tasks specifically focused on financial domain including finance benchmarks, Q&A, financial document retrieval, and corporate governance. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                                | type      | modalities   | languages   |
    |:------------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [FinanceBenchRetrieval](./available_tasks/retrieval.md#financebenchretrieval)       | Retrieval | text         | eng         |
    | [HC3FinanceRetrieval](./available_tasks/retrieval.md#hc3financeretrieval)           | Retrieval | text         | eng         |
    | [FinQARetrieval](./available_tasks/retrieval.md#finqaretrieval)                     | Retrieval | text         | eng         |
    | [EnglishFinance1Retrieval](./available_tasks/retrieval.md#englishfinance1retrieval) | Retrieval | text         | eng         |
    | [EnglishFinance2Retrieval](./available_tasks/retrieval.md#englishfinance2retrieval) | Retrieval | text         | eng         |
    | [EnglishFinance3Retrieval](./available_tasks/retrieval.md#englishfinance3retrieval) | Retrieval | text         | eng         |
    | [EnglishFinance4Retrieval](./available_tasks/retrieval.md#englishfinance4retrieval) | Retrieval | text         | eng         |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(fra, beta)

RTEB French is a subset of RTEB containing retrieval tasks in French across legal and general knowledge domains. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                          | type      | modalities   | languages     |
    |:------------------------------------------------------------------------------|:----------|:-------------|:--------------|
    | [CUREv1](./available_tasks/retrieval.md#curev1)                               | Retrieval | text         | eng, fra, spa |
    | [French1Retrieval](./available_tasks/retrieval.md#french1retrieval)           | Retrieval | text         | fra           |
    | [FrenchLegal1Retrieval](./available_tasks/retrieval.md#frenchlegal1retrieval) | Retrieval | text         | fra           |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RTEB(jpn, beta)

RTEB Japanese is a subset of RTEB  containing retrieval tasks in Japanese across legal and code domains. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).

??? info "Tasks"

    | name                                                                              | type      | modalities   | languages   |
    |:----------------------------------------------------------------------------------|:----------|:-------------|:------------|
    | [JapaneseCode1Retrieval](./available_tasks/retrieval.md#japanesecode1retrieval)   | Retrieval | text         | jpn         |
    | [JapaneseLegal1Retrieval](./available_tasks/retrieval.md#japaneselegal1retrieval) | Retrieval | text         | jpn         |


??? quote "Citation"


    ```bibtex
    @article{rteb2025,
      author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
      title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
      year = {2025},
    }
    ```




####  RuSciBench

RuSciBench is a benchmark designed for evaluating sentence encoders and language models on scientific texts in both Russian and English. The data is sourced from eLibrary (www.elibrary.ru), Russia's largest electronic library of scientific publications. This benchmark facilitates the evaluation and comparison of models on various research-related tasks.

[Learn more →](https://link.springer.com/article/10.1134/S1064562424602191)



??? info "Tasks"

    | name                                                                                                     | type           | modalities   | languages   |
    |:---------------------------------------------------------------------------------------------------------|:---------------|:-------------|:------------|
    | [RuSciBenchBitextMining](./available_tasks/bitextmining.md#ruscibenchbitextmining)                       | BitextMining   | text         | eng, rus    |
    | [RuSciBenchCoreRiscClassification](./available_tasks/classification.md#ruscibenchcoreriscclassification) | Classification | text         | eng, rus    |
    | [RuSciBenchGRNTIClassification.v2](./available_tasks/classification.md#ruscibenchgrnticlassification.v2) | Classification | text         | eng, rus    |
    | [RuSciBenchOECDClassification.v2](./available_tasks/classification.md#ruscibenchoecdclassification.v2)   | Classification | text         | eng, rus    |
    | [RuSciBenchPubTypeClassification](./available_tasks/classification.md#ruscibenchpubtypeclassification)   | Classification | text         | eng, rus    |
    | [RuSciBenchCiteRetrieval](./available_tasks/retrieval.md#ruscibenchciteretrieval)                        | Retrieval      | text         | eng, rus    |
    | [RuSciBenchCociteRetrieval](./available_tasks/retrieval.md#ruscibenchcociteretrieval)                    | Retrieval      | text         | eng, rus    |
    | [RuSciBenchCitedCountRegression](./available_tasks/regression.md#ruscibenchcitedcountregression)         | Regression     | text         | eng, rus    |
    | [RuSciBenchYearPublRegression](./available_tasks/regression.md#ruscibenchyearpublregression)             | Regression     | text         | eng, rus    |


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




####  VN-MTEB (vie, v1)

A benchmark for text-embedding performance in Vietnamese.

[Learn more →](https://arxiv.org/abs/2507.21500)



??? info "Tasks"

    | name                                                                                                                     | type               | modalities   | languages   |
    |:-------------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------|:------------|
    | [ArguAna-VN](./available_tasks/retrieval.md#arguana-vn)                                                                  | Retrieval          | text         | vie         |
    | [SciFact-VN](./available_tasks/retrieval.md#scifact-vn)                                                                  | Retrieval          | text         | vie         |
    | [ClimateFEVER-VN](./available_tasks/retrieval.md#climatefever-vn)                                                        | Retrieval          | text         | vie         |
    | [FEVER-VN](./available_tasks/retrieval.md#fever-vn)                                                                      | Retrieval          | text         | vie         |
    | [DBPedia-VN](./available_tasks/retrieval.md#dbpedia-vn)                                                                  | Retrieval          | text         | vie         |
    | [NQ-VN](./available_tasks/retrieval.md#nq-vn)                                                                            | Retrieval          | text         | vie         |
    | [HotpotQA-VN](./available_tasks/retrieval.md#hotpotqa-vn)                                                                | Retrieval          | text         | vie         |
    | [MSMARCO-VN](./available_tasks/retrieval.md#msmarco-vn)                                                                  | Retrieval          | text         | vie         |
    | [TRECCOVID-VN](./available_tasks/retrieval.md#treccovid-vn)                                                              | Retrieval          | text         | vie         |
    | [FiQA2018-VN](./available_tasks/retrieval.md#fiqa2018-vn)                                                                | Retrieval          | text         | vie         |
    | [NFCorpus-VN](./available_tasks/retrieval.md#nfcorpus-vn)                                                                | Retrieval          | text         | vie         |
    | [SCIDOCS-VN](./available_tasks/retrieval.md#scidocs-vn)                                                                  | Retrieval          | text         | vie         |
    | [Touche2020-VN](./available_tasks/retrieval.md#touche2020-vn)                                                            | Retrieval          | text         | vie         |
    | [Quora-VN](./available_tasks/retrieval.md#quora-vn)                                                                      | Retrieval          | text         | vie         |
    | [CQADupstackAndroid-VN](./available_tasks/retrieval.md#cqadupstackandroid-vn)                                            | Retrieval          | text         | vie         |
    | [CQADupstackGis-VN](./available_tasks/retrieval.md#cqadupstackgis-vn)                                                    | Retrieval          | text         | vie         |
    | [CQADupstackMathematica-VN](./available_tasks/retrieval.md#cqadupstackmathematica-vn)                                    | Retrieval          | text         | vie         |
    | [CQADupstackPhysics-VN](./available_tasks/retrieval.md#cqadupstackphysics-vn)                                            | Retrieval          | text         | vie         |
    | [CQADupstackProgrammers-VN](./available_tasks/retrieval.md#cqadupstackprogrammers-vn)                                    | Retrieval          | text         | vie         |
    | [CQADupstackStats-VN](./available_tasks/retrieval.md#cqadupstackstats-vn)                                                | Retrieval          | text         | vie         |
    | [CQADupstackTex-VN](./available_tasks/retrieval.md#cqadupstacktex-vn)                                                    | Retrieval          | text         | vie         |
    | [CQADupstackUnix-VN](./available_tasks/retrieval.md#cqadupstackunix-vn)                                                  | Retrieval          | text         | vie         |
    | [CQADupstackWebmasters-VN](./available_tasks/retrieval.md#cqadupstackwebmasters-vn)                                      | Retrieval          | text         | vie         |
    | [CQADupstackWordpress-VN](./available_tasks/retrieval.md#cqadupstackwordpress-vn)                                        | Retrieval          | text         | vie         |
    | [Banking77VNClassification](./available_tasks/classification.md#banking77vnclassification)                               | Classification     | text         | vie         |
    | [EmotionVNClassification](./available_tasks/classification.md#emotionvnclassification)                                   | Classification     | text         | vie         |
    | [AmazonCounterfactualVNClassification](./available_tasks/classification.md#amazoncounterfactualvnclassification)         | Classification     | text         | vie         |
    | [MTOPDomainVNClassification](./available_tasks/classification.md#mtopdomainvnclassification)                             | Classification     | text         | vie         |
    | [TweetSentimentExtractionVNClassification](./available_tasks/classification.md#tweetsentimentextractionvnclassification) | Classification     | text         | vie         |
    | [ToxicConversationsVNClassification](./available_tasks/classification.md#toxicconversationsvnclassification)             | Classification     | text         | vie         |
    | [ImdbVNClassification](./available_tasks/classification.md#imdbvnclassification)                                         | Classification     | text         | vie         |
    | [MTOPIntentVNClassification](./available_tasks/classification.md#mtopintentvnclassification)                             | Classification     | text         | vie         |
    | [MassiveScenarioVNClassification](./available_tasks/classification.md#massivescenariovnclassification)                   | Classification     | text         | vie         |
    | [MassiveIntentVNClassification](./available_tasks/classification.md#massiveintentvnclassification)                       | Classification     | text         | vie         |
    | [AmazonReviewsVNClassification](./available_tasks/classification.md#amazonreviewsvnclassification)                       | Classification     | text         | vie         |
    | [AmazonPolarityVNClassification](./available_tasks/classification.md#amazonpolarityvnclassification)                     | Classification     | text         | vie         |
    | [SprintDuplicateQuestions-VN](./available_tasks/pairclassification.md#sprintduplicatequestions-vn)                       | PairClassification | text         | vie         |
    | [TwitterSemEval2015-VN](./available_tasks/pairclassification.md#twittersemeval2015-vn)                                   | PairClassification | text         | vie         |
    | [TwitterURLCorpus-VN](./available_tasks/pairclassification.md#twitterurlcorpus-vn)                                       | PairClassification | text         | vie         |
    | [TwentyNewsgroupsClustering-VN](./available_tasks/clustering.md#twentynewsgroupsclustering-vn)                           | Clustering         | text         | vie         |
    | [RedditClusteringP2P-VN](./available_tasks/clustering.md#redditclusteringp2p-vn)                                         | Clustering         | text         | vie         |
    | [StackExchangeClusteringP2P-VN](./available_tasks/clustering.md#stackexchangeclusteringp2p-vn)                           | Clustering         | text         | vie         |
    | [StackExchangeClustering-VN](./available_tasks/clustering.md#stackexchangeclustering-vn)                                 | Clustering         | text         | vie         |
    | [RedditClustering-VN](./available_tasks/clustering.md#redditclustering-vn)                                               | Clustering         | text         | vie         |
    | [SciDocsRR-VN](./available_tasks/reranking.md#scidocsrr-vn)                                                              | Reranking          | text         | vie         |
    | [AskUbuntuDupQuestions-VN](./available_tasks/reranking.md#askubuntudupquestions-vn)                                      | Reranking          | text         | vie         |
    | [StackOverflowDupQuestions-VN](./available_tasks/reranking.md#stackoverflowdupquestions-vn)                              | Reranking          | text         | vie         |
    | [BIOSSES-VN](./available_tasks/sts.md#biosses-vn)                                                                        | STS                | text         | vie         |
    | [SICK-R-VN](./available_tasks/sts.md#sick-r-vn)                                                                          | STS                | text         | vie         |
    | [STSBenchmark-VN](./available_tasks/sts.md#stsbenchmark-vn)                                                              | STS                | text         | vie         |


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




####  ViDoRe(v1)

Retrieve associated pages according to questions.

[Learn more →](https://arxiv.org/abs/2407.01449)



??? info "Tasks"

    | name                                                                                                                                          | type                  | modalities   | languages   |
    |:----------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|:-------------|:------------|
    | [VidoreArxivQARetrieval](./available_tasks/documentunderstanding.md#vidorearxivqaretrieval)                                                   | DocumentUnderstanding | text, image  | eng         |
    | [VidoreDocVQARetrieval](./available_tasks/documentunderstanding.md#vidoredocvqaretrieval)                                                     | DocumentUnderstanding | text, image  | eng         |
    | [VidoreInfoVQARetrieval](./available_tasks/documentunderstanding.md#vidoreinfovqaretrieval)                                                   | DocumentUnderstanding | text, image  | eng         |
    | [VidoreTabfquadRetrieval](./available_tasks/documentunderstanding.md#vidoretabfquadretrieval)                                                 | DocumentUnderstanding | text, image  | eng         |
    | [VidoreTatdqaRetrieval](./available_tasks/documentunderstanding.md#vidoretatdqaretrieval)                                                     | DocumentUnderstanding | text, image  | eng         |
    | [VidoreShiftProjectRetrieval](./available_tasks/documentunderstanding.md#vidoreshiftprojectretrieval)                                         | DocumentUnderstanding | text, image  | eng         |
    | [VidoreSyntheticDocQAAIRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaairetrieval)                                 | DocumentUnderstanding | text, image  | eng         |
    | [VidoreSyntheticDocQAEnergyRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaenergyretrieval)                         | DocumentUnderstanding | text, image  | eng         |
    | [VidoreSyntheticDocQAGovernmentReportsRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqagovernmentreportsretrieval)   | DocumentUnderstanding | text, image  | eng         |
    | [VidoreSyntheticDocQAHealthcareIndustryRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqahealthcareindustryretrieval) | DocumentUnderstanding | text, image  | eng         |


??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




####  ViDoRe(v2)

Retrieve associated pages according to questions.

[Learn more →](https://arxiv.org/abs/2407.01449)



??? info "Tasks"

    | name                                                                                                                | type                  | modalities   | languages          |
    |:--------------------------------------------------------------------------------------------------------------------|:----------------------|:-------------|:-------------------|
    | [Vidore2ESGReportsRetrieval](./available_tasks/documentunderstanding.md#vidore2esgreportsretrieval)                 | DocumentUnderstanding | text, image  | deu, eng, fra, spa |
    | [Vidore2EconomicsReportsRetrieval](./available_tasks/documentunderstanding.md#vidore2economicsreportsretrieval)     | DocumentUnderstanding | text, image  | deu, eng, fra, spa |
    | [Vidore2BioMedicalLecturesRetrieval](./available_tasks/documentunderstanding.md#vidore2biomedicallecturesretrieval) | DocumentUnderstanding | text, image  | deu, eng, fra, spa |
    | [Vidore2ESGReportsHLRetrieval](./available_tasks/documentunderstanding.md#vidore2esgreportshlretrieval)             | DocumentUnderstanding | text, image  | eng                |


??? quote "Citation"


    ```bibtex

    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }

    ```




####  VisualDocumentRetrieval

A benchmark for evaluating visual document retrieval, combining ViDoRe v1 and v2.

[Learn more →](https://arxiv.org/abs/2407.01449)



??? info "Tasks"

    | name                                                                                                                                          | type                  | modalities   | languages          |
    |:----------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|:-------------|:-------------------|
    | [VidoreArxivQARetrieval](./available_tasks/documentunderstanding.md#vidorearxivqaretrieval)                                                   | DocumentUnderstanding | text, image  | eng                |
    | [VidoreDocVQARetrieval](./available_tasks/documentunderstanding.md#vidoredocvqaretrieval)                                                     | DocumentUnderstanding | text, image  | eng                |
    | [VidoreInfoVQARetrieval](./available_tasks/documentunderstanding.md#vidoreinfovqaretrieval)                                                   | DocumentUnderstanding | text, image  | eng                |
    | [VidoreTabfquadRetrieval](./available_tasks/documentunderstanding.md#vidoretabfquadretrieval)                                                 | DocumentUnderstanding | text, image  | eng                |
    | [VidoreTatdqaRetrieval](./available_tasks/documentunderstanding.md#vidoretatdqaretrieval)                                                     | DocumentUnderstanding | text, image  | eng                |
    | [VidoreShiftProjectRetrieval](./available_tasks/documentunderstanding.md#vidoreshiftprojectretrieval)                                         | DocumentUnderstanding | text, image  | eng                |
    | [VidoreSyntheticDocQAAIRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaairetrieval)                                 | DocumentUnderstanding | text, image  | eng                |
    | [VidoreSyntheticDocQAEnergyRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqaenergyretrieval)                         | DocumentUnderstanding | text, image  | eng                |
    | [VidoreSyntheticDocQAGovernmentReportsRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqagovernmentreportsretrieval)   | DocumentUnderstanding | text, image  | eng                |
    | [VidoreSyntheticDocQAHealthcareIndustryRetrieval](./available_tasks/documentunderstanding.md#vidoresyntheticdocqahealthcareindustryretrieval) | DocumentUnderstanding | text, image  | eng                |
    | [Vidore2ESGReportsRetrieval](./available_tasks/documentunderstanding.md#vidore2esgreportsretrieval)                                           | DocumentUnderstanding | text, image  | deu, eng, fra, spa |
    | [Vidore2EconomicsReportsRetrieval](./available_tasks/documentunderstanding.md#vidore2economicsreportsretrieval)                               | DocumentUnderstanding | text, image  | deu, eng, fra, spa |
    | [Vidore2BioMedicalLecturesRetrieval](./available_tasks/documentunderstanding.md#vidore2biomedicallecturesretrieval)                           | DocumentUnderstanding | text, image  | deu, eng, fra, spa |
    | [Vidore2ESGReportsHLRetrieval](./available_tasks/documentunderstanding.md#vidore2esgreportshlretrieval)                                       | DocumentUnderstanding | text, image  | eng                |


??? quote "Citation"


    ```bibtex

    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }

    ```
<!-- END TASK DESCRIPTION -->
