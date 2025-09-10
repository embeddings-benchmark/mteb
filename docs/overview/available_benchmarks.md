# Available Benchmarks


<!-- This following is auto-generated. Changes will be overwritten. Please change the generating script. -->
<!-- START TASK DESCRIPTION -->
###  BEIR

BEIR is a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your NLP-based retrieval models within the benchmark.

[Learn more →](https://arxiv.org/abs/2104.08663)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | TRECCOVID | Retrieval | ['text']  |
    | NFCorpus | Retrieval | ['text']  |
    | NQ | Retrieval | ['text']  |
    | HotpotQA | Retrieval | ['text']  |
    | FiQA2018 | Retrieval | ['text']  |
    | ArguAna | Retrieval | ['text']  |
    | Touche2020 | Retrieval | ['text']  |
    | CQADupstackRetrieval | Retrieval | ['text']  |
    | QuoraRetrieval | Retrieval | ['text']  |
    | DBPedia | Retrieval | ['text']  |
    | SCIDOCS | Retrieval | ['text']  |
    | FEVER | Retrieval | ['text']  |
    | ClimateFEVER | Retrieval | ['text']  |
    | SciFact | Retrieval | ['text']  |
    | MSMARCO | Retrieval | ['text']  |
    


###  BEIR-NL

BEIR-NL is a Dutch adaptation of the publicly available BEIR benchmark, created through automated translation.

[Learn more →](https://arxiv.org/abs/2412.08329)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | ArguAna-NL | Retrieval | ['text']  |
    | CQADupstack-NL | Retrieval | ['text']  |
    | FEVER-NL | Retrieval | ['text']  |
    | NQ-NL | Retrieval | ['text']  |
    | Touche2020-NL | Retrieval | ['text']  |
    | FiQA2018-NL | Retrieval | ['text']  |
    | Quora-NL | Retrieval | ['text']  |
    | HotpotQA-NL | Retrieval | ['text']  |
    | SCIDOCS-NL | Retrieval | ['text']  |
    | ClimateFEVER-NL | Retrieval | ['text']  |
    | mMARCO-NL | Retrieval | ['text']  |
    | SciFact-NL | Retrieval | ['text']  |
    | DBPedia-NL | Retrieval | ['text']  |
    | NFCorpus-NL | Retrieval | ['text']  |
    | TRECCOVID-NL | Retrieval | ['text']  |
    


###  BRIGHT

BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
    BRIGHT is the first text retrieval
    benchmark that requires intensive reasoning to retrieve relevant documents with
    a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
    economics, psychology, mathematics, and coding. These queries are drawn from
    naturally occurring and carefully curated human data.
    

[Learn more →](https://brightbenchmark.github.io/)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BrightRetrieval | Retrieval | ['text']  |
    


###  BRIGHT(long)

BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
BRIGHT is the first text retrieval
benchmark that requires intensive reasoning to retrieve relevant documents with
a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
economics, psychology, mathematics, and coding. These queries are drawn from
naturally occurring and carefully curated human data.

This is the long version of the benchmark, which only filter longer documents.
    

[Learn more →](https://brightbenchmark.github.io/)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BrightLongRetrieval | Retrieval | ['text']  |
    


###  BuiltBench(eng)

"Built-Bench" is an ongoing effort aimed at evaluating text embedding models in the context of built asset management, spanning over various dicsiplines such as architeture, engineering, constrcution, and operations management of the built environment.

[Learn more →](https://arxiv.org/abs/2411.12056)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BuiltBenchClusteringP2P | Clustering | ['text']  |
    | BuiltBenchClusteringS2S | Clustering | ['text']  |
    | BuiltBenchRetrieval | Retrieval | ['text']  |
    | BuiltBenchReranking | Reranking | ['text']  |
    


###  ChemTEB

ChemTEB evaluates the performance of text embedding models on chemical domain data.

[Learn more →](https://arxiv.org/abs/2412.00532)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | PubChemSMILESBitextMining | BitextMining | ['text']  |
    | SDSEyeProtectionClassification | Classification | ['text']  |
    | SDSGlovesClassification | Classification | ['text']  |
    | WikipediaBioMetChemClassification | Classification | ['text']  |
    | WikipediaGreenhouseEnantiopureClassification | Classification | ['text']  |
    | WikipediaSolidStateColloidalClassification | Classification | ['text']  |
    | WikipediaOrganicInorganicClassification | Classification | ['text']  |
    | WikipediaCryobiologySeparationClassification | Classification | ['text']  |
    | WikipediaChemistryTopicsClassification | Classification | ['text']  |
    | WikipediaTheoreticalAppliedClassification | Classification | ['text']  |
    | WikipediaChemFieldsClassification | Classification | ['text']  |
    | WikipediaLuminescenceClassification | Classification | ['text']  |
    | WikipediaIsotopesFissionClassification | Classification | ['text']  |
    | WikipediaSaltsSemiconductorsClassification | Classification | ['text']  |
    | WikipediaBiolumNeurochemClassification | Classification | ['text']  |
    | WikipediaCrystallographyAnalyticalClassification | Classification | ['text']  |
    | WikipediaCompChemSpectroscopyClassification | Classification | ['text']  |
    | WikipediaChemEngSpecialtiesClassification | Classification | ['text']  |
    | WikipediaChemistryTopicsClustering | Clustering | ['text']  |
    | WikipediaSpecialtiesInChemistryClustering | Clustering | ['text']  |
    | PubChemAISentenceParaphrasePC | PairClassification | ['text']  |
    | PubChemSMILESPC | PairClassification | ['text']  |
    | PubChemSynonymPC | PairClassification | ['text']  |
    | PubChemWikiParagraphsPC | PairClassification | ['text']  |
    | PubChemWikiPairClassification | PairClassification | ['text']  |
    | ChemNQRetrieval | Retrieval | ['text']  |
    | ChemHotpotQARetrieval | Retrieval | ['text']  |
    


###  CoIR

CoIR: A Comprehensive Benchmark for Code Information Retrieval Models

[Learn more →](https://github.com/CoIR-team/coir)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AppsRetrieval | Retrieval | ['text']  |
    | CodeFeedbackMT | Retrieval | ['text']  |
    | CodeFeedbackST | Retrieval | ['text']  |
    | CodeSearchNetCCRetrieval | Retrieval | ['text']  |
    | CodeTransOceanContest | Retrieval | ['text']  |
    | CodeTransOceanDL | Retrieval | ['text']  |
    | CosQA | Retrieval | ['text']  |
    | COIRCodeSearchNetRetrieval | Retrieval | ['text']  |
    | StackOverflowQA | Retrieval | ['text']  |
    | SyntheticText2SQL | Retrieval | ['text']  |
    


###  CodeRAG

A benchmark for evaluating code retrieval augmented generation, testing models' ability to retrieve relevant programming solutions, tutorials and documentation.

[Learn more →](https://arxiv.org/abs/2406.14497)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | CodeRAGLibraryDocumentationSolutions | Reranking | ['text']  |
    | CodeRAGOnlineTutorials | Reranking | ['text']  |
    | CodeRAGProgrammingSolutions | Reranking | ['text']  |
    | CodeRAGStackoverflowPosts | Reranking | ['text']  |
    


###  Encodechka

A benchmark for evaluating text embedding models on Russian data.

[Learn more →](https://github.com/avidale/encodechka)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | RUParaPhraserSTS | STS | ['text']  |
    | SentiRuEval2016 | Classification | ['text']  |
    | RuToxicOKMLCUPClassification | Classification | ['text']  |
    | InappropriatenessClassificationv2 | Classification | ['text']  |
    | RuNLUIntentClassification | Classification | ['text']  |
    | XNLI | PairClassification | ['text']  |
    | RuSTSBenchmarkSTS | STS | ['text']  |
    


###  FollowIR

Retrieval w/Instructions is the task of finding relevant documents for a query that has detailed instructions.

[Learn more →](https://arxiv.org/abs/2403.15246)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | Robust04InstructionRetrieval | InstructionReranking | ['text']  |
    | News21InstructionRetrieval | InstructionReranking | ['text']  |
    | Core17InstructionRetrieval | InstructionReranking | ['text']  |
    


###  JinaVDR

Multilingual, domain-diverse and layout-rich document retrieval benchmark.

[Learn more →](https://arxiv.org/abs/2506.18902)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | JinaVDRMedicalPrescriptionsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRStanfordSlideRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRDonutVQAISynHMPRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRTableVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRChartQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRTQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDROpenAINewsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDREuropeanaDeNewsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDREuropeanaEsNewsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDREuropeanaItScansRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDREuropeanaNlLegalRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRHindiGovVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRAutomobileCatelogRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRBeveragesCatalogueRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRRamensBenchmarkRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRJDocQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRHungarianDocQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRArabicChartQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRArabicInfographicsVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDROWIDChartsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRMPMQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRJina2024YearlyBookRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRWikimediaCommonsMapsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRPlotQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRMMTabRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRCharXivOCRRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRStudentEnrollmentSyntheticRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRGitHubReadmeRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRTweetStockSyntheticsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRAirbnbSyntheticRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRShanghaiMasterPlanRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRWikimediaCommonsDocumentsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDREuropeanaFrNewsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRDocQAHealthcareIndustryRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRDocQAAI | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRShiftProjectRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRTatQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRInfovqaRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRDocVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRDocQAGovReportRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRTabFQuadRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRDocQAEnergyRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | JinaVDRArxivQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    


###  LongEmbed

LongEmbed is a benchmark oriented at exploring models' performance on long-context retrieval.
    The benchmark comprises two synthetic tasks and four carefully chosen real-world tasks,
    featuring documents of varying length and dispersed target information.
    

[Learn more →](https://arxiv.org/abs/2404.12096v2)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | LEMBNarrativeQARetrieval | Retrieval | ['text']  |
    | LEMBNeedleRetrieval | Retrieval | ['text']  |
    | LEMBPasskeyRetrieval | Retrieval | ['text']  |
    | LEMBQMSumRetrieval | Retrieval | ['text']  |
    | LEMBSummScreenFDRetrieval | Retrieval | ['text']  |
    | LEMBWikimQARetrieval | Retrieval | ['text']  |
    


###  MIEB(Img)

A image-only version of MIEB(Multilingual) that consists of 49 tasks.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | CUB200I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | FORBI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | GLDv2I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | METI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | NIGHTSI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordEasyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordMediumI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordHardI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RP2kI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisEasyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisMediumI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisHardI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | SketchyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | SOPI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | StanfordCarsI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | Birdsnap | ImageClassification | ['image']  |
    | Caltech101 | ImageClassification | ['image']  |
    | CIFAR10 | ImageClassification | ['image']  |
    | CIFAR100 | ImageClassification | ['image']  |
    | Country211 | ImageClassification | ['image']  |
    | DTD | ImageClassification | ['image']  |
    | EuroSAT | ImageClassification | ['image']  |
    | FER2013 | ImageClassification | ['image']  |
    | FGVCAircraft | ImageClassification | ['image']  |
    | Food101Classification | ImageClassification | ['image']  |
    | GTSRB | ImageClassification | ['image']  |
    | Imagenet1k | ImageClassification | ['image']  |
    | MNIST | ImageClassification | ['image']  |
    | OxfordFlowersClassification | ImageClassification | ['image']  |
    | OxfordPets | ImageClassification | ['image']  |
    | PatchCamelyon | ImageClassification | ['image']  |
    | RESISC45 | ImageClassification | ['image']  |
    | StanfordCars | ImageClassification | ['image']  |
    | STL10 | ImageClassification | ['image']  |
    | SUN397 | ImageClassification | ['image']  |
    | UCF101 | ImageClassification | ['image']  |
    | CIFAR10Clustering | ImageClustering | ['image']  |
    | CIFAR100Clustering | ImageClustering | ['image']  |
    | ImageNetDog15Clustering | ImageClustering | ['image']  |
    | ImageNet10Clustering | ImageClustering | ['image']  |
    | TinyImageNetClustering | ImageClustering | ['image']  |
    | VOC2007 | ImageClassification | ['image']  |
    | STS12VisualSTS | VisualSTS(eng) | ['image']  |
    | STS13VisualSTS | VisualSTS(eng) | ['image']  |
    | STS14VisualSTS | VisualSTS(eng) | ['image']  |
    | STS15VisualSTS | VisualSTS(eng) | ['image']  |
    | STS16VisualSTS | VisualSTS(eng) | ['image']  |
    | STS17MultilingualVisualSTS | VisualSTS(multi) | ['image']  |
    | STSBenchmarkMultilingualVisualSTS | VisualSTS(multi) | ['image']  |
    


###  MIEB(Multilingual)

MIEB(Multilingual) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 130 tasks and a total of 39 languages.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document undestanding, visual STS, and CV-centric tasks. This benchmark consists of MIEB(eng) + 3 multilingual retrieval
    datasets + the multilingual parts of VisualSTS-b and VisualSTS-16.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | Birdsnap | ImageClassification | ['image']  |
    | Caltech101 | ImageClassification | ['image']  |
    | CIFAR10 | ImageClassification | ['image']  |
    | CIFAR100 | ImageClassification | ['image']  |
    | Country211 | ImageClassification | ['image']  |
    | DTD | ImageClassification | ['image']  |
    | EuroSAT | ImageClassification | ['image']  |
    | FER2013 | ImageClassification | ['image']  |
    | FGVCAircraft | ImageClassification | ['image']  |
    | Food101Classification | ImageClassification | ['image']  |
    | GTSRB | ImageClassification | ['image']  |
    | Imagenet1k | ImageClassification | ['image']  |
    | MNIST | ImageClassification | ['image']  |
    | OxfordFlowersClassification | ImageClassification | ['image']  |
    | OxfordPets | ImageClassification | ['image']  |
    | PatchCamelyon | ImageClassification | ['image']  |
    | RESISC45 | ImageClassification | ['image']  |
    | StanfordCars | ImageClassification | ['image']  |
    | STL10 | ImageClassification | ['image']  |
    | SUN397 | ImageClassification | ['image']  |
    | UCF101 | ImageClassification | ['image']  |
    | VOC2007 | ImageClassification | ['image']  |
    | CIFAR10Clustering | ImageClustering | ['image']  |
    | CIFAR100Clustering | ImageClustering | ['image']  |
    | ImageNetDog15Clustering | ImageClustering | ['image']  |
    | ImageNet10Clustering | ImageClustering | ['image']  |
    | TinyImageNetClustering | ImageClustering | ['image']  |
    | BirdsnapZeroShot | ZeroShotClassification | ['image', 'text']  |
    | Caltech101ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CIFAR10ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CIFAR100ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CLEVRZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CLEVRCountZeroShot | ZeroShotClassification | ['text', 'image']  |
    | Country211ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | DTDZeroShot | ZeroShotClassification | ['image', 'text']  |
    | EuroSATZeroShot | ZeroShotClassification | ['image', 'text']  |
    | FER2013ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | FGVCAircraftZeroShot | ZeroShotClassification | ['text', 'image']  |
    | Food101ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | GTSRBZeroShot | ZeroShotClassification | ['image']  |
    | Imagenet1kZeroShot | ZeroShotClassification | ['image', 'text']  |
    | MNISTZeroShot | ZeroShotClassification | ['image', 'text']  |
    | OxfordPetsZeroShot | ZeroShotClassification | ['text', 'image']  |
    | PatchCamelyonZeroShot | ZeroShotClassification | ['image', 'text']  |
    | RenderedSST2 | ZeroShotClassification | ['text', 'image']  |
    | RESISC45ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | StanfordCarsZeroShot | ZeroShotClassification | ['image', 'text']  |
    | STL10ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | SUN397ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | UCF101ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | BLINKIT2IMultiChoice | VisionCentricQA | ['text', 'image']  |
    | BLINKIT2TMultiChoice | VisionCentricQA | ['text', 'image']  |
    | CVBenchCount | VisionCentricQA | ['image', 'text']  |
    | CVBenchRelation | VisionCentricQA | ['text', 'image']  |
    | CVBenchDepth | VisionCentricQA | ['text', 'image']  |
    | CVBenchDistance | VisionCentricQA | ['text', 'image']  |
    | AROCocoOrder | Compositionality | ['text', 'image']  |
    | AROFlickrOrder | Compositionality | ['text', 'image']  |
    | AROVisualAttribution | Compositionality | ['text', 'image']  |
    | AROVisualRelation | Compositionality | ['text', 'image']  |
    | SugarCrepe | Compositionality | ['text', 'image']  |
    | Winoground | Compositionality | ['text', 'image']  |
    | ImageCoDe | Compositionality | ['text', 'image']  |
    | STS12VisualSTS | VisualSTS(eng) | ['image']  |
    | STS13VisualSTS | VisualSTS(eng) | ['image']  |
    | STS14VisualSTS | VisualSTS(eng) | ['image']  |
    | STS15VisualSTS | VisualSTS(eng) | ['image']  |
    | STS16VisualSTS | VisualSTS(eng) | ['image']  |
    | BLINKIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | BLINKIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | CIRRIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | CUB200I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | EDIST2ITRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Fashion200kI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Fashion200kT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | FashionIQIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Flickr30kI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Flickr30kT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | FORBI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | GLDv2I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | GLDv2I2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | HatefulMemesI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | HatefulMemesT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | ImageCoDeT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | InfoSeekIT2ITRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | InfoSeekIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | MemotionI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | MemotionT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | METI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | MSCOCOI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | MSCOCOT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | NIGHTSI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | OVENIT2ITRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | OVENIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | ROxfordEasyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordMediumI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordHardI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RP2kI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisEasyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisMediumI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisHardI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | SciMMIRI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | SciMMIRT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | SketchyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | SOPI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | StanfordCarsI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | TUBerlinT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | VidoreArxivQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreDocVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreInfoVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTabfquadRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTatdqaRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreShiftProjectRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAAIRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAEnergyRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAGovernmentReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAHealthcareIndustryRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VisualNewsI2TRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | VisualNewsT2IRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | VizWizIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | VQA2IT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | WebQAT2ITRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | WebQAT2TRetrieval | Any2AnyRetrieval | ['text']  |
    | WITT2IRetrieval | Any2AnyMultilingualRetrieval | ['text', 'image']  |
    | XFlickr30kCoT2IRetrieval | Any2AnyMultilingualRetrieval | ['text', 'image']  |
    | XM3600T2IRetrieval | Any2AnyMultilingualRetrieval | ['text', 'image']  |
    | VisualSTS17Eng | VisualSTS(eng) | ['image']  |
    | VisualSTS-b-Eng | VisualSTS(eng) | ['image']  |
    | VisualSTS17Multilingual | VisualSTS(multi) | ['image']  |
    | VisualSTS-b-Multilingual | VisualSTS(multi) | ['image']  |
    


###  MIEB(eng)

MIEB(eng) is a comprehensive image embeddings benchmark, spanning 8 task types, covering 125 tasks.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document undestanding, visual STS, and CV-centric tasks.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | Birdsnap | ImageClassification | ['image']  |
    | Caltech101 | ImageClassification | ['image']  |
    | CIFAR10 | ImageClassification | ['image']  |
    | CIFAR100 | ImageClassification | ['image']  |
    | Country211 | ImageClassification | ['image']  |
    | DTD | ImageClassification | ['image']  |
    | EuroSAT | ImageClassification | ['image']  |
    | FER2013 | ImageClassification | ['image']  |
    | FGVCAircraft | ImageClassification | ['image']  |
    | Food101Classification | ImageClassification | ['image']  |
    | GTSRB | ImageClassification | ['image']  |
    | Imagenet1k | ImageClassification | ['image']  |
    | MNIST | ImageClassification | ['image']  |
    | OxfordFlowersClassification | ImageClassification | ['image']  |
    | OxfordPets | ImageClassification | ['image']  |
    | PatchCamelyon | ImageClassification | ['image']  |
    | RESISC45 | ImageClassification | ['image']  |
    | StanfordCars | ImageClassification | ['image']  |
    | STL10 | ImageClassification | ['image']  |
    | SUN397 | ImageClassification | ['image']  |
    | UCF101 | ImageClassification | ['image']  |
    | VOC2007 | ImageClassification | ['image']  |
    | CIFAR10Clustering | ImageClustering | ['image']  |
    | CIFAR100Clustering | ImageClustering | ['image']  |
    | ImageNetDog15Clustering | ImageClustering | ['image']  |
    | ImageNet10Clustering | ImageClustering | ['image']  |
    | TinyImageNetClustering | ImageClustering | ['image']  |
    | BirdsnapZeroShot | ZeroShotClassification | ['image', 'text']  |
    | Caltech101ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CIFAR10ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CIFAR100ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CLEVRZeroShot | ZeroShotClassification | ['text', 'image']  |
    | CLEVRCountZeroShot | ZeroShotClassification | ['text', 'image']  |
    | Country211ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | DTDZeroShot | ZeroShotClassification | ['image', 'text']  |
    | EuroSATZeroShot | ZeroShotClassification | ['image', 'text']  |
    | FER2013ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | FGVCAircraftZeroShot | ZeroShotClassification | ['text', 'image']  |
    | Food101ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | GTSRBZeroShot | ZeroShotClassification | ['image']  |
    | Imagenet1kZeroShot | ZeroShotClassification | ['image', 'text']  |
    | MNISTZeroShot | ZeroShotClassification | ['image', 'text']  |
    | OxfordPetsZeroShot | ZeroShotClassification | ['text', 'image']  |
    | PatchCamelyonZeroShot | ZeroShotClassification | ['image', 'text']  |
    | RenderedSST2 | ZeroShotClassification | ['text', 'image']  |
    | RESISC45ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | StanfordCarsZeroShot | ZeroShotClassification | ['image', 'text']  |
    | STL10ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | SUN397ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | UCF101ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | BLINKIT2IMultiChoice | VisionCentricQA | ['text', 'image']  |
    | BLINKIT2TMultiChoice | VisionCentricQA | ['text', 'image']  |
    | CVBenchCount | VisionCentricQA | ['image', 'text']  |
    | CVBenchRelation | VisionCentricQA | ['text', 'image']  |
    | CVBenchDepth | VisionCentricQA | ['text', 'image']  |
    | CVBenchDistance | VisionCentricQA | ['text', 'image']  |
    | AROCocoOrder | Compositionality | ['text', 'image']  |
    | AROFlickrOrder | Compositionality | ['text', 'image']  |
    | AROVisualAttribution | Compositionality | ['text', 'image']  |
    | AROVisualRelation | Compositionality | ['text', 'image']  |
    | SugarCrepe | Compositionality | ['text', 'image']  |
    | Winoground | Compositionality | ['text', 'image']  |
    | ImageCoDe | Compositionality | ['text', 'image']  |
    | STS12VisualSTS | VisualSTS(eng) | ['image']  |
    | STS13VisualSTS | VisualSTS(eng) | ['image']  |
    | STS14VisualSTS | VisualSTS(eng) | ['image']  |
    | STS15VisualSTS | VisualSTS(eng) | ['image']  |
    | STS16VisualSTS | VisualSTS(eng) | ['image']  |
    | BLINKIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | BLINKIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | CIRRIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | CUB200I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | EDIST2ITRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Fashion200kI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Fashion200kT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | FashionIQIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Flickr30kI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | Flickr30kT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | FORBI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | GLDv2I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | GLDv2I2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | HatefulMemesI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | HatefulMemesT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | ImageCoDeT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | InfoSeekIT2ITRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | InfoSeekIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | MemotionI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | MemotionT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | METI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | MSCOCOI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | MSCOCOT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | NIGHTSI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | OVENIT2ITRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | OVENIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | ROxfordEasyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordMediumI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | ROxfordHardI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RP2kI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisEasyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisMediumI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | RParisHardI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | SciMMIRI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | SciMMIRT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | SketchyI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | SOPI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | StanfordCarsI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | TUBerlinT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | VidoreArxivQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreDocVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreInfoVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTabfquadRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTatdqaRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreShiftProjectRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAAIRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAEnergyRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAGovernmentReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAHealthcareIndustryRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VisualNewsI2TRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | VisualNewsT2IRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | VizWizIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | VQA2IT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | WebQAT2ITRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | WebQAT2TRetrieval | Any2AnyRetrieval | ['text']  |
    | VisualSTS17Eng | VisualSTS(eng) | ['image']  |
    | VisualSTS-b-Eng | VisualSTS(eng) | ['image']  |
    


###  MIEB(lite)

MIEB(lite) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 51 tasks.
    This is a lite version of MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining
    relative rank of models.

[Learn more →](https://arxiv.org/abs/2504.10471)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | Country211 | ImageClassification | ['image']  |
    | DTD | ImageClassification | ['image']  |
    | EuroSAT | ImageClassification | ['image']  |
    | GTSRB | ImageClassification | ['image']  |
    | OxfordPets | ImageClassification | ['image']  |
    | PatchCamelyon | ImageClassification | ['image']  |
    | RESISC45 | ImageClassification | ['image']  |
    | SUN397 | ImageClassification | ['image']  |
    | ImageNetDog15Clustering | ImageClustering | ['image']  |
    | TinyImageNetClustering | ImageClustering | ['image']  |
    | CIFAR100ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | Country211ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | FER2013ZeroShot | ZeroShotClassification | ['image', 'text']  |
    | FGVCAircraftZeroShot | ZeroShotClassification | ['text', 'image']  |
    | Food101ZeroShot | ZeroShotClassification | ['text', 'image']  |
    | OxfordPetsZeroShot | ZeroShotClassification | ['text', 'image']  |
    | StanfordCarsZeroShot | ZeroShotClassification | ['image', 'text']  |
    | BLINKIT2IMultiChoice | VisionCentricQA | ['text', 'image']  |
    | CVBenchCount | VisionCentricQA | ['image', 'text']  |
    | CVBenchRelation | VisionCentricQA | ['text', 'image']  |
    | CVBenchDepth | VisionCentricQA | ['text', 'image']  |
    | CVBenchDistance | VisionCentricQA | ['text', 'image']  |
    | AROCocoOrder | Compositionality | ['text', 'image']  |
    | AROFlickrOrder | Compositionality | ['text', 'image']  |
    | AROVisualAttribution | Compositionality | ['text', 'image']  |
    | AROVisualRelation | Compositionality | ['text', 'image']  |
    | Winoground | Compositionality | ['text', 'image']  |
    | ImageCoDe | Compositionality | ['text', 'image']  |
    | STS13VisualSTS | VisualSTS(eng) | ['image']  |
    | STS15VisualSTS | VisualSTS(eng) | ['image']  |
    | VisualSTS17Multilingual | VisualSTS(multi) | ['image']  |
    | VisualSTS-b-Multilingual | VisualSTS(multi) | ['image']  |
    | CIRRIT2IRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | CUB200I2IRetrieval | Any2AnyRetrieval | ['image']  |
    | Fashion200kI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | HatefulMemesI2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | InfoSeekIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | NIGHTSI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | OVENIT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | RP2kI2IRetrieval | Any2AnyRetrieval | ['image']  |
    | VidoreDocVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreInfoVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTabfquadRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTatdqaRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreShiftProjectRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAAIRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VisualNewsI2TRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | VQA2IT2TRetrieval | Any2AnyRetrieval | ['text', 'image']  |
    | WebQAT2ITRetrieval | Any2AnyRetrieval | ['image', 'text']  |
    | WITT2IRetrieval | Any2AnyMultilingualRetrieval | ['text', 'image']  |
    | XM3600T2IRetrieval | Any2AnyMultilingualRetrieval | ['text', 'image']  |
    


###  MINERSBitextMining

Bitext Mining texts from the MINERS benchmark, a benchmark designed to evaluate the
    ability of multilingual LMs in semantic retrieval tasks,
    including bitext mining and classification via retrieval-augmented contexts.
    

[Learn more →](https://arxiv.org/pdf/2406.07424)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BUCC | BitextMining | ['text']  |
    | LinceMTBitextMining | BitextMining | ['text']  |
    | NollySentiBitextMining | BitextMining | ['text']  |
    | NusaXBitextMining | BitextMining | ['text']  |
    | NusaTranslationBitextMining | BitextMining | ['text']  |
    | PhincBitextMining | BitextMining | ['text']  |
    | Tatoeba | BitextMining | ['text']  |
    


###  MTEB(Code, v1)

A massive code embedding benchmark covering retrieval tasks in a miriad of popular programming languages.

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AppsRetrieval | Retrieval | ['text']  |
    | CodeEditSearchRetrieval | Retrieval | ['text']  |
    | CodeFeedbackMT | Retrieval | ['text']  |
    | CodeFeedbackST | Retrieval | ['text']  |
    | CodeSearchNetCCRetrieval | Retrieval | ['text']  |
    | CodeSearchNetRetrieval | Retrieval | ['text']  |
    | CodeTransOceanContest | Retrieval | ['text']  |
    | CodeTransOceanDL | Retrieval | ['text']  |
    | CosQA | Retrieval | ['text']  |
    | COIRCodeSearchNetRetrieval | Retrieval | ['text']  |
    | StackOverflowQA | Retrieval | ['text']  |
    | SyntheticText2SQL | Retrieval | ['text']  |
    


###  MTEB(Europe, v1)

A regional geopolitical text embedding benchmark targetting embedding performance on European languages.

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BornholmBitextMining | BitextMining | ['text']  |
    | BibleNLPBitextMining | BitextMining | ['text']  |
    | BUCC.v2 | BitextMining | ['text']  |
    | DiaBlaBitextMining | BitextMining | ['text']  |
    | FloresBitextMining | BitextMining | ['text']  |
    | NorwegianCourtsBitextMining | BitextMining | ['text']  |
    | NTREXBitextMining | BitextMining | ['text']  |
    | BulgarianStoreReviewSentimentClassfication | Classification | ['text']  |
    | CzechProductReviewSentimentClassification | Classification | ['text']  |
    | GreekLegalCodeClassification | Classification | ['text']  |
    | DBpediaClassification | Classification | ['text']  |
    | FinancialPhrasebankClassification | Classification | ['text']  |
    | PoemSentimentClassification | Classification | ['text']  |
    | ToxicChatClassification | Classification | ['text']  |
    | ToxicConversationsClassification | Classification | ['text']  |
    | EstonianValenceClassification | Classification | ['text']  |
    | ItaCaseholdClassification | Classification | ['text']  |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | MultiHateClassification | Classification | ['text']  |
    | NordicLangClassification | Classification | ['text']  |
    | ScalaClassification | Classification | ['text']  |
    | SwissJudgementClassification | Classification | ['text']  |
    | TweetSentimentClassification | Classification | ['text']  |
    | CBD | Classification | ['text']  |
    | PolEmo2.0-OUT | Classification | ['text']  |
    | CSFDSKMovieReviewSentimentClassification | Classification | ['text']  |
    | DalajClassification | Classification | ['text']  |
    | WikiCitiesClustering | Clustering | ['text']  |
    | RomaniBibleClustering | Clustering | ['text']  |
    | BigPatentClustering.v2 | Clustering | ['text']  |
    | BiorxivClusteringP2P.v2 | Clustering | ['text']  |
    | AlloProfClusteringS2S.v2 | Clustering | ['text']  |
    | HALClusteringS2S.v2 | Clustering | ['text']  |
    | SIB200ClusteringS2S | Clustering | ['text']  |
    | WikiClusteringP2P.v2 | Clustering | ['text']  |
    | StackOverflowQA | Retrieval | ['text']  |
    | TwitterHjerneRetrieval | Retrieval | ['text']  |
    | LegalQuAD | Retrieval | ['text']  |
    | ArguAna | Retrieval | ['text']  |
    | HagridRetrieval | Retrieval | ['text']  |
    | LegalBenchCorporateLobbying | Retrieval | ['text']  |
    | LEMBPasskeyRetrieval | Retrieval | ['text']  |
    | SCIDOCS | Retrieval | ['text']  |
    | SpartQA | Retrieval | ['text']  |
    | TempReasonL1 | Retrieval | ['text']  |
    | WinoGrande | Retrieval | ['text']  |
    | AlloprofRetrieval | Retrieval | ['text']  |
    | BelebeleRetrieval | Retrieval | ['text']  |
    | StatcanDialogueDatasetRetrieval | Retrieval | ['text']  |
    | WikipediaRetrievalMultilingual | Retrieval | ['text']  |
    | Core17InstructionRetrieval | InstructionReranking | ['text']  |
    | News21InstructionRetrieval | InstructionReranking | ['text']  |
    | Robust04InstructionRetrieval | InstructionReranking | ['text']  |
    | MalteseNewsClassification | MultilabelClassification | ['text']  |
    | MultiEURLEXMultilabelClassification | MultilabelClassification | ['text']  |
    | CTKFactsNLI | PairClassification | ['text']  |
    | SprintDuplicateQuestions | PairClassification | ['text']  |
    | OpusparcusPC | PairClassification | ['text']  |
    | RTE3 | PairClassification | ['text']  |
    | XNLI | PairClassification | ['text']  |
    | PSC | PairClassification | ['text']  |
    | WebLINXCandidatesReranking | Reranking | ['text']  |
    | AlloprofReranking | Reranking | ['text']  |
    | WikipediaRerankingMultilingual | Reranking | ['text']  |
    | SICK-R | STS | ['text']  |
    | STS12 | STS | ['text']  |
    | STS14 | STS | ['text']  |
    | STS15 | STS | ['text']  |
    | STSBenchmark | STS | ['text']  |
    | FinParaSTS | STS | ['text']  |
    | STS17 | STS | ['text']  |
    | SICK-R-PL | STS | ['text']  |
    | STSES | STS | ['text']  |
    


###  MTEB(Indic, v1)

A regional geopolitical text embedding benchmark targetting embedding performance on Indic languages.

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | IN22ConvBitextMining | BitextMining | ['text']  |
    | IN22GenBitextMining | BitextMining | ['text']  |
    | IndicGenBenchFloresBitextMining | BitextMining | ['text']  |
    | LinceMTBitextMining | BitextMining | ['text']  |
    | SIB200ClusteringS2S | Clustering | ['text']  |
    | BengaliSentimentAnalysis | Classification | ['text']  |
    | GujaratiNewsClassification | Classification | ['text']  |
    | HindiDiscourseClassification | Classification | ['text']  |
    | SentimentAnalysisHindi | Classification | ['text']  |
    | MalayalamNewsClassification | Classification | ['text']  |
    | IndicLangClassification | Classification | ['text']  |
    | MTOPIntentClassification | Classification | ['text']  |
    | MultiHateClassification | Classification | ['text']  |
    | TweetSentimentClassification | Classification | ['text']  |
    | NepaliNewsClassification | Classification | ['text']  |
    | PunjabiNewsClassification | Classification | ['text']  |
    | SanskritShlokasClassification | Classification | ['text']  |
    | UrduRomanSentimentClassification | Classification | ['text']  |
    | XNLI | PairClassification | ['text']  |
    | BelebeleRetrieval | Retrieval | ['text']  |
    | XQuADRetrieval | Retrieval | ['text']  |
    | WikipediaRerankingMultilingual | Reranking | ['text']  |
    | IndicCrosslingualSTS | STS | ['text']  |
    


###  MTEB(Law, v1)

A benchmark of retrieval tasks in the legal domain.

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AILACasedocs | Retrieval | ['text']  |
    | AILAStatutes | Retrieval | ['text']  |
    | LegalSummarization | Retrieval | ['text']  |
    | GerDaLIRSmall | Retrieval | ['text']  |
    | LeCaRDv2 | Retrieval | ['text']  |
    | LegalBenchConsumerContractsQA | Retrieval | ['text']  |
    | LegalBenchCorporateLobbying | Retrieval | ['text']  |
    | LegalQuAD | Retrieval | ['text']  |
    


###  MTEB(Medical, v1)

A curated set of MTEB tasks designed to evaluate systems in the context of medical information retrieval.

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | CUREv1 | Retrieval | ['text']  |
    | NFCorpus | Retrieval | ['text']  |
    | TRECCOVID | Retrieval | ['text']  |
    | TRECCOVID-PL | Retrieval | ['text']  |
    | SciFact | Retrieval | ['text']  |
    | SciFact-PL | Retrieval | ['text']  |
    | MedicalQARetrieval | Retrieval | ['text']  |
    | PublicHealthQA | Retrieval | ['text']  |
    | MedrxivClusteringP2P.v2 | Clustering | ['text']  |
    | MedrxivClusteringS2S.v2 | Clustering | ['text']  |
    | CmedqaRetrieval | Retrieval | ['text']  |
    | CMedQAv2-reranking | Reranking | ['text']  |
    


###  MTEB(Multilingual, v1)

A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages. This benhcmark has been replaced by MTEB(Multilingual, v2) as one of the datasets (SNLHierarchicalClustering) included in v1 was removed from the Hugging Face Hub.

[Learn more →](https://arxiv.org/abs/2502.13595)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BornholmBitextMining | BitextMining | ['text']  |
    | BibleNLPBitextMining | BitextMining | ['text']  |
    | BUCC.v2 | BitextMining | ['text']  |
    | DiaBlaBitextMining | BitextMining | ['text']  |
    | FloresBitextMining | BitextMining | ['text']  |
    | IN22GenBitextMining | BitextMining | ['text']  |
    | IndicGenBenchFloresBitextMining | BitextMining | ['text']  |
    | NollySentiBitextMining | BitextMining | ['text']  |
    | NorwegianCourtsBitextMining | BitextMining | ['text']  |
    | NTREXBitextMining | BitextMining | ['text']  |
    | NusaTranslationBitextMining | BitextMining | ['text']  |
    | NusaXBitextMining | BitextMining | ['text']  |
    | Tatoeba | BitextMining | ['text']  |
    | BulgarianStoreReviewSentimentClassfication | Classification | ['text']  |
    | CzechProductReviewSentimentClassification | Classification | ['text']  |
    | GreekLegalCodeClassification | Classification | ['text']  |
    | DBpediaClassification | Classification | ['text']  |
    | FinancialPhrasebankClassification | Classification | ['text']  |
    | PoemSentimentClassification | Classification | ['text']  |
    | ToxicConversationsClassification | Classification | ['text']  |
    | TweetTopicSingleClassification | Classification | ['text']  |
    | EstonianValenceClassification | Classification | ['text']  |
    | FilipinoShopeeReviewsClassification | Classification | ['text']  |
    | GujaratiNewsClassification | Classification | ['text']  |
    | SentimentAnalysisHindi | Classification | ['text']  |
    | IndonesianIdClickbaitClassification | Classification | ['text']  |
    | ItaCaseholdClassification | Classification | ['text']  |
    | KorSarcasmClassification | Classification | ['text']  |
    | KurdishSentimentClassification | Classification | ['text']  |
    | MacedonianTweetSentimentClassification | Classification | ['text']  |
    | AfriSentiClassification | Classification | ['text']  |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | CataloniaTweetClassification | Classification | ['text']  |
    | CyrillicTurkicLangClassification | Classification | ['text']  |
    | IndicLangClassification | Classification | ['text']  |
    | MasakhaNEWSClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MultiHateClassification | Classification | ['text']  |
    | NordicLangClassification | Classification | ['text']  |
    | NusaParagraphEmotionClassification | Classification | ['text']  |
    | NusaX-senti | Classification | ['text']  |
    | ScalaClassification | Classification | ['text']  |
    | SwissJudgementClassification | Classification | ['text']  |
    | NepaliNewsClassification | Classification | ['text']  |
    | OdiaNewsClassification | Classification | ['text']  |
    | PunjabiNewsClassification | Classification | ['text']  |
    | PolEmo2.0-OUT | Classification | ['text']  |
    | PAC | Classification | ['text']  |
    | SinhalaNewsClassification | Classification | ['text']  |
    | CSFDSKMovieReviewSentimentClassification | Classification | ['text']  |
    | SiswatiNewsClassification | Classification | ['text']  |
    | SlovakMovieReviewSentimentClassification | Classification | ['text']  |
    | SwahiliNewsClassification | Classification | ['text']  |
    | DalajClassification | Classification | ['text']  |
    | TswanaNewsClassification | Classification | ['text']  |
    | IsiZuluNewsClassification | Classification | ['text']  |
    | WikiCitiesClustering | Clustering | ['text']  |
    | MasakhaNEWSClusteringS2S | Clustering | ['text']  |
    | RomaniBibleClustering | Clustering | ['text']  |
    | ArXivHierarchicalClusteringP2P | Clustering | ['text']  |
    | ArXivHierarchicalClusteringS2S | Clustering | ['text']  |
    | BigPatentClustering.v2 | Clustering | ['text']  |
    | BiorxivClusteringP2P.v2 | Clustering | ['text']  |
    | MedrxivClusteringP2P.v2 | Clustering | ['text']  |
    | StackExchangeClustering.v2 | Clustering | ['text']  |
    | AlloProfClusteringS2S.v2 | Clustering | ['text']  |
    | HALClusteringS2S.v2 | Clustering | ['text']  |
    | SIB200ClusteringS2S | Clustering | ['text']  |
    | WikiClusteringP2P.v2 | Clustering | ['text']  |
    | PlscClusteringP2P.v2 | Clustering | ['text']  |
    | SwednClusteringP2P | Clustering | ['text']  |
    | CLSClusteringP2P.v2 | Clustering | ['text']  |
    | StackOverflowQA | Retrieval | ['text']  |
    | TwitterHjerneRetrieval | Retrieval | ['text']  |
    | AILAStatutes | Retrieval | ['text']  |
    | ArguAna | Retrieval | ['text']  |
    | HagridRetrieval | Retrieval | ['text']  |
    | LegalBenchCorporateLobbying | Retrieval | ['text']  |
    | LEMBPasskeyRetrieval | Retrieval | ['text']  |
    | SCIDOCS | Retrieval | ['text']  |
    | SpartQA | Retrieval | ['text']  |
    | TempReasonL1 | Retrieval | ['text']  |
    | TRECCOVID | Retrieval | ['text']  |
    | WinoGrande | Retrieval | ['text']  |
    | BelebeleRetrieval | Retrieval | ['text']  |
    | MLQARetrieval | Retrieval | ['text']  |
    | StatcanDialogueDatasetRetrieval | Retrieval | ['text']  |
    | WikipediaRetrievalMultilingual | Retrieval | ['text']  |
    | CovidRetrieval | Retrieval | ['text']  |
    | Core17InstructionRetrieval | InstructionReranking | ['text']  |
    | News21InstructionRetrieval | InstructionReranking | ['text']  |
    | Robust04InstructionRetrieval | InstructionReranking | ['text']  |
    | KorHateSpeechMLClassification | MultilabelClassification | ['text']  |
    | MalteseNewsClassification | MultilabelClassification | ['text']  |
    | MultiEURLEXMultilabelClassification | MultilabelClassification | ['text']  |
    | BrazilianToxicTweetsClassification | MultilabelClassification | ['text']  |
    | CEDRClassification | MultilabelClassification | ['text']  |
    | CTKFactsNLI | PairClassification | ['text']  |
    | SprintDuplicateQuestions | PairClassification | ['text']  |
    | TwitterURLCorpus | PairClassification | ['text']  |
    | ArmenianParaphrasePC | PairClassification | ['text']  |
    | indonli | PairClassification | ['text']  |
    | OpusparcusPC | PairClassification | ['text']  |
    | PawsXPairClassification | PairClassification | ['text']  |
    | RTE3 | PairClassification | ['text']  |
    | XNLI | PairClassification | ['text']  |
    | PpcPC | PairClassification | ['text']  |
    | TERRa | PairClassification | ['text']  |
    | WebLINXCandidatesReranking | Reranking | ['text']  |
    | AlloprofReranking | Reranking | ['text']  |
    | VoyageMMarcoReranking | Reranking | ['text']  |
    | WikipediaRerankingMultilingual | Reranking | ['text']  |
    | RuBQReranking | Reranking | ['text']  |
    | T2Reranking | Reranking | ['text']  |
    | GermanSTSBenchmark | STS | ['text']  |
    | SICK-R | STS | ['text']  |
    | STS12 | STS | ['text']  |
    | STS13 | STS | ['text']  |
    | STS14 | STS | ['text']  |
    | STS15 | STS | ['text']  |
    | STSBenchmark | STS | ['text']  |
    | FaroeseSTS | STS | ['text']  |
    | FinParaSTS | STS | ['text']  |
    | JSICK | STS | ['text']  |
    | IndicCrosslingualSTS | STS | ['text']  |
    | SemRel24STS | STS | ['text']  |
    | STS17 | STS | ['text']  |
    | STS22.v2 | STS | ['text']  |
    | STSES | STS | ['text']  |
    | STSB | STS | ['text']  |
    | MIRACLRetrievalHardNegatives | Retrieval | ['text']  |
    | SNLHierarchicalClusteringP2P | Clustering | ['text']  |
    


###  MTEB(Multilingual, v2)

A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages. 

[Learn more →](https://arxiv.org/abs/2502.13595)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BornholmBitextMining | BitextMining | ['text']  |
    | BibleNLPBitextMining | BitextMining | ['text']  |
    | BUCC.v2 | BitextMining | ['text']  |
    | DiaBlaBitextMining | BitextMining | ['text']  |
    | FloresBitextMining | BitextMining | ['text']  |
    | IN22GenBitextMining | BitextMining | ['text']  |
    | IndicGenBenchFloresBitextMining | BitextMining | ['text']  |
    | NollySentiBitextMining | BitextMining | ['text']  |
    | NorwegianCourtsBitextMining | BitextMining | ['text']  |
    | NTREXBitextMining | BitextMining | ['text']  |
    | NusaTranslationBitextMining | BitextMining | ['text']  |
    | NusaXBitextMining | BitextMining | ['text']  |
    | Tatoeba | BitextMining | ['text']  |
    | BulgarianStoreReviewSentimentClassfication | Classification | ['text']  |
    | CzechProductReviewSentimentClassification | Classification | ['text']  |
    | GreekLegalCodeClassification | Classification | ['text']  |
    | DBpediaClassification | Classification | ['text']  |
    | FinancialPhrasebankClassification | Classification | ['text']  |
    | PoemSentimentClassification | Classification | ['text']  |
    | ToxicConversationsClassification | Classification | ['text']  |
    | TweetTopicSingleClassification | Classification | ['text']  |
    | EstonianValenceClassification | Classification | ['text']  |
    | FilipinoShopeeReviewsClassification | Classification | ['text']  |
    | GujaratiNewsClassification | Classification | ['text']  |
    | SentimentAnalysisHindi | Classification | ['text']  |
    | IndonesianIdClickbaitClassification | Classification | ['text']  |
    | ItaCaseholdClassification | Classification | ['text']  |
    | KorSarcasmClassification | Classification | ['text']  |
    | KurdishSentimentClassification | Classification | ['text']  |
    | MacedonianTweetSentimentClassification | Classification | ['text']  |
    | AfriSentiClassification | Classification | ['text']  |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | CataloniaTweetClassification | Classification | ['text']  |
    | CyrillicTurkicLangClassification | Classification | ['text']  |
    | IndicLangClassification | Classification | ['text']  |
    | MasakhaNEWSClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MultiHateClassification | Classification | ['text']  |
    | NordicLangClassification | Classification | ['text']  |
    | NusaParagraphEmotionClassification | Classification | ['text']  |
    | NusaX-senti | Classification | ['text']  |
    | ScalaClassification | Classification | ['text']  |
    | SwissJudgementClassification | Classification | ['text']  |
    | NepaliNewsClassification | Classification | ['text']  |
    | OdiaNewsClassification | Classification | ['text']  |
    | PunjabiNewsClassification | Classification | ['text']  |
    | PolEmo2.0-OUT | Classification | ['text']  |
    | PAC | Classification | ['text']  |
    | SinhalaNewsClassification | Classification | ['text']  |
    | CSFDSKMovieReviewSentimentClassification | Classification | ['text']  |
    | SiswatiNewsClassification | Classification | ['text']  |
    | SlovakMovieReviewSentimentClassification | Classification | ['text']  |
    | SwahiliNewsClassification | Classification | ['text']  |
    | DalajClassification | Classification | ['text']  |
    | TswanaNewsClassification | Classification | ['text']  |
    | IsiZuluNewsClassification | Classification | ['text']  |
    | WikiCitiesClustering | Clustering | ['text']  |
    | MasakhaNEWSClusteringS2S | Clustering | ['text']  |
    | RomaniBibleClustering | Clustering | ['text']  |
    | ArXivHierarchicalClusteringP2P | Clustering | ['text']  |
    | ArXivHierarchicalClusteringS2S | Clustering | ['text']  |
    | BigPatentClustering.v2 | Clustering | ['text']  |
    | BiorxivClusteringP2P.v2 | Clustering | ['text']  |
    | MedrxivClusteringP2P.v2 | Clustering | ['text']  |
    | StackExchangeClustering.v2 | Clustering | ['text']  |
    | AlloProfClusteringS2S.v2 | Clustering | ['text']  |
    | HALClusteringS2S.v2 | Clustering | ['text']  |
    | SIB200ClusteringS2S | Clustering | ['text']  |
    | WikiClusteringP2P.v2 | Clustering | ['text']  |
    | PlscClusteringP2P.v2 | Clustering | ['text']  |
    | SwednClusteringP2P | Clustering | ['text']  |
    | CLSClusteringP2P.v2 | Clustering | ['text']  |
    | StackOverflowQA | Retrieval | ['text']  |
    | TwitterHjerneRetrieval | Retrieval | ['text']  |
    | AILAStatutes | Retrieval | ['text']  |
    | ArguAna | Retrieval | ['text']  |
    | HagridRetrieval | Retrieval | ['text']  |
    | LegalBenchCorporateLobbying | Retrieval | ['text']  |
    | LEMBPasskeyRetrieval | Retrieval | ['text']  |
    | SCIDOCS | Retrieval | ['text']  |
    | SpartQA | Retrieval | ['text']  |
    | TempReasonL1 | Retrieval | ['text']  |
    | TRECCOVID | Retrieval | ['text']  |
    | WinoGrande | Retrieval | ['text']  |
    | BelebeleRetrieval | Retrieval | ['text']  |
    | MLQARetrieval | Retrieval | ['text']  |
    | StatcanDialogueDatasetRetrieval | Retrieval | ['text']  |
    | WikipediaRetrievalMultilingual | Retrieval | ['text']  |
    | CovidRetrieval | Retrieval | ['text']  |
    | Core17InstructionRetrieval | InstructionReranking | ['text']  |
    | News21InstructionRetrieval | InstructionReranking | ['text']  |
    | Robust04InstructionRetrieval | InstructionReranking | ['text']  |
    | KorHateSpeechMLClassification | MultilabelClassification | ['text']  |
    | MalteseNewsClassification | MultilabelClassification | ['text']  |
    | MultiEURLEXMultilabelClassification | MultilabelClassification | ['text']  |
    | BrazilianToxicTweetsClassification | MultilabelClassification | ['text']  |
    | CEDRClassification | MultilabelClassification | ['text']  |
    | CTKFactsNLI | PairClassification | ['text']  |
    | SprintDuplicateQuestions | PairClassification | ['text']  |
    | TwitterURLCorpus | PairClassification | ['text']  |
    | ArmenianParaphrasePC | PairClassification | ['text']  |
    | indonli | PairClassification | ['text']  |
    | OpusparcusPC | PairClassification | ['text']  |
    | PawsXPairClassification | PairClassification | ['text']  |
    | RTE3 | PairClassification | ['text']  |
    | XNLI | PairClassification | ['text']  |
    | PpcPC | PairClassification | ['text']  |
    | TERRa | PairClassification | ['text']  |
    | WebLINXCandidatesReranking | Reranking | ['text']  |
    | AlloprofReranking | Reranking | ['text']  |
    | VoyageMMarcoReranking | Reranking | ['text']  |
    | WikipediaRerankingMultilingual | Reranking | ['text']  |
    | RuBQReranking | Reranking | ['text']  |
    | T2Reranking | Reranking | ['text']  |
    | GermanSTSBenchmark | STS | ['text']  |
    | SICK-R | STS | ['text']  |
    | STS12 | STS | ['text']  |
    | STS13 | STS | ['text']  |
    | STS14 | STS | ['text']  |
    | STS15 | STS | ['text']  |
    | STSBenchmark | STS | ['text']  |
    | FaroeseSTS | STS | ['text']  |
    | FinParaSTS | STS | ['text']  |
    | JSICK | STS | ['text']  |
    | IndicCrosslingualSTS | STS | ['text']  |
    | SemRel24STS | STS | ['text']  |
    | STS17 | STS | ['text']  |
    | STS22.v2 | STS | ['text']  |
    | STSES | STS | ['text']  |
    | STSB | STS | ['text']  |
    | MIRACLRetrievalHardNegatives | Retrieval | ['text']  |
    


###  MTEB(Scandinavian, v1)

A curated selection of tasks coverering the Scandinavian languages; Danish, Swedish and Norwegian, including Bokmål and Nynorsk.

[Learn more →](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | BornholmBitextMining | BitextMining | ['text']  |
    | NorwegianCourtsBitextMining | BitextMining | ['text']  |
    | AngryTweetsClassification | Classification | ['text']  |
    | DanishPoliticalCommentsClassification | Classification | ['text']  |
    | DalajClassification | Classification | ['text']  |
    | DKHateClassification | Classification | ['text']  |
    | LccSentimentClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | NordicLangClassification | Classification | ['text']  |
    | NoRecClassification | Classification | ['text']  |
    | NorwegianParliamentClassification | Classification | ['text']  |
    | ScalaClassification | Classification | ['text']  |
    | SwedishSentimentClassification | Classification | ['text']  |
    | SweRecClassification | Classification | ['text']  |
    | DanFeverRetrieval | Retrieval | ['text']  |
    | NorQuadRetrieval | Retrieval | ['text']  |
    | SNLRetrieval | Retrieval | ['text']  |
    | SwednRetrieval | Retrieval | ['text']  |
    | SweFaqRetrieval | Retrieval | ['text']  |
    | TV2Nordretrieval | Retrieval | ['text']  |
    | TwitterHjerneRetrieval | Retrieval | ['text']  |
    | SNLHierarchicalClusteringS2S | Clustering | ['text']  |
    | SNLHierarchicalClusteringP2P | Clustering | ['text']  |
    | SwednClusteringP2P | Clustering | ['text']  |
    | SwednClusteringS2S | Clustering | ['text']  |
    | VGHierarchicalClusteringS2S | Clustering | ['text']  |
    | VGHierarchicalClusteringP2P | Clustering | ['text']  |
    


###  MTEB(cmn, v1)

The Chinese Massive Text Embedding Benchmark (C-MTEB) is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets.

[Learn more →](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | T2Retrieval | Retrieval | ['text']  |
    | MMarcoRetrieval | Retrieval | ['text']  |
    | DuRetrieval | Retrieval | ['text']  |
    | CovidRetrieval | Retrieval | ['text']  |
    | CmedqaRetrieval | Retrieval | ['text']  |
    | EcomRetrieval | Retrieval | ['text']  |
    | MedicalRetrieval | Retrieval | ['text']  |
    | VideoRetrieval | Retrieval | ['text']  |
    | T2Reranking | Reranking | ['text']  |
    | MMarcoReranking | Reranking | ['text']  |
    | CMedQAv1-reranking | Reranking | ['text']  |
    | CMedQAv2-reranking | Reranking | ['text']  |
    | Ocnli | PairClassification | ['text']  |
    | Cmnli | PairClassification | ['text']  |
    | CLSClusteringS2S | Clustering | ['text']  |
    | CLSClusteringP2P | Clustering | ['text']  |
    | ThuNewsClusteringS2S | Clustering | ['text']  |
    | ThuNewsClusteringP2P | Clustering | ['text']  |
    | LCQMC | STS | ['text']  |
    | PAWSX | STS | ['text']  |
    | AFQMC | STS | ['text']  |
    | QBQTC | STS | ['text']  |
    | TNews | Classification | ['text']  |
    | IFlyTek | Classification | ['text']  |
    | Waimai | Classification | ['text']  |
    | OnlineShopping | Classification | ['text']  |
    | JDReview | Classification | ['text']  |
    | MultilingualSentiment | Classification | ['text']  |
    | ATEC | STS | ['text']  |
    | BQ | STS | ['text']  |
    | STSB | STS | ['text']  |
    | MultilingualSentiment | Classification | ['text']  |
    


###  MTEB(deu, v1)

A benchmark for text-embedding performance in German.

[Learn more →](https://arxiv.org/html/2401.02709v1)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | AmazonReviewsClassification | Classification | ['text']  |
    | MTOPDomainClassification | Classification | ['text']  |
    | MTOPIntentClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | BlurbsClusteringP2P | Clustering | ['text']  |
    | BlurbsClusteringS2S | Clustering | ['text']  |
    | TenKGnadClusteringP2P | Clustering | ['text']  |
    | TenKGnadClusteringS2S | Clustering | ['text']  |
    | FalseFriendsGermanEnglish | PairClassification | ['text']  |
    | PawsXPairClassification | PairClassification | ['text']  |
    | MIRACLReranking | Reranking | ['text']  |
    | GermanQuAD-Retrieval | Retrieval | ['text']  |
    | GermanDPR | Retrieval | ['text']  |
    | XMarket | Retrieval | ['text']  |
    | GerDaLIR | Retrieval | ['text']  |
    | GermanSTSBenchmark | STS | ['text']  |
    | STS22 | STS | ['text']  |
    


###  MTEB(eng, v1)

The original English benchmark by Muennighoff et al., (2023).
We recommend that you use [MTEB(eng, v2)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v2%29) instead, as it uses updated versions of the task, making it notably faster to run and resolving [a known bug](https://github.com/embeddings-benchmark/mteb/issues/1156) in existing tasks. This benchmark also removes datasets common for fine-tuning, such as MSMARCO, which makes model performance scores more comparable. However, generally, both benchmarks provide similar estimates.
    

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AmazonPolarityClassification | Classification | ['text']  |
    | AmazonReviewsClassification | Classification | ['text']  |
    | ArguAna | Retrieval | ['text']  |
    | ArxivClusteringP2P | Clustering | ['text']  |
    | ArxivClusteringS2S | Clustering | ['text']  |
    | AskUbuntuDupQuestions | Reranking | ['text']  |
    | BIOSSES | STS | ['text']  |
    | Banking77Classification | Classification | ['text']  |
    | BiorxivClusteringP2P | Clustering | ['text']  |
    | BiorxivClusteringS2S | Clustering | ['text']  |
    | CQADupstackRetrieval | Retrieval | ['text']  |
    | ClimateFEVER | Retrieval | ['text']  |
    | DBPedia | Retrieval | ['text']  |
    | EmotionClassification | Classification | ['text']  |
    | FEVER | Retrieval | ['text']  |
    | FiQA2018 | Retrieval | ['text']  |
    | HotpotQA | Retrieval | ['text']  |
    | ImdbClassification | Classification | ['text']  |
    | MTOPDomainClassification | Classification | ['text']  |
    | MTOPIntentClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | MedrxivClusteringP2P | Clustering | ['text']  |
    | MedrxivClusteringS2S | Clustering | ['text']  |
    | MindSmallReranking | Reranking | ['text']  |
    | NFCorpus | Retrieval | ['text']  |
    | NQ | Retrieval | ['text']  |
    | QuoraRetrieval | Retrieval | ['text']  |
    | RedditClustering | Clustering | ['text']  |
    | RedditClusteringP2P | Clustering | ['text']  |
    | SCIDOCS | Retrieval | ['text']  |
    | SICK-R | STS | ['text']  |
    | STS12 | STS | ['text']  |
    | STS13 | STS | ['text']  |
    | STS14 | STS | ['text']  |
    | STS15 | STS | ['text']  |
    | STS16 | STS | ['text']  |
    | STSBenchmark | STS | ['text']  |
    | SciDocsRR | Reranking | ['text']  |
    | SciFact | Retrieval | ['text']  |
    | SprintDuplicateQuestions | PairClassification | ['text']  |
    | StackExchangeClustering | Clustering | ['text']  |
    | StackExchangeClusteringP2P | Clustering | ['text']  |
    | StackOverflowDupQuestions | Reranking | ['text']  |
    | SummEval | Summarization | ['text']  |
    | TRECCOVID | Retrieval | ['text']  |
    | Touche2020 | Retrieval | ['text']  |
    | ToxicConversationsClassification | Classification | ['text']  |
    | TweetSentimentExtractionClassification | Classification | ['text']  |
    | TwentyNewsgroupsClustering | Clustering | ['text']  |
    | TwitterSemEval2015 | PairClassification | ['text']  |
    | TwitterURLCorpus | PairClassification | ['text']  |
    | MSMARCO | Retrieval | ['text']  |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | STS17 | STS | ['text']  |
    | STS22 | STS | ['text']  |
    


###  MTEB(eng, v2)

The new English Massive Text Embedding Benchmark.
This benchmark was created to account for the fact that many models have now been finetuned
to tasks in the original MTEB, and contains tasks that are not as frequently used for model training.
This way the new benchmark and leaderboard can give our users a more realistic expectation of models' generalization performance.

The original MTEB leaderboard is available under the name [MTEB(eng, v1)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v1%29).
    

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | ArguAna | Retrieval | ['text']  |
    | ArXivHierarchicalClusteringP2P | Clustering | ['text']  |
    | ArXivHierarchicalClusteringS2S | Clustering | ['text']  |
    | AskUbuntuDupQuestions | Reranking | ['text']  |
    | BIOSSES | STS | ['text']  |
    | Banking77Classification | Classification | ['text']  |
    | BiorxivClusteringP2P.v2 | Clustering | ['text']  |
    | CQADupstackGamingRetrieval | Retrieval | ['text']  |
    | CQADupstackUnixRetrieval | Retrieval | ['text']  |
    | ClimateFEVERHardNegatives | Retrieval | ['text']  |
    | FEVERHardNegatives | Retrieval | ['text']  |
    | FiQA2018 | Retrieval | ['text']  |
    | HotpotQAHardNegatives | Retrieval | ['text']  |
    | ImdbClassification | Classification | ['text']  |
    | MTOPDomainClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | MedrxivClusteringP2P.v2 | Clustering | ['text']  |
    | MedrxivClusteringS2S.v2 | Clustering | ['text']  |
    | MindSmallReranking | Reranking | ['text']  |
    | SCIDOCS | Retrieval | ['text']  |
    | SICK-R | STS | ['text']  |
    | STS12 | STS | ['text']  |
    | STS13 | STS | ['text']  |
    | STS14 | STS | ['text']  |
    | STS15 | STS | ['text']  |
    | STSBenchmark | STS | ['text']  |
    | SprintDuplicateQuestions | PairClassification | ['text']  |
    | StackExchangeClustering.v2 | Clustering | ['text']  |
    | StackExchangeClusteringP2P.v2 | Clustering | ['text']  |
    | TRECCOVID | Retrieval | ['text']  |
    | Touche2020Retrieval.v3 | Retrieval | ['text']  |
    | ToxicConversationsClassification | Classification | ['text']  |
    | TweetSentimentExtractionClassification | Classification | ['text']  |
    | TwentyNewsgroupsClustering.v2 | Clustering | ['text']  |
    | TwitterSemEval2015 | PairClassification | ['text']  |
    | TwitterURLCorpus | PairClassification | ['text']  |
    | SummEvalSummarization.v2 | Summarization | ['text']  |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | STS17 | STS | ['text']  |
    | STS22.v2 | STS | ['text']  |
    


###  MTEB(fas, v1)

The Persian Massive Text Embedding Benchmark (FaMTEB) is a comprehensive benchmark for Persian text embeddings covering 7 tasks and 60+ datasets.

[Learn more →](https://arxiv.org/abs/2502.11571)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | PersianFoodSentimentClassification | Classification | ['text']  |
    | SynPerChatbotConvSAClassification | Classification | ['text']  |
    | SynPerChatbotConvSAToneChatbotClassification | Classification | ['text']  |
    | SynPerChatbotConvSAToneUserClassification | Classification | ['text']  |
    | SynPerChatbotSatisfactionLevelClassification | Classification | ['text']  |
    | SynPerChatbotRAGToneChatbotClassification | Classification | ['text']  |
    | SynPerChatbotRAGToneUserClassification | Classification | ['text']  |
    | SynPerChatbotToneChatbotClassification | Classification | ['text']  |
    | SynPerChatbotToneUserClassification | Classification | ['text']  |
    | SynPerTextToneClassification | Classification | ['text']  |
    | SIDClassification | Classification | ['text']  |
    | DeepSentiPers | Classification | ['text']  |
    | PersianTextEmotion | Classification | ['text']  |
    | SentimentDKSF | Classification | ['text']  |
    | NLPTwitterAnalysisClassification | Classification | ['text']  |
    | DigikalamagClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | BeytooteClustering | Clustering | ['text']  |
    | DigikalamagClustering | Clustering | ['text']  |
    | HamshahriClustring | Clustering | ['text']  |
    | NLPTwitterAnalysisClustering | Clustering | ['text']  |
    | SIDClustring | Clustering | ['text']  |
    | FarsTail | PairClassification | ['text']  |
    | CExaPPC | PairClassification | ['text']  |
    | SynPerChatbotRAGFAQPC | PairClassification | ['text']  |
    | FarsiParaphraseDetection | PairClassification | ['text']  |
    | SynPerTextKeywordsPC | PairClassification | ['text']  |
    | SynPerQAPC | PairClassification | ['text']  |
    | ParsinluEntail | PairClassification | ['text']  |
    | ParsinluQueryParaphPC | PairClassification | ['text']  |
    | MIRACLReranking | Reranking | ['text']  |
    | WikipediaRerankingMultilingual | Reranking | ['text']  |
    | SynPerQARetrieval | Retrieval | ['text']  |
    | SynPerChatbotTopicsRetrieval | Retrieval | ['text']  |
    | SynPerChatbotRAGTopicsRetrieval | Retrieval | ['text']  |
    | SynPerChatbotRAGFAQRetrieval | Retrieval | ['text']  |
    | PersianWebDocumentRetrieval | Retrieval | ['text']  |
    | WikipediaRetrievalMultilingual | Retrieval | ['text']  |
    | MIRACLRetrieval | Retrieval | ['text']  |
    | ClimateFEVER-Fa | Retrieval | ['text']  |
    | DBPedia-Fa | Retrieval | ['text']  |
    | HotpotQA-Fa | Retrieval | ['text']  |
    | MSMARCO-Fa | Retrieval | ['text']  |
    | NQ-Fa | Retrieval | ['text']  |
    | ArguAna-Fa | Retrieval | ['text']  |
    | CQADupstackRetrieval-Fa | Retrieval | ['text']  |
    | FiQA2018-Fa | Retrieval | ['text']  |
    | NFCorpus-Fa | Retrieval | ['text']  |
    | QuoraRetrieval-Fa | Retrieval | ['text']  |
    | SCIDOCS-Fa | Retrieval | ['text']  |
    | SciFact-Fa | Retrieval | ['text']  |
    | TRECCOVID-Fa | Retrieval | ['text']  |
    | Touche2020-Fa | Retrieval | ['text']  |
    | Farsick | STS | ['text']  |
    | SynPerSTS | STS | ['text']  |
    | Query2Query | STS | ['text']  |
    | SAMSumFa | BitextMining | ['text']  |
    | SynPerChatbotSumSRetrieval | BitextMining | ['text']  |
    | SynPerChatbotRAGSumSRetrieval | BitextMining | ['text']  |
    


###  MTEB(fra, v1)

MTEB-French, a French expansion of the original benchmark with high-quality native French datasets.

[Learn more →](https://arxiv.org/abs/2405.20468)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AmazonReviewsClassification | Classification | ['text']  |
    | MasakhaNEWSClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | MTOPDomainClassification | Classification | ['text']  |
    | MTOPIntentClassification | Classification | ['text']  |
    | AlloProfClusteringP2P | Clustering | ['text']  |
    | AlloProfClusteringS2S | Clustering | ['text']  |
    | HALClusteringS2S | Clustering | ['text']  |
    | MasakhaNEWSClusteringP2P | Clustering | ['text']  |
    | MasakhaNEWSClusteringS2S | Clustering | ['text']  |
    | MLSUMClusteringP2P | Clustering | ['text']  |
    | MLSUMClusteringS2S | Clustering | ['text']  |
    | PawsXPairClassification | PairClassification | ['text']  |
    | AlloprofReranking | Reranking | ['text']  |
    | SyntecReranking | Reranking | ['text']  |
    | AlloprofRetrieval | Retrieval | ['text']  |
    | BSARDRetrieval | Retrieval | ['text']  |
    | MintakaRetrieval | Retrieval | ['text']  |
    | SyntecRetrieval | Retrieval | ['text']  |
    | XPQARetrieval | Retrieval | ['text']  |
    | SICKFr | STS | ['text']  |
    | STSBenchmarkMultilingualSTS | STS | ['text']  |
    | SummEvalFr | Summarization | ['text']  |
    | STS22 | STS | ['text']  |
    


###  MTEB(jpn, v1)

JMTEB is a benchmark for evaluating Japanese text embedding models.

[Learn more →](https://github.com/sbintuitions/JMTEB)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | LivedoorNewsClustering.v2 | Clustering | ['text']  |
    | MewsC16JaClustering | Clustering | ['text']  |
    | AmazonReviewsClassification | Classification | ['text']  |
    | AmazonCounterfactualClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | JSTS | STS | ['text']  |
    | JSICK | STS | ['text']  |
    | PawsXPairClassification | PairClassification | ['text']  |
    | JaqketRetrieval | Retrieval | ['text']  |
    | MrTidyRetrieval | Retrieval | ['text']  |
    | JaGovFaqsRetrieval | Retrieval | ['text']  |
    | NLPJournalTitleAbsRetrieval | Retrieval | ['text']  |
    | NLPJournalAbsIntroRetrieval | Retrieval | ['text']  |
    | NLPJournalTitleIntroRetrieval | Retrieval | ['text']  |
    | ESCIReranking | Reranking | ['text']  |
    


###  MTEB(kor, v1)

A benchmark and leaderboard for evaluation of text embedding in Korean.

??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | KLUE-TC | Classification | ['text']  |
    | MIRACLReranking | Reranking | ['text']  |
    | MIRACLRetrieval | Retrieval | ['text']  |
    | Ko-StrategyQA | Retrieval | ['text']  |
    | KLUE-STS | STS | ['text']  |
    | KorSTS | STS | ['text']  |
    


###  MTEB(pol, v1)

Polish Massive Text Embedding Benchmark (PL-MTEB), a comprehensive benchmark for text embeddings in Polish. The PL-MTEB consists of 28 diverse NLP
tasks from 5 task types. With tasks adapted based on previously used datasets by the Polish
NLP community. In addition, a new PLSC (Polish Library of Science Corpus) dataset was created
consisting of titles and abstracts of scientific publications in Polish, which was used as the basis for
two novel clustering tasks.

[Learn more →](https://arxiv.org/abs/2405.10138)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | AllegroReviews | Classification | ['text']  |
    | CBD | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | PolEmo2.0-IN | Classification | ['text']  |
    | PolEmo2.0-OUT | Classification | ['text']  |
    | PAC | Classification | ['text']  |
    | EightTagsClustering | Clustering | ['text']  |
    | PlscClusteringS2S | Clustering | ['text']  |
    | PlscClusteringP2P | Clustering | ['text']  |
    | CDSC-E | PairClassification | ['text']  |
    | PpcPC | PairClassification | ['text']  |
    | PSC | PairClassification | ['text']  |
    | SICK-E-PL | PairClassification | ['text']  |
    | CDSC-R | STS | ['text']  |
    | SICK-R-PL | STS | ['text']  |
    | STS22 | STS | ['text']  |
    


###  MTEB(rus, v1)

A Russian version of the Massive Text Embedding Benchmark with a number of novel Russian tasks in all task categories of the original MTEB.

[Learn more →](https://aclanthology.org/2023.eacl-main.148/)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | GeoreviewClassification | Classification | ['text']  |
    | HeadlineClassification | Classification | ['text']  |
    | InappropriatenessClassification | Classification | ['text']  |
    | KinopoiskClassification | Classification | ['text']  |
    | MassiveIntentClassification | Classification | ['text']  |
    | MassiveScenarioClassification | Classification | ['text']  |
    | RuReviewsClassification | Classification | ['text']  |
    | RuSciBenchGRNTIClassification | Classification | ['text']  |
    | RuSciBenchOECDClassification | Classification | ['text']  |
    | GeoreviewClusteringP2P | Clustering | ['text']  |
    | RuSciBenchGRNTIClusteringP2P | Clustering | ['text']  |
    | RuSciBenchOECDClusteringP2P | Clustering | ['text']  |
    | CEDRClassification | MultilabelClassification | ['text']  |
    | SensitiveTopicsClassification | MultilabelClassification | ['text']  |
    | TERRa | PairClassification | ['text']  |
    | MIRACLReranking | Reranking | ['text']  |
    | RuBQReranking | Reranking | ['text']  |
    | MIRACLRetrieval | Retrieval | ['text']  |
    | RiaNewsRetrieval | Retrieval | ['text']  |
    | RuBQRetrieval | Retrieval | ['text']  |
    | RUParaPhraserSTS | STS | ['text']  |
    | STS22 | STS | ['text']  |
    | RuSTSBenchmarkSTS | STS | ['text']  |
    


###  NanoBEIR

A benchmark to evaluate with subsets of BEIR datasets to use less computational power

[Learn more →](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | NanoArguAnaRetrieval | Retrieval | ['text']  |
    | NanoClimateFeverRetrieval | Retrieval | ['text']  |
    | NanoDBPediaRetrieval | Retrieval | ['text']  |
    | NanoFEVERRetrieval | Retrieval | ['text']  |
    | NanoFiQA2018Retrieval | Retrieval | ['text']  |
    | NanoHotpotQARetrieval | Retrieval | ['text']  |
    | NanoMSMARCORetrieval | Retrieval | ['text']  |
    | NanoNFCorpusRetrieval | Retrieval | ['text']  |
    | NanoNQRetrieval | Retrieval | ['text']  |
    | NanoQuoraRetrieval | Retrieval | ['text']  |
    | NanoSCIDOCSRetrieval | Retrieval | ['text']  |
    | NanoSciFactRetrieval | Retrieval | ['text']  |
    | NanoTouche2020Retrieval | Retrieval | ['text']  |
    


###  R2MED

R2MED: First Reasoning-Driven Medical Retrieval Benchmark.
    R2MED is a high-quality, high-resolution information retrieval (IR) dataset designed for medical scenarios.
    It contains 876 queries with three retrieval tasks, five medical scenarios, and twelve body systems.
    

[Learn more →](https://r2med.github.io/)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | R2MEDBiologyRetrieval | Retrieval | ['text']  |
    | R2MEDBioinformaticsRetrieval | Retrieval | ['text']  |
    | R2MEDMedicalSciencesRetrieval | Retrieval | ['text']  |
    | R2MEDMedXpertQAExamRetrieval | Retrieval | ['text']  |
    | R2MEDMedQADiagRetrieval | Retrieval | ['text']  |
    | R2MEDPMCTreatmentRetrieval | Retrieval | ['text']  |
    | R2MEDPMCClinicalRetrieval | Retrieval | ['text']  |
    | R2MEDIIYiClinicalRetrieval | Retrieval | ['text']  |
    


###  RAR-b

A benchmark to evaluate reasoning capabilities of retrievers.

[Learn more →](https://arxiv.org/abs/2404.06347)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | ARCChallenge | Retrieval | ['text']  |
    | AlphaNLI | Retrieval | ['text']  |
    | HellaSwag | Retrieval | ['text']  |
    | WinoGrande | Retrieval | ['text']  |
    | PIQA | Retrieval | ['text']  |
    | SIQA | Retrieval | ['text']  |
    | Quail | Retrieval | ['text']  |
    | SpartQA | Retrieval | ['text']  |
    | TempReasonL1 | Retrieval | ['text']  |
    | TempReasonL2Pure | Retrieval | ['text']  |
    | TempReasonL2Fact | Retrieval | ['text']  |
    | TempReasonL2Context | Retrieval | ['text']  |
    | TempReasonL3Pure | Retrieval | ['text']  |
    | TempReasonL3Fact | Retrieval | ['text']  |
    | TempReasonL3Context | Retrieval | ['text']  |
    | RARbCode | Retrieval | ['text']  |
    | RARbMath | Retrieval | ['text']  |
    


###  RuSciBench

RuSciBench is a benchmark designed for evaluating sentence encoders and language models on scientific texts in both Russian and English. The data is sourced from eLibrary (www.elibrary.ru), Russia's largest electronic library of scientific publications. This benchmark facilitates the evaluation and comparison of models on various research-related tasks.

[Learn more →](https://link.springer.com/article/10.1134/S1064562424602191)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | RuSciBenchBitextMining | BitextMining | ['text']  |
    | RuSciBenchCoreRiscClassification | Classification | ['text']  |
    | RuSciBenchGRNTIClassification.v2 | Classification | ['text']  |
    | RuSciBenchOECDClassification.v2 | Classification | ['text']  |
    | RuSciBenchPubTypeClassification | Classification | ['text']  |
    | RuSciBenchCiteRetrieval | Retrieval | ['text']  |
    | RuSciBenchCociteRetrieval | Retrieval | ['text']  |
    | RuSciBenchCitedCountRegression | Regression | ['text']  |
    | RuSciBenchYearPublRegression | Regression | ['text']  |
    


###  VN-MTEB (vie, v1)

A benchmark for text-embedding performance in Vietnamese.

[Learn more →](https://arxiv.org/abs/2507.21500)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | ArguAna-VN | Retrieval | ['text']  |
    | SciFact-VN | Retrieval | ['text']  |
    | ClimateFEVER-VN | Retrieval | ['text']  |
    | FEVER-VN | Retrieval | ['text']  |
    | DBPedia-VN | Retrieval | ['text']  |
    | NQ-VN | Retrieval | ['text']  |
    | HotpotQA-VN | Retrieval | ['text']  |
    | MSMARCO-VN | Retrieval | ['text']  |
    | TRECCOVID-VN | Retrieval | ['text']  |
    | FiQA2018-VN | Retrieval | ['text']  |
    | NFCorpus-VN | Retrieval | ['text']  |
    | SCIDOCS-VN | Retrieval | ['text']  |
    | Touche2020-VN | Retrieval | ['text']  |
    | Quora-VN | Retrieval | ['text']  |
    | CQADupstackAndroid-VN | Retrieval | ['text']  |
    | CQADupstackGis-VN | Retrieval | ['text']  |
    | CQADupstackMathematica-VN | Retrieval | ['text']  |
    | CQADupstackPhysics-VN | Retrieval | ['text']  |
    | CQADupstackProgrammers-VN | Retrieval | ['text']  |
    | CQADupstackStats-VN | Retrieval | ['text']  |
    | CQADupstackTex-VN | Retrieval | ['text']  |
    | CQADupstackUnix-VN | Retrieval | ['text']  |
    | CQADupstackWebmasters-VN | Retrieval | ['text']  |
    | CQADupstackWordpress-VN | Retrieval | ['text']  |
    | Banking77VNClassification | Classification | ['text']  |
    | EmotionVNClassification | Classification | ['text']  |
    | AmazonCounterfactualVNClassification | Classification | ['text']  |
    | MTOPDomainVNClassification | Classification | ['text']  |
    | TweetSentimentExtractionVNClassification | Classification | ['text']  |
    | ToxicConversationsVNClassification | Classification | ['text']  |
    | ImdbVNClassification | Classification | ['text']  |
    | MTOPIntentVNClassification | Classification | ['text']  |
    | MassiveScenarioVNClassification | Classification | ['text']  |
    | MassiveIntentVNClassification | Classification | ['text']  |
    | AmazonReviewsVNClassification | Classification | ['text']  |
    | AmazonPolarityVNClassification | Classification | ['text']  |
    | SprintDuplicateQuestions-VN | PairClassification | ['text']  |
    | TwitterSemEval2015-VN | PairClassification | ['text']  |
    | TwitterURLCorpus-VN | PairClassification | ['text']  |
    | TwentyNewsgroupsClustering-VN | Clustering | ['text']  |
    | RedditClusteringP2P-VN | Clustering | ['text']  |
    | StackExchangeClusteringP2P-VN | Clustering | ['text']  |
    | StackExchangeClustering-VN | Clustering | ['text']  |
    | RedditClustering-VN | Clustering | ['text']  |
    | SciDocsRR-VN | Reranking | ['text']  |
    | AskUbuntuDupQuestions-VN | Reranking | ['text']  |
    | StackOverflowDupQuestions-VN | Reranking | ['text']  |
    | BIOSSES-VN | STS | ['text']  |
    | SICK-R-VN | STS | ['text']  |
    | STSBenchmark-VN | STS | ['text']  |
    


###  ViDoRe(v1)

Retrieve associated pages according to questions.

[Learn more →](https://arxiv.org/abs/2407.01449)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | VidoreArxivQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreDocVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreInfoVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTabfquadRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTatdqaRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreShiftProjectRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAAIRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAEnergyRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAGovernmentReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAHealthcareIndustryRetrieval | DocumentUnderstanding | ['text', 'image']  |
    


###  ViDoRe(v2)

Retrieve associated pages according to questions.

[Learn more →](https://arxiv.org/abs/2407.01449)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | Vidore2ESGReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2EconomicsReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2BioMedicalLecturesRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2ESGReportsHLRetrieval | DocumentUnderstanding | ['text', 'image']  |
    


###  VisualDocumentRetrieval

A benchmark for evaluating visual document retrieval, combining ViDoRe v1 and v2.

[Learn more →](https://arxiv.org/abs/2407.01449)



??? info Tasks

    | Task| type  | modalities  |
    | ---| ---| --- |
    | VidoreArxivQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreDocVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreInfoVQARetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTabfquadRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreTatdqaRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreShiftProjectRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAAIRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAEnergyRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAGovernmentReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | VidoreSyntheticDocQAHealthcareIndustryRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2ESGReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2EconomicsReportsRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2BioMedicalLecturesRetrieval | DocumentUnderstanding | ['text', 'image']  |
    | Vidore2ESGReportsHLRetrieval | DocumentUnderstanding | ['text', 'image']  |
<!-- END TASK DESCRIPTION -->
