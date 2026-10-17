"""Domain-specialized benchmarks (chemistry, built environment, law, medical)."""

from mteb.benchmarks.benchmark import Benchmark
from mteb.get_tasks import get_tasks

MTEB_RETRIEVAL_LAW = Benchmark(
    # This benchmark is likely in the need of an update
    name="MTEB(Law, v1)",
    aliases=["MTEB(law)"],
    display_name="Legal",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-library.svg",
    tasks=get_tasks(
        tasks=[
            "AILACasedocs",
            "AILAStatutes",
            "LegalSummarization",
            "GerDaLIRSmall",
            "LeCaRDv2",
            "LegalBenchConsumerContractsQA",
            "LegalBenchCorporateLobbying",
            "LegalQuAD",
        ]
    ),
    description="Legal document retrieval across case documents, statutes, legal Q&A, and legal summarization in multiple languages.",
    reference=None,
    citation=None,
)

MTEB_RETRIEVAL_MEDICAL = Benchmark(
    name="MTEB(Medical, v1)",
    aliases=["MTEB(Medical)"],
    display_name="Medical",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-hospital.svg",
    tasks=get_tasks(
        tasks=[
            "CUREv1",
            "NFCorpus",
            "TRECCOVID",
            "TRECCOVID-PL",
            "SciFact",
            "SciFact-PL",
            "MedicalQARetrieval",
            "PublicHealthQA",
            "MedrxivClusteringP2P.v2",
            "MedrxivClusteringS2S.v2",
            "CmedqaRetrieval",
            "CMedQAv2-reranking",
        ],
    ),
    description="Medical information retrieval across clinical, biomedical, and consumer health domains, spanning retrieval, reranking, and clustering tasks.",
    reference=None,
    citation=None,
)

CHEMTEB = Benchmark(
    name="ChemTEB",
    aliases=["ChemTEB(v1)"],
    display_name="Chemical",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-purge.svg",
    tasks=get_tasks(
        tasks=[
            "PubChemSMILESBitextMining",
            "SDSEyeProtectionClassification",
            "SDSGlovesClassification",
            "WikipediaBioMetChemClassification",
            "WikipediaGreenhouseEnantiopureClassification",
            "WikipediaSolidStateColloidalClassification",
            "WikipediaOrganicInorganicClassification",
            "WikipediaCryobiologySeparationClassification",
            "WikipediaChemistryTopicsClassification",
            "WikipediaTheoreticalAppliedClassification",
            "WikipediaChemFieldsClassification",
            "WikipediaLuminescenceClassification",
            "WikipediaIsotopesFissionClassification",
            "WikipediaSaltsSemiconductorsClassification",
            "WikipediaBiolumNeurochemClassification",
            "WikipediaCrystallographyAnalyticalClassification",
            "WikipediaCompChemSpectroscopyClassification",
            "WikipediaChemEngSpecialtiesClassification",
            "WikipediaChemistryTopicsClustering",
            "WikipediaSpecialtiesInChemistryClustering",
            "PubChemAISentenceParaphrasePC",
            "PubChemSMILESPC",
            "PubChemSynonymPC",
            "PubChemWikiParagraphsPC",
            "PubChemWikiPairClassification",
            "ChemNQRetrieval",
            "ChemHotpotQARetrieval",
        ],
    ),
    description="Chemical domain text embedding quality across bitext mining, classification, clustering, pair classification, and retrieval.",
    reference="https://arxiv.org/abs/2412.00532",
    citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}
""",
)

CHEMTEB_V1_1 = Benchmark(
    name="ChemTEB(v1.1)",
    aliases=["ChemTEB(latest)"],
    display_name="Chemical",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-purge.svg",
    tasks=get_tasks(
        tasks=[
            "PubChemSMILESBitextMining",
            "SDSEyeProtectionClassification",
            "SDSGlovesClassification",
            "WikipediaBioMetChemClassification",
            "WikipediaGreenhouseEnantiopureClassification",
            "WikipediaSolidStateColloidalClassification",
            "WikipediaOrganicInorganicClassification",
            "WikipediaCryobiologySeparationClassification",
            "WikipediaChemistryTopicsClassification",
            "WikipediaTheoreticalAppliedClassification",
            "WikipediaChemFieldsClassification",
            "WikipediaLuminescenceClassification",
            "WikipediaIsotopesFissionClassification",
            "WikipediaSaltsSemiconductorsClassification",
            "WikipediaBiolumNeurochemClassification",
            "WikipediaCrystallographyAnalyticalClassification",
            "WikipediaCompChemSpectroscopyClassification",
            "WikipediaChemEngSpecialtiesClassification",
            "WikipediaChemistryTopicsClustering",
            "WikipediaSpecialtiesInChemistryClustering",
            "PubChemAISentenceParaphrasePC",
            "PubChemSMILESPC",
            "PubChemSynonymPC",
            "PubChemWikiParagraphsPC",
            "PubChemWikiPairClassification",
            "ChemNQRetrieval",
            "ChemHotpotQARetrieval",
            "ChemRxivRetrieval",
        ],
    ),
    description="Chemical domain text embedding quality across bitext mining, classification, clustering, pair classification, and retrieval. v1.1 adds the ChemRxivRetrieval task.",
    reference="https://arxiv.org/abs/2412.00532",
    citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}

@article{kasmaee2025chembed,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Astaraki, Mahdi and Saloot, Mohammad Arshi and Sherck, Nicholas and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2508.01643},
  title = {Chembed: Enhancing chemical literature search through domain-specific text embeddings},
  year = {2025},
}
""",
)

BUILT_MTEB = Benchmark(
    name="BuiltBench(eng)",
    tasks=get_tasks(
        tasks=[
            "BuiltBenchClusteringP2P",
            "BuiltBenchClusteringS2S",
            "BuiltBenchRetrieval",
            "BuiltBenchReranking",
        ],
    ),
    description="Text embedding quality in the built environment domain across clustering, retrieval, and reranking, spanning architecture, engineering, construction, and operations management.",
    reference="https://arxiv.org/abs/2411.12056",
    citation=r"""
@article{shahinmoghadam2024benchmarking,
  author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
  journal = {arXiv preprint arXiv:2411.12056},
  title = {Benchmarking pre-trained text embedding models in aligning built asset information},
  year = {2024},
}
""",
    contacts=["mehrzadshm"],
)

R2MED = Benchmark(
    name="R2MED",
    display_name="Reasoning-driven medical retrieval",
    tasks=get_tasks(
        tasks=[
            "R2MEDBiologyRetrieval",
            "R2MEDBioinformaticsRetrieval",
            "R2MEDMedicalSciencesRetrieval",
            "R2MEDMedXpertQAExamRetrieval",
            "R2MEDMedQADiagRetrieval",
            "R2MEDPMCTreatmentRetrieval",
            "R2MEDPMCClinicalRetrieval",
            "R2MEDIIYiClinicalRetrieval",
        ]
    ),
    description="Reasoning-driven medical retrieval quality across biology, bioinformatics, medical sciences, clinical, and treatment scenarios, requiring models to perform multi-step reasoning over medical literature.",
    reference="https://r2med.github.io/",
    citation=r"""
@misc{zhang2026r2medbenchmarkreasoningdrivenmedical,
  archiveprefix = {arXiv},
  author = {Xiangxu Zhang and Lei Li and Xiao Zhou and Zheng Liu},
  eprint = {2505.14558},
  primaryclass = {cs.IR},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://arxiv.org/abs/2505.14558},
  year = {2026},
}
""",
)
