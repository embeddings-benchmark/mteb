"""General-purpose retrieval benchmarks (BEIR, BRIGHT, NanoBEIR, long-context, instruction-following, ...)."""

from mteb.benchmarks.benchmark import (
    Benchmark,
    BenchmarkAggregation,
    CustomGroup,
    CustomGrouping,
)
from mteb.get_tasks import MTEBTasks, get_task, get_tasks

MTEB_RETRIEVAL_WITH_INSTRUCTIONS = Benchmark(
    name="FollowIR",
    aliases=["MTEB(Retrieval w/Instructions)"],
    display_name="Instruction Following",
    tasks=get_tasks(
        tasks=[
            "Robust04InstructionRetrieval",
            "News21InstructionRetrieval",
            "Core17InstructionRetrieval",
        ]
    ),
    description="Instruction-following retrieval quality, measuring how well models retrieve relevant documents when given detailed natural language instructions alongside queries.",
    reference="https://arxiv.org/abs/2403.15246",
    citation=r"""
@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
}
""",
)

MTEB_RETRIEVAL_WITH_DOMAIN_INSTRUCTIONS = Benchmark(
    name="IFIR",
    display_name="IFIR",
    tasks=get_tasks(
        tasks=[
            "IFIRAila",
            "IFIRCds",
            "IFIRFiQA",
            "IFIRFire",
            "IFIRNFCorpus",
            "IFIRPm",
            "IFIRScifact",
        ]
    ),
    description="Instruction-following retrieval quality in expert domains including healthcare, finance, and legal, where queries are paired with domain-specific natural language instructions.",
    reference="https://arxiv.org/abs/2503.04644",
    citation=r"""
@inproceedings{song2025ifir,
  author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {10186--10204},
  title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
  year = {2025},
}
""",
)

RAR_b = Benchmark(
    name="RAR-b",
    display_name="Reasoning as retrieval",
    tasks=get_tasks(
        tasks=[
            "ARCChallenge",
            "AlphaNLI",
            "HellaSwag",
            "WinoGrande",
            "PIQA",
            "SIQA",
            "Quail",
            "SpartQA",
            "TempReasonL1",
            "TempReasonL2Pure",
            "TempReasonL2Fact",
            "TempReasonL2Context",
            "TempReasonL3Pure",
            "TempReasonL3Fact",
            "TempReasonL3Context",
            "RARbCode",
            "RARbMath",
        ]
    ),
    description="Reasoning capabilities of retrieval models, framing commonsense, temporal, and domain-specific reasoning tasks as retrieval problems.",
    reference="https://arxiv.org/abs/2404.06347",
    citation=r"""
@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Al Moubayed, Noura},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
    contacts=["gowitheflow-1998"],
)

_LONG_EMBED_LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _long_embed_length_group(n: int) -> CustomGroup:
    split = f"test_{n}"
    label = f"{n // 1024}K" if n >= 1024 else str(n)
    return CustomGroup(
        label=label,
        tasks=[
            get_task("LEMBNeedleRetrieval", eval_splits=[split]),
            get_task("LEMBPasskeyRetrieval", eval_splits=[split]),
        ],
        description=f"Needle- and passkey-retrieval accuracy at a {n}-token context length.",
    )


LONG_EMBED = Benchmark(
    name="LongEmbed",
    display_name="Long-context Retrieval",
    tasks=get_tasks(
        tasks=[
            "LEMBNarrativeQARetrieval",
            "LEMBNeedleRetrieval",
            "LEMBPasskeyRetrieval",
            "LEMBQMSumRetrieval",
            "LEMBSummScreenFDRetrieval",
            "LEMBWikimQARetrieval",
        ],
    ),
    description="Long-context retrieval quality across synthetic and real-world tasks featuring documents of varying length with dispersed target information.",
    reference="https://arxiv.org/abs/2404.12096v2",
    citation=r"""
@article{zhu2024longembed,
  author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
  journal = {arXiv preprint arXiv:2404.12096},
  title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
  year = {2024},
}
""",
    aggregations=(
        BenchmarkAggregation.MEAN_TASK,
        BenchmarkAggregation.MEAN_TASK_TYPE,
        BenchmarkAggregation.TASK_TYPES,
        CustomGrouping(
            name="Context Length",
            groups=tuple(_long_embed_length_group(n) for n in _LONG_EMBED_LENGTHS),
        ),
    ),
)

LMEB = Benchmark(
    name="LMEB",
    display_name="Long-Horizon Memory",
    tasks=get_tasks(
        tasks=[
            "EPBench",
            "KnowMeBench",
            "LoCoMo",
            "LongMemEval",
            "REALTALK",
            "TMD",
            "MemBench",
            "ConvoMem",
            "QASPER",
            "NovelQA",
            "PeerQA",
            "CovidQA",
            "ESGReports",
            "LMEBMLDR",
            "LooGLE",
            "LMEB_SciFact",
            "Gorilla",
            "ToolBench",
            "ReMe",
            "ProceduralMemBench",
            "MemGovern",
            "DeepPlanning",
        ]
    ),
    description="Long-horizon memory retrieval quality across episodic, dialogue, semantic, and procedural retrieval tasks, measuring how well embedding models retrieve evidence in long-term memory scenarios.",
    reference="https://arxiv.org/abs/2603.12572",
    citation=r"""
@misc{zhao2026lmeb,
  archiveprefix = {arXiv},
  author = {Zhao, Xinping and Hu, Xinshuo and Xu, Jiaxin and Tang, Danyu and Zhang, Xin and Zhou, Mengjia and Zhong, Yan and Zhou, Yao and Shan, Zifei and Zhang, Meishan and Hu, Baotian and Zhang, Min},
  eprint = {2603.12572},
  primaryclass = {cs.CL},
  title = {LMEB: Long-horizon Memory Embedding Benchmark},
  url = {https://arxiv.org/abs/2603.12572},
  year = {2026},
}
""",
    aggregations=(
        BenchmarkAggregation.MEAN_TASK,
        CustomGrouping(
            name="Memory Type",
            groups=(
                CustomGroup(
                    label="Episodic",
                    tasks=get_tasks(["EPBench", "KnowMeBench"]),
                    description="Episodic memory retrieval aims to recall past events "
                    "grounded in temporal cues, entities, contents, and spatial context.",
                ),
                CustomGroup(
                    label="Dialogue",
                    tasks=get_tasks(
                        [
                            "LoCoMo",
                            "LongMemEval",
                            "REALTALK",
                            "TMD",
                            "MemBench",
                            "ConvoMem",
                        ]
                    ),
                    description="Dialogue memory retrieval aims to maintain context "
                    "across multi-turn interactions by recalling relevant dialogue "
                    "history and user preference.",
                ),
                CustomGroup(
                    label="Semantic",
                    tasks=get_tasks(
                        [
                            "QASPER",
                            "NovelQA",
                            "PeerQA",
                            "CovidQA",
                            "ESGReports",
                            "LMEBMLDR",
                            "LooGLE",
                            "LMEB_SciFact",
                        ]
                    ),
                    description="Semantic memory retrieval focuses on recalling general "
                    "knowledge and concepts that are largely independent of time or "
                    "specific context.",
                ),
                CustomGroup(
                    label="Procedural",
                    tasks=get_tasks(
                        [
                            "Gorilla",
                            "ToolBench",
                            "ReMe",
                            "ProceduralMemBench",
                            "MemGovern",
                            "DeepPlanning",
                        ]
                    ),
                    description="Procedural memory retrieval focuses on recalling "
                    "learned skills, action patterns, and structured procedures that "
                    "guide task execution and multi-step reasoning.",
                ),
            ),
        ),
    ),
)

BRIGHT = Benchmark(
    name="BRIGHT",
    display_name="Reasoning Retrieval",
    tasks=get_tasks(tasks=["BrightRetrieval"], eval_splits=["standard"]),
    description="Reasoning-intensive retrieval quality across real-world queries spanning diverse domains including economics, psychology, mathematics, and coding, drawn from naturally occurring and carefully curated human data.",
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
    superseded_by=["BRIGHT(v1.1)"],
)

BRIGHT_LONG = Benchmark(
    name="BRIGHT (long)",
    aliases=["BRIGHT(long)"],
    tasks=MTEBTasks(
        (
            get_task(
                "BrightLongRetrieval",
            ),
        )
    ),
    description="Reasoning-intensive retrieval quality across real-world queries spanning diverse domains, filtered to longer documents to stress-test models on extended contexts.",
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
    superseded_by=["BRIGHT(v1.1)"],
)

_BRIGHT_SHORT_TASKS = [
    "BrightBiologyRetrieval",
    "BrightEarthScienceRetrieval",
    "BrightEconomicsRetrieval",
    "BrightPsychologyRetrieval",
    "BrightRoboticsRetrieval",
    "BrightStackoverflowRetrieval",
    "BrightSustainableLivingRetrieval",
    "BrightPonyRetrieval",
    "BrightLeetcodeRetrieval",
    "BrightAopsRetrieval",
    "BrightTheoremQATheoremsRetrieval",
    "BrightTheoremQAQuestionsRetrieval",
]

_BRIGHT_LONG_TASKS = [
    "BrightBiologyLongRetrieval",
    "BrightEarthScienceLongRetrieval",
    "BrightEconomicsLongRetrieval",
    "BrightPsychologyLongRetrieval",
    "BrightRoboticsLongRetrieval",
    "BrightStackoverflowLongRetrieval",
    "BrightSustainableLivingLongRetrieval",
    "BrightPonyLongRetrieval",
]

BRIGHT_V1_1 = Benchmark(
    name="BRIGHT(v1.1)",
    display_name="Reasoning Retrieval",
    tasks=get_tasks(tasks=[*_BRIGHT_SHORT_TASKS, *_BRIGHT_LONG_TASKS]),
    description="Reasoning-intensive retrieval quality across real-world queries spanning diverse domains including economics, psychology, mathematics, and coding. v1.1 restructures tasks into separate datasets and adds per-task prompts.",
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
    benchmark_hf_repo="mteb/BRIGHT",
    aggregations=(
        BenchmarkAggregation.MEAN_TASK,
        CustomGrouping(
            name="Document Length",
            groups=(
                CustomGroup(
                    label="Short",
                    tasks=get_tasks(_BRIGHT_SHORT_TASKS),
                    description="Standard-length BRIGHT documents.",
                ),
                CustomGroup(
                    label="Long",
                    tasks=get_tasks(_BRIGHT_LONG_TASKS),
                    description="Documents filtered to longer context lengths, stress-testing retrieval over extended documents.",
                ),
            ),
        ),
    ),
)

BEIR = Benchmark(
    name="BEIR",
    display_name="BEIR",
    icon="https://github.com/lipis/flag-icons/raw/refs/heads/main/flags/4x3/us.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "TRECCOVID",
                "NFCorpus",
                "NQ",
                "HotpotQA",
                "FiQA2018",
                "ArguAna",
                "Touche2020",
                "CQADupstackRetrieval",
                "QuoraRetrieval",
                "DBPedia",
                "SCIDOCS",
                "FEVER",
                "ClimateFEVER",
                "SciFact",
            ],
        )
        + get_tasks(tasks=["MSMARCO"], languages=["eng"], eval_splits=["dev"])
    ),
    description="Zero-shot retrieval quality across a heterogeneous set of IR tasks and domains, providing a common framework for comparing NLP-based retrieval models.",
    reference="https://arxiv.org/abs/2104.08663",
    citation=r"""
@inproceedings{thakur2021beir,
  archiveprefix = {arXiv},
  author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  eprint = {2104.08663},
  title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
  year = {2021},
}
""",
)

NANOBEIR = Benchmark(
    name="NanoBEIR",
    tasks=get_tasks(
        tasks=[
            "NanoArguAnaRetrieval",
            "NanoClimateFeverRetrieval",
            "NanoDBPediaRetrieval",
            "NanoFEVERRetrieval",
            "NanoFiQA2018Retrieval",
            "NanoHotpotQARetrieval",
            "NanoMSMARCORetrieval",
            "NanoNFCorpusRetrieval",
            "NanoNQRetrieval",
            "NanoQuoraRetrieval",
            "NanoSCIDOCSRetrieval",
            "NanoSciFactRetrieval",
            "NanoTouche2020Retrieval",
        ],
    ),
    description="Zero-shot retrieval quality using subsets of the BEIR datasets, designed for faster evaluation with reduced computational cost.",
    reference="https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6",
    citation=None,
)

NANOBEIR_EXTENDED = Benchmark(
    name="NanoBEIR-multilingual",
    display_name="Multilingual NanoBEIR",
    tasks=get_tasks(
        tasks=[
            "MultilingualNanoArguAnaRetrieval",
            "MultilingualNanoClimateFeverRetrieval",
            "MultilingualNanoDBPediaRetrieval",
            "MultilingualNanoFEVERRetrieval",
            "MultilingualNanoFiQA2018Retrieval",
            "MultilingualNanoHotpotQARetrieval",
            "MultilingualNanoMSMARCORetrieval",
            "MultilingualNanoNFCorpusRetrieval",
            "MultilingualNanoNQRetrieval",
            "MultilingualNanoQuoraRetrieval",
            "MultilingualNanoSCIDOCSRetrieval",
            "MultilingualNanoSciFactRetrieval",
            "MultilingualNanoTouche2020Retrieval",
        ],
    ),
    description="A translated benchmark targeting zero-shot retrieval quality using translated subsets of the BEIR datasets, designed for faster evaluation during training with reduced computational cost.",
    reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
    citation=None,
)

BRIGHT_PRO = Benchmark(
    name="BRIGHT-Pro",
    display_name="BRIGHT-Pro",
    tasks=get_tasks(
        tasks=[
            "BrightProBiologyRetrieval",
            "BrightProEarthScienceRetrieval",
            "BrightProEconomicsRetrieval",
            "BrightProPsychologyRetrieval",
            "BrightProRoboticsRetrieval",
            "BrightProStackoverflowRetrieval",
            "BrightProSustainableLivingRetrieval",
        ],
    ),
    description="""Reasoning-intensive retrieval quality on real-world StackExchange questions, scored against multi-aspect gold evidence. Each query is paired with a long-form reference answer whose cited passages collectively cover several reasoning aspects, so retrievers are measured on surfacing an aspect-diverse evidence set rather than a single relevant passage. Extends BRIGHT with these aspect annotations for agentic search settings.""",
    reference="https://huggingface.co/datasets/yale-nlp/Bright-Pro",
    citation=r"""
@article{Zhao2026RethinkingRR,
  author = {Yilun Zhao and Jinbiao Wei and Tingyu Song and Siyue Zhang and Chen Zhao and Arman Cohan},
  journal = {arXiv preprint arXiv:2605.04018},
  title = {Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems},
  year = {2026},
}
""",
)
