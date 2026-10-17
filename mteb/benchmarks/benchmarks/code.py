"""Code embedding and code retrieval benchmarks."""

from mteb.benchmarks.benchmark import Benchmark
from mteb.benchmarks.benchmarks.text_language import MMTEB_CITATION
from mteb.get_tasks import get_tasks

CoIR = Benchmark(
    name="CoIR",
    display_name="Code Information Retrieval",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            "AppsRetrieval",
            "CodeFeedbackMT",
            "CodeFeedbackST",
            "CodeSearchNetCCRetrieval",
            "CodeTransOceanContest",
            "CodeTransOceanDL",
            "CosQA",
            "COIRCodeSearchNetRetrieval",
            "StackOverflowQA",
            "SyntheticText2SQL",
        ]
    ),
    description="Code information retrieval across diverse programming languages and coding tasks, including code search, question answering, and text-to-SQL retrieval.",
    reference="https://github.com/CoIR-team/coir",
    citation=r"""
@misc{li2024coircomprehensivebenchmarkcode,
  archiveprefix = {arXiv},
  author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
  eprint = {2407.02883},
  primaryclass = {cs.IR},
  title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
  url = {https://arxiv.org/abs/2407.02883},
  year = {2024},
}
""",
)

MTEB_code = Benchmark(
    name="MTEB(Code, v1)",
    aliases=["MTEB(code)"],
    display_name="Code",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            # Retrieval
            "AppsRetrieval",
            "CodeEditSearchRetrieval",
            "CodeFeedbackMT",
            "CodeFeedbackST",
            "CodeSearchNetCCRetrieval",
            "CodeSearchNetRetrieval",
            "CodeTransOceanContest",
            "CodeTransOceanDL",
            "CosQA",
            "COIRCodeSearchNetRetrieval",
            "StackOverflowQA",
            "SyntheticText2SQL",
        ],
    ),
    description="Code retrieval quality across a wide range of popular programming languages, covering code search, text-to-SQL, and code feedback tasks.",
    reference=None,
    citation=MMTEB_CITATION,
)

CODE_RAG = Benchmark(
    name="CodeRAG",
    tasks=get_tasks(
        tasks=[
            "CodeRAGLibraryDocumentationSolutions",
            "CodeRAGOnlineTutorials",
            "CodeRAGProgrammingSolutions",
            "CodeRAGStackoverflowPosts",
        ],
    ),
    description="Code retrieval quality for retrieval-augmented generation, covering programming solutions, online tutorials, library documentation, and Stack Overflow posts.",
    reference="https://arxiv.org/abs/2406.14497",
    citation=r"""
@misc{wang2024coderagbenchretrievalaugmentcode,
  archiveprefix = {arXiv},
  author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
  eprint = {2406.14497},
  primaryclass = {cs.SE},
  title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
  url = {https://arxiv.org/abs/2406.14497},
  year = {2024},
}
""",
)

CoREB = Benchmark(
    name="CoREB(v1)",
    aliases=["CoREB"],
    display_name="CoREB",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            "CorebC2TRetrieval",
            "CorebC2CRetrieval",
            "CorebT2CRetrieval",
            "CorebC2TReranking",
            "CorebC2CReranking",
            "CorebT2CReranking",
        ],
    ),
    description="Code embedding and reranking quality across code-to-text, text-to-code, and code-to-code retrieval tasks, using counterfactually rewritten problems in five programming languages to limit training data contamination.",
    reference="https://arxiv.org/abs/2605.04615",
    citation=r"""
@article{xue2026coreb,
  author = {Xue, Siqiao and Liao, Zihan and Qin, Jin and Zhang, Ziyin and Mu, Yixiang and Zhou, Fan and Yu, Hang},
  journal = {arXiv preprint arXiv:2605.04615},
  title = {Beyond Retrieval: A Multitask Benchmark and Model for Code Search},
  url = {https://arxiv.org/abs/2605.04615},
  year = {2026},
}
""",
    contacts=["Geralt-Targaryen"],
)
