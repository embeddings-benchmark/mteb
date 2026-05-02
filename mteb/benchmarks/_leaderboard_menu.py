from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mteb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.benchmarks import Benchmark


@dataclass
class MenuEntry:
    """A menu entry for the benchmark selector.

    Attributes:
        name: The name of the menu entry.
        benchmarks: A list of benchmarks or nested menu entries.
        description: An optional description of the menu entry.
        open: Whether the accordion is open by default.
        size: The size of the buttons. Can be "sm" or "md".
    """

    name: str | None
    benchmarks: Sequence[Benchmark | MenuEntry]
    description: str | None = None
    open: bool = False
    size: str = "sm"


GP_BENCHMARK_ENTRIES = [
    MenuEntry(
        name="General Purpose",
        description="",
        open=False,
        benchmarks=mteb.get_benchmarks(
            ["MTEB(Multilingual, v2)", "MTEB(eng, v2)", "HUME(v1)"]
        )
        + [
            MenuEntry(
                "Image",
                mteb.get_benchmarks(
                    [
                        "MIEB(Multilingual)",
                        "MIEB(eng)",
                        "MIEB(lite)",
                        "MIEB(Img)",
                    ]
                ),
            ),
            MenuEntry(
                "Audio",
                mteb.get_benchmarks(
                    [
                        "MAEB(beta)",
                        "MAEB(beta, audio-only)",
                    ]
                ),
            ),
            MenuEntry(
                "Domain-Specific ",
                mteb.get_benchmarks(
                    [
                        "MTEB(Code, v1)",
                        "MTEB(Law, v1)",
                        "MTEB(Medical, v1)",
                        "ChemTEB",
                    ]
                ),
            ),
            MenuEntry(
                "Language-specific",
                mteb.get_benchmarks(
                    [
                        "MTEB(Europe, v1)",
                        "MTEB(Indic, v1)",
                        "MTEB(Scandinavian, v1)",
                        "MTEB(cmn, v1)",
                        "MTEB(deu, v1)",
                        "MTEB(fra, v1)",
                        "JMTEB(v2)",
                        "MTEB(kor, v1)",
                        "MTEB(nld, v1)",
                        "MTEB(pol, v1)",
                        "MTEB(rus, v1.1)",
                        "MTEB(tha, v1)",
                        "MTEB(fas, v2)",
                        "VN-MTEB (vie, v1)",
                        "MTEB(spa, v1)",
                    ]
                )
                + [
                    MenuEntry(
                        "Other",
                        mteb.get_benchmarks(
                            [
                                "MTEB(eng, v1)",
                                "MTEB(fas, v1)",
                                "MTEB(rus, v1)",
                                "MTEB(jpn, v1)",
                            ]
                        ),
                    )
                ],
            ),
            MenuEntry(
                "Miscellaneous",  # All of these are retrieval benchmarks
                mteb.get_benchmarks(
                    [
                        "BuiltBench(eng)",
                        "MINERSBitextMining",
                    ]
                ),
            ),
        ],
    ),
]

R_BENCHMARK_ENTRIES = [
    MenuEntry(
        name="Retrieval",
        description=None,
        open=False,
        benchmarks=[
            mteb.get_benchmark("RTEB(beta)"),
            mteb.get_benchmark("RTEB(eng, beta)"),
            MenuEntry(
                "Image",
                description=None,
                open=True,
                benchmarks=[
                    mteb.get_benchmark("ViDoRe(v3)"),
                    mteb.get_benchmark("JinaVDR"),
                    MenuEntry("Other", [mteb.get_benchmark("ViDoRe(v1&v2)")]),
                ],
            ),
            MenuEntry(
                "Domain-Specific",
                description=None,
                open=False,
                benchmarks=[
                    mteb.get_benchmark("RTEB(fin, beta)"),
                    mteb.get_benchmark("RTEB(Law, beta)"),
                    mteb.get_benchmark("RTEB(Code, beta)"),
                    mteb.get_benchmark("CoIR"),
                    mteb.get_benchmark("RTEB(Health, beta)"),
                    mteb.get_benchmark("FollowIR"),
                    mteb.get_benchmark("LongEmbed"),
                    mteb.get_benchmark("BRIGHT(v1.1)"),
                ],
            ),
            MenuEntry(
                "Language-specific",
                description=None,
                open=False,
                benchmarks=[
                    mteb.get_benchmark("RTEB(fra, beta)"),
                    mteb.get_benchmark("RTEB(deu, beta)"),
                    mteb.get_benchmark("RTEB(jpn, beta)"),
                    mteb.get_benchmark("BEIR"),
                    mteb.get_benchmark("BEIR-NL"),
                ],
            ),
            MenuEntry(
                "Miscellaneous",
                mteb.get_benchmarks(
                    [
                        "NanoBEIR",
                        "BRIGHT",
                        "BRIGHT (long)",
                        "RAR-b",
                    ]
                ),
            ),
        ],
    )
]
