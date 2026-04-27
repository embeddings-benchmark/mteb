from .avqa import AVQAVideoAudioCentricQA, AVQAVideoCentricQA
from .blink_it2i_multi_choice import BLINKIT2IMultiChoice
from .blink_it2t_multi_choice import BLINKIT2TMultiChoice
from .cv_bench import CVBenchCount, CVBenchDepth, CVBenchDistance, CVBenchRelation
from .nextqa import NExTQAVideoCentricQA

__all__ = [
    "AVQAVideoAudioCentricQA",
    "AVQAVideoCentricQA",
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "CVBenchCount",
    "CVBenchDepth",
    "CVBenchDistance",
    "CVBenchRelation",
    "NExTQAVideoCentricQA",
]
