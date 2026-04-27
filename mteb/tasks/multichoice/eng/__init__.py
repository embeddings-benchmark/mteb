from .avmeme_exam import AVMemeExamVideoAudioCentricQA, AVMemeExamVideoCentricQA
from .blink_it2i_multi_choice import BLINKIT2IMultiChoice
from .blink_it2t_multi_choice import BLINKIT2TMultiChoice
from .cv_bench import CVBenchCount, CVBenchDepth, CVBenchDistance, CVBenchRelation
from .egoschema import EgoSchemaVideoCentricQA
from .nextqa import NExTQAVideoCentricQA

__all__ = [
    "AVMemeExamVideoAudioCentricQA",
    "AVMemeExamVideoCentricQA",
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "CVBenchCount",
    "CVBenchDepth",
    "CVBenchDistance",
    "CVBenchRelation",
    "EgoSchemaVideoCentricQA",
    "NExTQAVideoCentricQA",
]
