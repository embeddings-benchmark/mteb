from .avmeme_exam import AVMemeExamVideoAudioCentricQA, AVMemeExamVideoCentricQA
from .avqa import AVQAVideoAudioCentricQA, AVQAVideoCentricQA
from .blink_it2i_multi_choice import BLINKIT2IMultiChoice
from .blink_it2t_multi_choice import BLINKIT2TMultiChoice
from .cv_bench import CVBenchCount, CVBenchDepth, CVBenchDistance, CVBenchRelation
from .daily_omni import DailyOmniVideoAudioCentricQA, DailyOmniVideoCentricQA
from .egoschema import EgoSchemaVideoCentricQA
from .nextqa import NExTQAVideoCentricQA
from .perception_test import (
    PerceptionTestVideoAudioCentricQA,
    PerceptionTestVideoCentricQA,
)
from .video_mme import VideoMMEShortVideoAudioCentricQA, VideoMMEShortVideoCentricQA

__all__ = [
    "AVMemeExamVideoAudioCentricQA",
    "AVMemeExamVideoCentricQA",
    "AVQAVideoAudioCentricQA",
    "AVQAVideoCentricQA",
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "CVBenchCount",
    "CVBenchDepth",
    "CVBenchDistance",
    "CVBenchRelation",
    "DailyOmniVideoAudioCentricQA",
    "DailyOmniVideoCentricQA",
    "EgoSchemaVideoCentricQA",
    "NExTQAVideoCentricQA",
    "PerceptionTestVideoAudioCentricQA",
    "PerceptionTestVideoCentricQA",
    "VideoMMEShortVideoAudioCentricQA",
    "VideoMMEShortVideoCentricQA",
]
