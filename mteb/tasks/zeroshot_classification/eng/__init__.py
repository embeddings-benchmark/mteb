from .birdsnap import BirdsnapZeroShotClassification
from .breakfast_classification import BreakfastZeroShotClassification
from .caltech101 import Caltech101ZeroShotClassification
from .cifar import CIFAR10ZeroShotClassification, CIFAR100ZeroShotClassification
from .clevr import CLEVR, CLEVRCount
from .country211 import Country211ZeroShotClassification
from .dtd import DTDZeroShotClassification
from .euro_sat import EuroSATZeroShotClassification
from .fer2013 import FER2013ZeroShotClassification
from .fgvc_aircraft import FGVCAircraftZeroShotClassification
from .food101 import Food101ZeroShotClassification
from .gtsrb import GTSRBZeroShotClassification
from .human_animal_cartoon import HumanAnimalCartoonZeroShotClassification
from .imagenet1k import Imagenet1kZeroShotClassification
from .kinetics400 import (
    Kinetics400VAZeroShotClassification,
    Kinetics400ZeroShotClassification,
)
from .meld_classification import (
    MELDAudioVideoZeroShotClassification,
    MELDVideoZeroShotClassification,
)
from .mnist import MNISTZeroShotClassification
from .oxford_pets import OxfordPetsZeroShotClassification
from .patch_camelyon import PatchCamelyonZeroShotClassification
from .ravdess import RavdessZeroshotClassification
from .rendered_sst2 import RenderedSST2
from .resisc45 import RESISC45ZeroShotClassification
from .sci_mmir import SciMMIR
from .speech_commands import (
    SpeechCommandsZeroshotClassificationV01,
    SpeechCommandsZeroshotClassificationv02,
)
from .stanford_cars import StanfordCarsZeroShotClassification
from .stl10 import STL10ZeroShotClassification
from .sun397 import SUN397ZeroShotClassification
from .ucf101 import (
    UCF101VideoAudioZeroShotClassification,
    UCF101VideoZeroShotClassification,
    UCF101ZeroShotClassification,
)
from .worldsense_classification import (
    WorldSenseAudioVideoZeroShotClassification,
    WorldSenseVideoZeroShotClassification,
)

__all__ = [
    "CLEVR",
    "BirdsnapZeroShotClassification",
    "BreakfastZeroShotClassification",
    "CIFAR10ZeroShotClassification",
    "CIFAR100ZeroShotClassification",
    "CLEVRCount",
    "Caltech101ZeroShotClassification",
    "Country211ZeroShotClassification",
    "DTDZeroShotClassification",
    "EuroSATZeroShotClassification",
    "FER2013ZeroShotClassification",
    "FGVCAircraftZeroShotClassification",
    "Food101ZeroShotClassification",
    "GTSRBZeroShotClassification",
    "HumanAnimalCartoonZeroShotClassification",
    "Imagenet1kZeroShotClassification",
    "Kinetics400VAZeroShotClassification",
    "Kinetics400ZeroShotClassification",
    "MELDAudioVideoZeroShotClassification",
    "MELDVideoZeroShotClassification",
    "MNISTZeroShotClassification",
    "OxfordPetsZeroShotClassification",
    "PatchCamelyonZeroShotClassification",
    "RESISC45ZeroShotClassification",
    "RavdessZeroshotClassification",
    "RenderedSST2",
    "STL10ZeroShotClassification",
    "SUN397ZeroShotClassification",
    "SciMMIR",
    "SpeechCommandsZeroshotClassificationV01",
    "SpeechCommandsZeroshotClassificationv02",
    "StanfordCarsZeroShotClassification",
    "UCF101VideoAudioZeroShotClassification",
    "UCF101VideoZeroShotClassification",
    "UCF101ZeroShotClassification",
    "WorldSenseAudioVideoZeroShotClassification",
    "WorldSenseVideoZeroShotClassification",
]
