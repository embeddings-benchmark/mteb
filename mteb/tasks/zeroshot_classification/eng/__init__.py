from .birdsnap import BirdsnapZeroShotClassification
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
from .imagenet1k import Imagenet1kZeroShotClassification
from .mnist import MNISTZeroShotClassification
from .oxford_pets import OxfordPetsZeroShotClassification
from .patch_camelyon import PatchCamelyonZeroShotClassification
from .rendered_sst2 import RenderedSST2
from .resisc45 import RESISC45ZeroShotClassification
from .sci_mmir import SciMMIR
from .stanford_cars import StanfordCarsZeroShotClassification
from .stl10 import STL10ZeroShotClassification
from .sun397 import SUN397ZeroShotClassification
from .ucf101 import UCF101ZeroShotClassification

__all__ = [
    "CLEVR",
    "BirdsnapZeroShotClassification",
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
    "Imagenet1kZeroShotClassification",
    "MNISTZeroShotClassification",
    "OxfordPetsZeroShotClassification",
    "PatchCamelyonZeroShotClassification",
    "RESISC45ZeroShotClassification",
    "RenderedSST2",
    "STL10ZeroShotClassification",
    "SUN397ZeroShotClassification",
    "SciMMIR",
    "StanfordCarsZeroShotClassification",
    "UCF101ZeroShotClassification",
]
