from .bird_clef_clustering import BirdCLEFSpeciesClustering
from .esc50_clustering import ESC50Clustering
from .gtzan_genre_clustering import GTZANGenreClustering
from .music_genre import MusicGenreClustering
from .n_synth_clustering import NSynthInstrumentFamilyClustering
from .tau_acoustic_scenes_2022_mobile_clustering import (
    TAUAcousticScenes2022MobileClustering,
)
from .urban_sound8k_clustering import UrbanSound8kClustering
from .vehicle_sound_clustering import VehicleSoundClustering
from .vim_sketch_clustering import VimSketchImitationClustering

__all__ = [
    "BirdCLEFSpeciesClustering",
    "ESC50Clustering",
    "GTZANGenreClustering",
    "MusicGenreClustering",
    "NSynthInstrumentFamilyClustering",
    "TAUAcousticScenes2022MobileClustering",
    "UrbanSound8kClustering",
    "VehicleSoundClustering",
    "VimSketchImitationClustering",
]
