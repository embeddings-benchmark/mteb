from __future__ import annotations

from mteb.tasks.Audio.Clustering.eng.AmbientAcousticContextClustering import (
    AmbientAcousticContextClustering,
)
from mteb.tasks.Audio.Clustering.eng.ESC50Clustering import ESC50Clustering
from mteb.tasks.Audio.Clustering.eng.MusicGenre import MusicGenreClustering
from mteb.tasks.Audio.Clustering.eng.TUTAcousticScenesClustering import (
    TUTAcousticScenesClustering,
)
from mteb.tasks.Audio.Clustering.eng.VehicleSoundClustering import (
    VehicleSoundClustering,
)
from mteb.tasks.Audio.Clustering.eng.VoiceGender import VoiceGenderClustering

__all__ = [
    "ESC50Clustering",
    "TUTAcousticScenesClustering",
    "AmbientAcousticContextClustering",
    "MusicGenreClustering",
    "VehicleSoundClustering",
    "VoiceGenderClustering",
]
