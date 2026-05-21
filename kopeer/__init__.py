from .engine import PeerTutoringEngine
from .model import PeerTutoringModel, StudentProfile, LightNeuralNetwork
from .config import KopeerConfig
from .utils import normalize_df, extract_notas_from_alumnos  # ← Agregar extract_notas_from_alumnos

__version__ = "0.2.1"  # ← Cambiar versión
__all__ = [
    'PeerTutoringEngine',
    'PeerTutoringModel', 
    'StudentProfile',
    'LightNeuralNetwork',
    'KopeerConfig',
    'normalize_df',
    'extract_notas_from_alumnos'  # ← Agregar a __all__
]