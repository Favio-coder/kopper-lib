"""Configuración dinámica para adaptarse a diferentes números de atributos"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class KopeerConfig:
    """Configuración flexible para el modelo de peer tutoring"""
    
    # Nombres de los campos de atributos
    atributos: List[str] = field(default_factory=lambda: [
        'conocimiento', 'actitud', 'participacion', 'interes', 'estrella'
    ])
    
    # Pesos iniciales por defecto (se optimizarán)
    pesos_iniciales: Optional[List[float]] = None
    
    # Hiperparámetros
    alpha: float = 0.7  # Factor de balance para sigmoide
    learning_rate: float = 0.01  # Tasa de aprendizaje para red neuronal
    epochs: int = 100  # Épocas de entrenamiento
    hidden_size: int = 32  # Tamaño de capa oculta (red neuronal ligera)
    
    # Umbrales
    min_score: float = 0.3  # Score mínimo para considerar emparejamiento
    max_score: float = 0.95  # Score máximo (evita sobreajuste)
    
    @property
    def n_atributos(self) -> int:
        return len(self.atributos)
    
    def get_pesos_iniciales(self) -> List[float]:
        if self.pesos_iniciales is None:
            # Pesos uniformes
            return [1.0 / self.n_atributos] * self.n_atributos
        return self.pesos_iniciales