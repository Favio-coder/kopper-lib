"""Utilidades para procesamiento de datos"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

def normalize_df(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                 max_value: float = 20.0) -> pd.DataFrame:
    """Normaliza DataFrame a escala 0-1"""
    if columns is None:
        columns = df.columns.tolist()
    
    df_norm = df.copy()
    df_norm[columns] = df_norm[columns] / max_value
    
    return df_norm

def extract_notas_from_alumnos(alumnos_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extrae notas de la estructura de datos de alumnos"""
    rows = []
    
    for alumno in alumnos_data:
        row = {
            'c_usua': alumno.get('c_usua'),
            'nombre_completo': alumno.get('nombre_completo'),
            'email': alumno.get('email')
        }
        
        notas = alumno.get('notas', {})
        if notas:
            row['conocimiento'] = notas.get('conocimiento', 0)
            row['actitud'] = notas.get('actitud', 0)
            row['participacion'] = notas.get('participacion', 0)
            row['interes'] = notas.get('interes', 0)
            row['estrella'] = notas.get('estrella', 0)
        else:
            row['conocimiento'] = 0
            row['actitud'] = 0
            row['participacion'] = 0
            row['interes'] = 0
            row['estrella'] = 0
        
        rows.append(row)
    
    return pd.DataFrame(rows)

# utils.py - Versión alternativa sin import circular
def create_config_from_atributos(atributos_list: List[str]):
    """Crea configuración dinámicamente desde lista de atributos"""
    # Importar inline para evitar circular dependency
    try:
        from .config import KopeerConfig
        return KopeerConfig(atributos=atributos_list)
    except ImportError:
        # Fallback - crear diccionario en lugar de clase
        from dataclasses import dataclass, field
        
        @dataclass
        class TempConfig:
            atributos: List[str] = field(default_factory=lambda: atributos_list)
            alpha: float = 0.7
            learning_rate: float = 0.01
            epochs: int = 100
            hidden_size: int = 32
            min_score: float = 0.3
            max_score: float = 0.95
            
            @property
            def n_atributos(self) -> int:
                return len(self.atributos)
        
        return TempConfig()