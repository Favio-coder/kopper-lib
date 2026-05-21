"""Motor principal de peer tutoring"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .model import PeerTutoringModel, StudentProfile
from .config import KopeerConfig


class PeerTutoringEngine:
    """Motor principal para recomendación de pares tutor-alumno"""
    
    def __init__(self, config: Optional[KopeerConfig] = None):
        self.config = config or KopeerConfig()
        self.model: Optional[PeerTutoringModel] = None
        self.students: List[StudentProfile] = []
        self._is_initialized = False
    
    def load_students(self, df: pd.DataFrame, id_col: str = 'c_usua', 
                      nombre_col: str = 'nombre_completo') -> None:
        """Carga estudiantes desde DataFrame"""
        self.students = []
        
        for _, row in df.iterrows():
            atributos = {}
            for attr in self.config.atributos:
                # Buscar la nota en las columnas anidadas o directas
                if 'notas' in row and isinstance(row['notas'], dict):
                    atributos[attr] = row['notas'].get(attr, 0.0)
                elif attr in row:
                    atributos[attr] = row[attr] if pd.notna(row[attr]) else 0.0
                else:
                    atributos[attr] = 0.0
            
            self.students.append(StudentProfile(
                id=str(row[id_col]),
                nombre=row[nombre_col],
                atributos=atributos
            ))
        
        self._is_initialized = True
        print(f"✅ Cargados {len(self.students)} estudiantes")
    
    def train(self, iterations: int = None) -> None:
        """Entrena el modelo con los estudiantes cargados"""
        if not self._is_initialized or not self.students:
            raise ValueError("No hay estudiantes cargados. Llama a load_students() primero.")
        
        # Convertir a DataFrame normalizado
        data = []
        for student in self.students:
            row = [student.atributos.get(attr, 0.0) for attr in self.config.atributos]
            data.append(row)
        
        df_norm = pd.DataFrame(data, columns=self.config.atributos)
        
        # Normalizar (asumiendo que las notas están en escala 0-20)
        df_norm = df_norm / 20.0  # Normalizar a 0-1
        
        # Crear y entrenar modelo
        self.model = PeerTutoringModel(self.config)
        self.model.train(df_norm, iterations)
    
    def get_recommendations(self, student_id: Optional[str] = None, 
                           top_n: int = 10) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones de pares tutor-alumno"""
        
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        all_pairs = self.model.predict_all_pairs(self.students)
        
        if student_id:
            # Filtrar por estudiante específico
            filtered = [p for p in all_pairs if p['tutor_id'] == student_id or p['alumno_id'] == student_id]
            return filtered[:top_n]
        
        return all_pairs[:top_n]
    
    def get_best_tutor_for_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Encuentra el mejor tutor para un estudiante específico"""
        recommendations = self.get_recommendations(student_id=student_id, top_n=1)
        
        if recommendations:
            return recommendations[0]
        return None
    
    def get_best_match_for_class(self) -> List[Dict[str, Any]]:
        """Obtiene los mejores emparejamientos para toda la clase"""
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        all_pairs = self.model.predict_all_pairs(self.students)
        
        # Algoritmo greedy para emparejamientos únicos
        used_tutors = set()
        used_alumnos = set()
        matches = []
        
        for pair in all_pairs:
            if (pair['tutor_id'] not in used_tutors and 
                pair['alumno_id'] not in used_alumnos and
                pair['tutor_id'] != pair['alumno_id']):
                matches.append(pair)
                used_tutors.add(pair['tutor_id'])
                used_alumnos.add(pair['alumno_id'])
            
            if len(matches) >= len(self.students) // 2:
                break
        
        return matches
    
    def export_results(self, results: List[Dict[str, Any]], format: str = 'json') -> Any:
        """Exporta resultados en diferentes formatos"""
        if format == 'json':
            return results
        elif format == 'dataframe':
            return pd.DataFrame(results)
        else:
            raise ValueError(f"Formato {format} no soportado")