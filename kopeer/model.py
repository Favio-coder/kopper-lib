"""Modelo de peer tutoring con redes neuronales ligeras"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd

# Red neuronal ligera (2 capas)
class LightNeuralNetwork:
    """Red neuronal pequeña para aprendizaje adaptativo"""
    
    def __init__(self, input_size: int, hidden_size: int = 32, learning_rate: float = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Inicialización Xavier/Glorot
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
        
        self._cache = {}  # Cache para backpropagation
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self._cache['Z1'] = X @ self.W1 + self.b1
        self._cache['A1'] = np.tanh(self._cache['Z1'])  # Tanh para mejor gradiente
        self._cache['Z2'] = self._cache['A1'] @ self.W2 + self.b2
        self._cache['A2'] = self.sigmoid(self._cache['Z2'])
        return self._cache['A2']
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward pass"""
        m = X.shape[0]
        
        # Gradiente de salida
        dZ2 = output - y
        dW2 = (self._cache['A1'].T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Gradiente de capa oculta
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - np.power(self._cache['A1'], 2))  # Derivada de tanh
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Actualizar pesos
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train_step(self, X: np.ndarray, y: np.ndarray):
        """Un paso de entrenamiento"""
        output = self.forward(X)
        self.backward(X, y, output)
        return np.mean((output - y) ** 2)  # MSE loss


@dataclass
class StudentProfile:
    """Perfil de estudiante con sus notas"""
    id: str
    nombre: str
    atributos: Dict[str, float]  # conocimiento, actitud, etc.
    
    def to_vector(self, atributos_order: List[str]) -> np.ndarray:
        """Convierte el perfil a vector numérico"""
        return np.array([self.atributos.get(attr, 0.0) for attr in atributos_order])


class PeerTutoringModel:
    """Modelo principal de peer tutoring"""
    
    def __init__(self, config: Any):
        self.config = config
        self.nn_model: Optional[LightNeuralNetwork] = None
        self.pesos: Optional[np.ndarray] = None
        self._is_trained = False
    
    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud entre dos estudiantes"""
        if self.pesos is not None:
            weighted_diff = np.abs(a - b) * self.pesos
        else:
            weighted_diff = np.abs(a - b)
        
        similarity = 1 - np.sum(weighted_diff)
        
        # Aplicar sigmoide para normalizar entre 0 y 1
        score = 1 / (1 + np.exp(-self.config.alpha * (similarity * 10 - 5)))
        
        return np.clip(score, 0, 1)
    
    def _compute_similarity_vectorized(self, X_i: np.ndarray, X_j: np.ndarray) -> np.ndarray:
        """Versión vectorizada para múltiples pares"""
        delta = np.abs(X_i - X_j)
        
        if self.pesos is not None:
            weighted_delta = delta * self.pesos
        else:
            weighted_delta = delta
        
        sum_delta = np.sum(weighted_delta, axis=2) if weighted_delta.ndim > 2 else np.sum(weighted_delta, axis=1)
        similarity = 1 - sum_delta
        
        # Aplicar sigmoide vectorizada
        scores = 1 / (1 + np.exp(-self.config.alpha * (similarity * 10 - 5)))
        
        return np.clip(scores, 0, 1)
    
    def train(self, df_norm: pd.DataFrame, iterations: int = None):
        """Entrena el modelo usando optimización de pesos + red neuronal"""
        
        X = df_norm.values  # shape (n_students, n_features)
        n = X.shape[0]
        
        if iterations is None:
            iterations = self.config.epochs
        
        # Fase 1: Optimización de pesos
        print("Fase 1: Optimizando pesos de atributos...")
        self.pesos = self._optimize_weights_global(X, iterations=iterations // 2)
        
        # Fase 2: Entrenar red neuronal ligera
        print("Fase 2: Entrenando red neuronal...")
        
        # Preparar datos de entrenamiento (todos los pares posibles)
        pairs_data = []
        pairs_labels = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim = self._compute_similarity(X[i], X[j])
                    pairs_data.append(np.concatenate([X[i], X[j]]))  # Concatena ambos perfiles
                    pairs_labels.append([sim])
        
        if pairs_data:
            X_train = np.array(pairs_data)
            y_train = np.array(pairs_labels)
            
            self.nn_model = LightNeuralNetwork(
                input_size=2 * self.config.n_atributos,
                hidden_size=self.config.hidden_size,
                learning_rate=self.config.learning_rate
            )
            
            # Entrenar
            for epoch in range(iterations // 2):
                loss = self.nn_model.train_step(X_train, y_train)
                if epoch % 50 == 0:
                    print(f"  Época {epoch}: Loss = {loss:.4f}")
        
        self._is_trained = True
        print("✅ Modelo entrenado correctamente!")
    
    def _optimize_weights_global(self, X: np.ndarray, iterations: int = 500) -> np.ndarray:
        """Optimización global de pesos usando búsqueda aleatoria mejorada"""
        n = X.shape[0]
        best_score = -1
        best_weights = None
        
        # Pesos iniciales uniformes
        base_weights = np.ones(self.config.n_atributos) / self.config.n_atributos
        
        for iteration in range(iterations):
            # Exploración con ruido gaussiano
            noise = np.random.normal(0, 0.1, self.config.n_atributos)
            w = base_weights + noise * (1 - iteration / iterations)  # Decaimiento de ruido
            w = np.maximum(w, 0)  # No negativos
            w = w / np.sum(w)  # Normalizar
            
            # Evaluar
            scores = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = np.abs(X[i] - X[j])
                        weighted_diff = diff * w
                        sim = 1 - np.sum(weighted_diff)
                        score = 1 / (1 + np.exp(-self.config.alpha * (sim * 10 - 5)))
                        if score > self.config.min_score:
                            scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_weights = w.copy()
                    base_weights = w.copy()
        
        return best_weights if best_weights is not None else base_weights
    
    def predict_score(self, tutor: StudentProfile, alumno: StudentProfile) -> float:
        """Predice el score de compatibilidad entre tutor y alumno"""
        if not self._is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        v1 = tutor.to_vector(self.config.atributos)
        v2 = alumno.to_vector(self.config.atributos)
        
        if self.nn_model is not None:
            # Usar red neuronal
            input_vector = np.concatenate([v1, v2]).reshape(1, -1)
            score = float(self.nn_model.forward(input_vector)[0, 0])
        else:
            # Usar método clásico
            score = self._compute_similarity(v1, v2)
        
        return np.clip(score, self.config.min_score, self.config.max_score)
    
    def predict_all_pairs(self, students: List[StudentProfile]) -> List[Dict[str, Any]]:
        """Predice scores para todos los pares posibles"""
        results = []
        
        n = len(students)
        for i in range(n):
            for j in range(n):
                if i != j:
                    score = self.predict_score(students[i], students[j])
                    if score >= self.config.min_score:
                        results.append({
                            'tutor_id': students[i].id,
                            'tutor_nombre': students[i].nombre,
                            'alumno_id': students[j].id,
                            'alumno_nombre': students[j].nombre,
                            'score': round(score, 4)
                        })
        
        # Ordenar por score descendente
        results.sort(key=lambda x: x['score'], reverse=True)
        return results