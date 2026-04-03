import pandas as pd
import numpy as np
from .model import tutoring_score_vectorized  # versión vectorizada de tutoring_score

def compute_pairs(df_norm, weights, alpha=0.7):
    # Nuevo código usando vectorización para mejorar la eifiencia

    names = df_norm.index.tolist()
    X = df_norm.values  # shape (n_students, n_features)
    w = np.array(weights)  # shape (n_features,)
    n = X.shape[0]

    # Broadcasting para generar todas las combinaciones i,j
    X_i = X[:, np.newaxis, :]  # shape (n,1,f)
    X_j = X[np.newaxis, :, :]  # shape (1,n,f)

    # Vectorizamos el cálculo de tutoring_score
    # Suponiendo que creamos una función que recibe delta_i_j (shape n,n,f) y devuelve scores
    scores_matrix = tutoring_score_vectorized(X_i, X_j, w, alpha)  # shape (n,n)

    # Evitar pares donde i == j
    np.fill_diagonal(scores_matrix, 0)

    # Convertir la matriz en DataFrame de resultados
    tutor_indices, alumno_indices = np.where(scores_matrix > 0)  # solo scores positivos
    scores = scores_matrix[tutor_indices, alumno_indices]

    results = pd.DataFrame({
        "Tutor": [names[i] for i in tutor_indices],
        "Alumno": [names[j] for j in alumno_indices],
        "Score": scores
    })

    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)