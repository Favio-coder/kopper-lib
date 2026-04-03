import numpy as np 

# Función para calcular la similaridad de dos estudiantes
def similarity(a,b, weights):
    return 1 - np.sum(weights * np.abs(a-b))


# Sigmoide 
def sigmoid(x):
    return 1 /  (1 + np.exp(-x))

# Penalización de emparejamiento 
def gaussian_penalty(delta, mu=0.3, beta=5):
    return np.exp(-beta * (delta - mu)**2)

# Core: Algoritmo de emparejamiento 
def tutoring_score_vectorized(X_i, X_j, w, alpha=0.7):
    delta = np.abs(X_i - X_j)  # diferencia absoluta
    weighted_delta = delta * w  # aplicar pesos
    sum_delta = np.sum(weighted_delta, axis=2)
    
    # Score entre 0 y 1 (similitud)
    scores = 1 - sum_delta
    

    scores = 1 / (1 + np.exp(-scores))  # ejemplo sigmoid para más penalidad
    
    return scores

def tutoring_score(a_i, a_j, w, alpha=0.7):
    delta = np.abs(a_i - a_j)
    weighted_delta = delta * w
    sum_delta = np.sum(weighted_delta)
    score = 1 - sum_delta
    score = 1 / (1 + np.exp(-score))
    return score


    