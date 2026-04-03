import numpy as np 
from .model import tutoring_score_vectorized
from .model import tutoring_score
# Función para generar pesos aleatorios 
def random_weights():
    w = np.random.rand(4) #Flontas de 4 

    return w / np.sum(w)

# Optimizador de pesos 
def optimize_weights(df_norm, iterations = 1000, alpha = 0.7):

    best_score = -1
    best_weights = None
    names = df_norm.index.tolist()

    # Eficiencia de o al cuadrado, mejorar para próxima versión
    for _ in range(iterations):
        w = random_weights()
        scores = []

        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    a_i = df_norm.iloc[i].values
                    a_j = df_norm.iloc[j].values

                    s = tutoring_score(a_i, a_j, w, alpha)

                    if s > 0: 
                        scores.append(s)
            
        if scores: 
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_weights = w
    
    return best_weights



