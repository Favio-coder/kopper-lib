import pandas as pd
from kopeer import compute_pairs, optimize_weights, normalize_df

# Datos de ejemplo
students = {
    "Lucia":   [9, 8, 9, 8],
    "Marcos":  [5, 6, 7, 6],
    "Valeria": [7, 5, 6, 9],
    "Diego":   [3, 4, 8, 5],
    "Camila":  [6, 7, 5, 7]
}

df = pd.DataFrame(students).T
df.columns = ["c","p","m","i"]

# Normalizar
df_norm = normalize_df(df)

# Optimizar pesos
weights = optimize_weights(df_norm, iterations=100)

# Generar pares
pairs = compute_pairs(df_norm, weights)

print("Pesos óptimos:", weights)
print(pairs)