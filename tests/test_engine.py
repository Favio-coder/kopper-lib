import pandas as pd 
from kopeer.engine import compute_pairs
from kopeer.optimizer import optimize_weights
from kopeer.utils import normalize_df

def test_compute_pairs():
    students = {
        "Lucia":   [9, 8, 9, 8],
        "Marcos":  [5, 6, 7, 6],
        "Valeria": [7, 5, 6, 9],
        "Diego":   [3, 4, 8, 5],
        "Camila":  [6, 7, 5, 7]
    }
    df = pd.DataFrame(students).T
    df.columns = ["c","p","m","i"]
    df_norm = normalize_df(df)
    weights = optimize_weights(df_norm, iterations=100)
    df_pairs = compute_pairs(df_norm, weights)
    assert not df_pairs.empty
    assert "Score" in df_pairs.columns