
from sklearn.impute import KNNImputer
import pandas as pd

def impute_knn(df, n_neighbors=5):
    df_imputed = df.copy()

    # On sélectionne uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        return df  # Rien à imputer

    # On applique KNN uniquement aux colonnes numériques
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df_imputed
