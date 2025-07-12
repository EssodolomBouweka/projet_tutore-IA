import pandas as pd

# Imputation par la moyenne
def impute_mean(df):
    return df.fillna(df.mean(numeric_only=True))

# Imputation par la médiane
def impute_median(df):
    return df.fillna(df.median(numeric_only=True))

# Imputation par le mode
def impute_mode(df):
    return df.fillna(df.mode().iloc[0])

# Imputation par une valeur constante pour les colonnes numériques
def impute_constant_numeric(df, value=0):
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(value)
    return df

# Imputation par une valeur constante pour les colonnes catégorielles
def impute_constant_categorical(df, value="Inconnu"):
    df = df.copy()
    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna(value)
    return df

# Suppression des lignes avec au moins 60% de valeurs manquantes
def delete_rows(df):
    """
    Supprime les lignes contenant 60% ou plus de valeurs manquantes.
    """
    nan_ratio_per_row = df.isna().mean(axis=1)
    rows_to_drop = nan_ratio_per_row[nan_ratio_per_row >= 0.6].index
    return df.drop(index=rows_to_drop)

# Suppression des colonnes avec au moins 60% de valeurs manquantes
def delete_columns(df):
    """
    Supprime les colonnes contenant 60% ou plus de valeurs manquantes.
    """
    nan_ratio_per_col = df.isna().mean()
    cols_to_drop = nan_ratio_per_col[nan_ratio_per_col >= 0.6].index
    return df.drop(columns=cols_to_drop)

