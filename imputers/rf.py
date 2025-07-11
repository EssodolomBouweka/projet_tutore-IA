import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def impute_rf(df, group_col=None, min_samples=5):
    df_imputed = df.copy()
    all_cols = df.select_dtypes(include=[np.number]).columns

    for col in all_cols:
        # Si pas de NaN dans cette colonne, on continue
        if df[col].isna().sum() == 0:
            continue

        known = df_imputed[df_imputed[col].notna()]
        unknown = df_imputed[df_imputed[col].isna()]

        if known.empty or unknown.empty:
            continue

        # Features = toutes les colonnes sauf la cible et group_col
        features = [c for c in all_cols if c != col and (group_col is None or c != group_col)]
        features = [f for f in features if df_imputed[f].isna().sum() == 0]  # éviter les NaN dans les features

        if not features:
            continue

        # Choisir le bon modèle
        is_numeric = np.issubdtype(df[col].dtype, np.floating) or np.issubdtype(df[col].dtype, np.integer)

        model_class = RandomForestRegressor if is_numeric else RandomForestClassifier

        # Imputation par groupe si demandé
        if group_col and group_col in df.columns:
            for group_val in unknown[group_col].dropna().unique():
                group_known = known[known[group_col] == group_val]
                group_unknown = unknown[unknown[group_col] == group_val]

                if len(group_known) >= min_samples:
                    model = model_class(n_estimators=100, random_state=42)
                    model.fit(group_known[features], group_known[col])
                    preds = model.predict(group_unknown[features])
                    df_imputed.loc[group_unknown.index, col] = preds
        else:
            if len(known) >= min_samples:
                model = model_class(n_estimators=100, random_state=42)
                model.fit(known[features], known[col])
                preds = model.predict(unknown[features])
                df_imputed.loc[unknown.index, col] = preds

    return df_imputed
