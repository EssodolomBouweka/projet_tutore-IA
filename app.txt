
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def impute_rf(df, group_col=None, min_samples=5):
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        if df[col].isna().sum() == 0:
            continue

        known = df_imputed[df_imputed[col].notna()]
        unknown = df_imputed[df_imputed[col].isna()]

        if known.empty or unknown.empty:
            continue

        # Features = toutes les colonnes numériques sauf la cible
        features = [c for c in numeric_cols if c != col]

        if group_col and group_col in df.columns:
            for group_val in unknown[group_col].dropna().unique():
                group_known = known[known[group_col] == group_val]
                group_unknown = unknown[unknown[group_col] == group_val]

                if len(group_known) >= min_samples:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(group_known[features], group_known[col])
                    preds = model.predict(group_unknown[features])
                    df_imputed.loc[group_unknown.index, col] = preds
        else:
            if len(known) >= min_samples:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(known[features], known[col])
                preds = model.predict(unknown[features])
                df_imputed.loc[unknown.index, col] = preds

    return df_imputed
