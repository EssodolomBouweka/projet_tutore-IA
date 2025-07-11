import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute_mice(df):
    df = df.copy()
    numeric_df = df.select_dtypes(include="number")
    imputer = IterativeImputer(random_state=0)
    df[numeric_df.columns] = imputer.fit_transform(numeric_df)
    return df
