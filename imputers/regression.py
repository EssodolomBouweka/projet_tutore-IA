import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def impute_with_linear_regression(df):
    df = df.copy()
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        if numeric_df[column].isnull().sum() == 0:
            continue
        train = numeric_df[numeric_df[column].notnull()]
        test = numeric_df[numeric_df[column].isnull()]
        X_train = train.drop(columns=[column])
        y_train = train[column]
        X_test = test.drop(columns=[column])

        if X_train.empty or X_test.empty:
            continue

        imp = SimpleImputer(strategy="mean")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        df.loc[df[column].isnull(), column] = predicted

    return df
