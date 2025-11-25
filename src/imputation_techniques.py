import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import KNNImputer

def impute_mean(df: pd.DataFrame ):
    """Impute missing valuesusing the mean of the column."""

    # copy given data to avoid changing the original one
    df = df.copy()
    # iterate over the columns and fill missing values with the mean
    for column in df.select_dtypes(include=[np.number]).columns:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

    return df

def impute_knn(df: pd.DataFrame, n_neighbors: int ):
    """Impute missing values using K-Nearest Neighbors algorithm."""

    df = df.copy()
    imputer = KNNImputer(n_neighbors = n_neighbors)
    df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(
        df.select_dtypes(include=[np.number])
    )

    return df

