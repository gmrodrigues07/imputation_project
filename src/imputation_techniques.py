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

def impute_missForest(df: pd.DataFrame):
    """Impute missing values using MissForest algorithm."""
    return df

def impute_mice(df: pd.DataFrame, n_imputations: int = 5):
    """Impute missing values using MICE (Multiple Imputation by Chained Equations) algorithm ."""
    return df

'''def impute_knn_manually(df: pd.DataFrame, n_neighbors: int):
    """Impute missing values using K-Nearest Neighbors algorithm"""

    df = df.copy()
    collumns = df.select_dtypes(include=[np.number]).columns # only collumns with numeric values
    df_numeric = df[collumns].to_numpy() # convert to array
    
    for i in range(df_numeric.shape[0]): # iterate over rows

        if np.any(np.isnan(df_numeric[i])): # if has a missing value

            # checks the distance to each other point and calculates the norm to which points are closest
            distances = np.linalg.norm(df_numeric - df_numeric[i], axis=1)
            distances[i] = np.inf  # exclude itself by setting its distance to infinity
            neighbor_indices = np.argsort(distances)[:n_neighbors] # sort from smallest to largest and keeps the n_neighbors closest

            for j, value in enumerate(df_numeric[i]): # iterate over each column

                if np.isnan(value): # if there is a value missing
                    neighbor_values = df_numeric[neighbor_indices, j] # get the values of the neighbors for that column
                    df_numeric[i, j] = np.nanmean(neighbor_values) # does the mean of the neighbors and assigns it to the missing value
    
    df[collumns] = df_numeric # replaces the numeric collumns with the new ones with the imputed values
    return df'''