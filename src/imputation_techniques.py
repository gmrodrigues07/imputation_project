import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Union

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

def impute_missForest(
    df: pd.DataFrame,
    max_iter: int = 10,
    n_estimators: int = 100,
    random_state: int = None
) -> pd.DataFrame:
    """
    Impute missing values using MissForest algorithm.
    
    MissForest uses Random Forest models to impute missing values iteratively.
    It handles both numerical and categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with missing values to impute.
    max_iter : int, default=10
        Maximum number of imputation iterations.
    n_estimators : int, default=100
        Number of trees in the Random Forest.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with imputed values.
    """
    df = df.copy()
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle NaN by treating it as a special category during fit
        mask = df[col].notna()
        if mask.any():
            le.fit(df.loc[mask, col])
            label_encoders[col] = le
            # Transform non-null values, keep NaN as NaN
            df.loc[mask, col] = le.transform(df.loc[mask, col])
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Prepare data for imputation
    all_cols = numeric_cols + categorical_cols
    if not all_cols:
        return df
    
    # Use IterativeImputer with RandomForest estimator
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        ),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy='mean'
    )
    
    # Fit and transform
    imputed_data = imputer.fit_transform(df[all_cols])
    df[all_cols] = imputed_data
    
    # Decode categorical variables back to original categories
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Round to nearest integer and clip to valid range
            df[col] = df[col].round().astype(int)
            df[col] = df[col].clip(0, len(le.classes_) - 1)
            df[col] = le.inverse_transform(df[col])
    
    return df

def impute_mice(
    df: pd.DataFrame,
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: int = None
) -> List[pd.DataFrame]:
    """
    Impute missing values using MICE (Multiple Imputation by Chained Equations) algorithm.
    
    MICE performs multiple imputations, creating several complete datasets.
    Each imputation uses an iterative approach where each feature is imputed
    using the others as predictors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with missing values to impute.
    n_imputations : int, default=5
        Number of imputed datasets to generate.
    max_iter : int, default=10
        Maximum number of imputation iterations per dataset.
    random_state : int, default=None
        Random seed for reproducibility. Each imputation will use
        random_state + i as its seed.
    
    Returns:
    --------
    List[pd.DataFrame]
        List of n_imputations DataFrames, each with imputed values.
    """
    imputed_datasets = []
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    if not all_cols:
        return [df.copy() for _ in range(n_imputations)]
    
    # Encode categorical variables once
    df_encoded = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        mask = df_encoded[col].notna()
        if mask.any():
            le.fit(df_encoded.loc[mask, col])
            label_encoders[col] = le
            df_encoded.loc[mask, col] = le.transform(df_encoded.loc[mask, col])
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    # Generate multiple imputations
    for i in range(n_imputations):
        df_imp = df_encoded.copy()
        
        # Set seed for this imputation
        seed = None if random_state is None else random_state + i
        
        # Use IterativeImputer (MICE implementation in sklearn)
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=seed,
            initial_strategy='mean',
            sample_posterior=True  # Important for proper multiple imputation
        )
        
        # Fit and transform
        imputed_data = imputer.fit_transform(df_imp[all_cols])
        df_imp[all_cols] = imputed_data
        
        # Decode categorical variables back to original categories
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Round to nearest integer and clip to valid range
                df_imp[col] = df_imp[col].round().astype(int)
                df_imp[col] = df_imp[col].clip(0, len(le.classes_) - 1)
                df_imp[col] = le.inverse_transform(df_imp[col])
        
        imputed_datasets.append(df_imp)
    
    return imputed_datasets


def pool_mice_results(imputed_datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Pool results from multiple MICE imputations into a single dataset.
    
    For numeric columns, takes the mean across imputations.
    For categorical columns, takes the mode (most frequent value).
    
    Parameters:
    -----------
    imputed_datasets : List[pd.DataFrame]
        List of imputed DataFrames from impute_mice().
    
    Returns:
    --------
    pd.DataFrame
        Single DataFrame with pooled imputed values.
    """
    if not imputed_datasets:
        raise ValueError("No imputed datasets provided")
    
    if len(imputed_datasets) == 1:
        return imputed_datasets[0].copy()
    
    pooled = imputed_datasets[0].copy()
    numeric_cols = pooled.select_dtypes(include=[np.number]).columns
    categorical_cols = pooled.select_dtypes(include=['object', 'category']).columns
    
    # Pool numeric columns by averaging
    for col in numeric_cols:
        values = np.array([df[col].values for df in imputed_datasets])
        pooled[col] = np.mean(values, axis=0)
    
    # Pool categorical columns by mode
    for col in categorical_cols:
        values = pd.DataFrame({i: df[col] for i, df in enumerate(imputed_datasets)})
        pooled[col] = values.mode(axis=1)[0]
    
    return pooled

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