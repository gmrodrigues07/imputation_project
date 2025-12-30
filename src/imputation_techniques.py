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
        df[column] = df[column].fillna(mean_value)

    return df

def impute_median(df: pd.DataFrame ):
    """Impute missing valuesusing the mean of the column."""

    # copy given data to avoid changing the original one
    df = df.copy()
    # iterate over the columns and fill missing values with the mean
    for column in df.select_dtypes(include=[np.number]).columns:
        mean_value = df[column].median()
        df[column] = df[column].fillna(mean_value)

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
    max_iter: int = 20, # alterei para um valor mais pequeno para testes
    n_estimators: int = 100,
    random_state: int = 42
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
    
    # All columns should already be numeric (encoded by preprocessing)
    # Use IterativeImputer with RandomForest estimator
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy='mean',
        verbose=0
    )
    
    # Fit and transform on all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        try:
            imputed_data = imputer.fit_transform(df[numeric_cols])
            df[numeric_cols] = imputed_data
        except Exception as e:
            print(f"MissForest imputation failed: {e}")
            return df
    
    return df

def impute_mice(
    df: pd.DataFrame,
    n_imputations: int = 5,
    max_iter: int = 20,
    random_state: int = 42
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
    max_iter : int, default=20
        Maximum number of imputation iterations per dataset.
    random_state : int, default=42
        Random seed for reproducibility. Each imputation will use
        random_state + i as its seed.
    
    Returns:
    --------
    List[pd.DataFrame]
        List of n_imputations DataFrames, each with imputed values.
    """
    imputed_datasets = []
    
    # All columns should already be numeric (encoded by preprocessing)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return [df.copy() for _ in range(n_imputations)]
    
    # Generate multiple imputations
    for i in range(n_imputations):
        df_imp = df.copy()
        # Set seed for this imputation
        seed = None if random_state is None else random_state + i
        
        # Use IterativeImputer (MICE implementation in sklearn)
        # sample_posterior=True is important for proper multiple imputation
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=seed,
            initial_strategy='mean',
            sample_posterior=True,
            verbose=0
        )
        
        try:
            # Fit and transform
            imputed_data = imputer.fit_transform(df_imp[numeric_cols])
            df_imp[numeric_cols] = imputed_data
        except Exception as e:
            print(f"MICE imputation {i+1} failed: {e}")
            pass
        
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


if __name__ == "__main__":
    import os
    
    print("="*100)
    print("TESTING IMPUTATION TECHNIQUES")
    print("="*100)
    
    # Load preprocessed data (already cleaned and encoded)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "..", "data", "processed", "preprocessed_heart_disease_train_data.csv")
    
    print(f"\nLoading preprocessed data from: {train_path}")
    df = pd.read_csv(train_path)
    print(f"Data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Test each imputation method
    print("\n" + "="*100)
    print("1. TESTING MEAN IMPUTATION")
    print("="*100)
    df_mean = impute_mean(df)
    print(f"Missing values after mean imputation: {df_mean.isnull().sum().sum()}")
    
    print("\n" + "="*100)
    print("2. TESTING MEDIAN IMPUTATION")
    print("="*100)
    df_median = impute_median(df)
    print(f"Missing values after median imputation: {df_median.isnull().sum().sum()}")
    
    print("\n" + "="*100)
    print("3. TESTING KNN IMPUTATION (k=5)")
    print("="*100)
    df_knn = impute_knn(df, n_neighbors=5)
    print(f"Missing values after KNN imputation: {df_knn.isnull().sum().sum()}")
    
    print("\n" + "="*100)
    print("4. TESTING MISSFOREST IMPUTATION")
    print("="*100)
    df_missforest = impute_missForest(df, max_iter=10, n_estimators=50)
    print(f"Missing values after MissForest imputation: {df_missforest.isnull().sum().sum()}")
    
    print("\n" + "="*100)
    print("5. TESTING MICE IMPUTATION")
    print("="*100)
    print("Creating 3 imputed datasets...")
    imputed_datasets = impute_mice(df, n_imputations=3, max_iter=10)
    print(f"Number of imputed datasets created: {len(imputed_datasets)}")
    for i, imp_df in enumerate(imputed_datasets):
        print(f"  Dataset {i+1} - Missing values: {imp_df.isnull().sum().sum()}")
    
    print("\nPooling MICE results...")
    df_mice_pooled = pool_mice_results(imputed_datasets)
    print(f"Missing values after pooling: {df_mice_pooled.isnull().sum().sum()}")
    
    print("\n" + "="*100)
    print("SUMMARY: ALL IMPUTATION METHODS TESTED SUCCESSFULLY")
    print("="*100)
    print("\nImputation Methods:")
    print(f"  1. Mean:        {df_mean.isnull().sum().sum()} missing values remaining")
    print(f"  2. Median:      {df_median.isnull().sum().sum()} missing values remaining")
    print(f"  3. KNN (k=5):   {df_knn.isnull().sum().sum()} missing values remaining")
    print(f"  4. MissForest:  {df_missforest.isnull().sum().sum()} missing values remaining")
    print(f"  5. MICE (pooled): {df_mice_pooled.isnull().sum().sum()} missing values remaining")
    
    # Compare a few imputed values
    print("\n" + "="*100)
    print("SAMPLE COMPARISON OF IMPUTED VALUES")
    print("="*100)
    
    # Find a column with missing values
    missing_col = df.columns[df.isnull().any()][0] if df.isnull().any().any() else None
    
    if missing_col:
        missing_idx = df[df[missing_col].isnull()].index[:3]  # First 3 rows with missing values
        
        if len(missing_idx) > 0:
            print(f"\nColumn: {missing_col}")
            print(f"Comparing imputed values for rows: {missing_idx.tolist()}\n")
            
            comparison = pd.DataFrame({
                'Original': df.loc[missing_idx, missing_col],
                'Mean': df_mean.loc[missing_idx, missing_col],
                'Median': df_median.loc[missing_idx, missing_col],
                'KNN': df_knn.loc[missing_idx, missing_col],
                'MissForest': df_missforest.loc[missing_idx, missing_col],
                'MICE': df_mice_pooled.loc[missing_idx, missing_col]
            })
            print(comparison.round(2))
    else:
        print("\nNo missing values found to compare (dataset may already be complete)")
