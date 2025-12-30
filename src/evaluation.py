import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # to use for the logistic regression 
from sklearn.pipeline import make_pipeline # to use for the logistic regression as well
import time


def create_missing_mask(data, missing_percentage=0.2, random_state=53):
    """
    Create a mask of artificial missing values for evaluation purposes.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Complete dataset
    missing_percentage : float
        Percentage of values to mark as missing (0-1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (data_with_missing, missing_mask, original_values)
    """
    np.random.seed(random_state)
    
    data_copy = data.copy()
    missing_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    original_values = {}
    
    # Only create missing values in numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        # Skip columns that already have too many missing values
        if data[col].isnull().sum() / len(data) > 0.5:
            continue
            
        # Get indices of non-missing values
        valid_indices = data[col].notna()
        n_valid = valid_indices.sum()
        
        if n_valid == 0:
            continue
        
        # Determine number of values to remove
        n_to_remove = int(n_valid * missing_percentage)
        
        if n_to_remove == 0:
            continue
        
        # Randomly select indices to remove
        valid_positions = data[valid_indices].index.tolist()
        indices_to_remove = np.random.choice(valid_positions, size=n_to_remove, replace=False)
        
        # Store original values
        original_values[col] = data.loc[indices_to_remove, col].copy()
        
        # Create missing mask
        missing_mask.loc[indices_to_remove, col] = True
        
        # Set values to NaN
        data_copy.loc[indices_to_remove, col] = np.nan
    
    return data_copy, missing_mask, original_values


def evaluate_imputation(original_values, imputed_values, metric='rmse'):
    """
    Evaluate imputation quality by comparing imputed values with original values.
    
    Parameters:
    -----------
    original_values : dict or pd.Series
        Original values before artificial missingness
    imputed_values : dict or pd.Series
        Imputed values
    metric : str
        Metric to use: 'rmse', 'mae', 'r2', 'mape'
        
    Returns:
    --------
    float
        Evaluation score
    """
    if isinstance(original_values, dict):
        # Flatten dictionary of series to single arrays
        orig_vals = np.concatenate([v.values for v in original_values.values()])
        imp_vals = np.concatenate([imputed_values[k].loc[v.index].values 
                                   for k, v in original_values.items()])
    else:
        orig_vals = original_values.values
        imp_vals = imputed_values.values
    
    # Remove any remaining NaN values
    mask = ~(np.isnan(orig_vals) | np.isnan(imp_vals))
    orig_vals = orig_vals[mask]
    imp_vals = imp_vals[mask]
    
    if len(orig_vals) == 0:
        return np.nan
    
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(orig_vals, imp_vals))
    elif metric == 'mae':
        return mean_absolute_error(orig_vals, imp_vals)
    elif metric == 'r2':
        return r2_score(orig_vals, imp_vals)
    elif metric == 'mape':
        # Mean Absolute Percentage Error
        return np.mean(np.abs((orig_vals - imp_vals) / orig_vals)) * 100
    else:
        raise ValueError(f"Unknown metric: {metric}")

# might need to eliminate this function later because we will implement it on the main.py file due to the monte carlo simulation
def cross_validate_imputation(data, imputation_method, n_splits=5, missing_percentage=0.2, 
                               stratify_column=None, metrics=['rmse', 'mae', 'r2'], 
                               random_state=53, verbose=True):
    """
    Perform k-fold cross-validation for imputation methods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Complete dataset (or with existing missing values)
    imputation_method : object
        Imputation method with fit() and transform() or fit_transform() methods
        Should be compatible with sklearn API
    n_splits : int
        Number of folds for cross-validation
    missing_percentage : float
        Percentage of values to artificially remove for testing (0-1)
    stratify_column : str, optional
        Column name to use for stratified k-fold
    metrics : list
        List of metrics to compute: 'rmse', 'mae', 'r2', 'mape'
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    dict
        Dictionary containing results for each fold and aggregate statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("K-FOLD CROSS-VALIDATION FOR IMPUTATION")
        print("="*60)
        print(f"Method: {imputation_method.__class__.__name__}")
        print(f"Number of folds: {n_splits}")
        print(f"Artificial missing percentage: {missing_percentage*100}%")
        print(f"Metrics: {metrics}")
    
    # Initialize results storage
    results = {
        'fold_results': [],
        'fold_times': [],
        'aggregate_metrics': {}
    }
    
    # Set up cross-validation splitter
    if stratify_column is not None and stratify_column in data.columns:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kfold.split(data, data[stratify_column])
        if verbose:
            print(f"Using Stratified K-Fold on column: {stratify_column}")
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kfold.split(data)
        if verbose:
            print("Using regular K-Fold")
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        if verbose:
            print(f"\n{'-'*60}")
            print(f"Fold {fold_idx}/{n_splits}")
            print(f"{'-'*60}")
        
        fold_start_time = time.time()
        
        # Split data
        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()
        
        if verbose:
            print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        # Create artificial missing values in test set
        test_with_missing, missing_mask, original_values = create_missing_mask(
            test_data, 
            missing_percentage=missing_percentage,
            random_state=random_state + fold_idx
        )
        
        if verbose:
            total_missing = sum(len(v) for v in original_values.values())
            print(f"Created {total_missing} artificial missing values")
        
        # Fit imputation method on training data
        try:
            if hasattr(imputation_method, 'fit'):
                imputation_method.fit(train_data)
                imputed_test = imputation_method.transform(test_with_missing)
            elif hasattr(imputation_method, 'fit_transform'):
                # Fit on train, transform test
                imputation_method.fit_transform(train_data)
                imputed_test = imputation_method.transform(test_with_missing)
            else:
                raise AttributeError("Imputation method must have fit() and transform() or fit_transform() methods")
        except Exception as e:
            if verbose:
                print(f"Error in fold {fold_idx}: {str(e)}")
            continue
        
        # Convert to DataFrame if necessary
        if not isinstance(imputed_test, pd.DataFrame):
            imputed_test = pd.DataFrame(imputed_test, columns=test_data.columns, index=test_data.index)
        
        # Evaluate imputation quality
        fold_metrics = {}
        for metric in metrics:
            try:
                score = evaluate_imputation(original_values, imputed_test, metric=metric)
                fold_metrics[metric] = score
                if verbose:
                    print(f"{metric.upper()}: {score:.4f}")
            except Exception as e:
                if verbose:
                    print(f"Error computing {metric}: {str(e)}")
                fold_metrics[metric] = np.nan
        
        fold_time = time.time() - fold_start_time
        
        # Store results
        results['fold_results'].append(fold_metrics)
        results['fold_times'].append(fold_time)
        
        if verbose:
            print(f"Fold time: {fold_time:.2f}s")
    
    # Compute aggregate statistics
    if verbose:
        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
    
    for metric in metrics:
        scores = [fold[metric] for fold in results['fold_results'] if not np.isnan(fold[metric])]
        if scores:
            results['aggregate_metrics'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'scores': scores
            }
            if verbose:
                print(f"\n{metric.upper()}:")
                print(f"  Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
                print(f"  Min:  {np.min(scores):.4f}")
                print(f"  Max:  {np.max(scores):.4f}")
    
    total_time = sum(results['fold_times'])
    results['total_time'] = total_time
    results['mean_fold_time'] = np.mean(results['fold_times'])
    
    if verbose:
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Average time per fold: {np.mean(results['fold_times']):.2f}s")
    
    return results

# might need to eliminate this function later because we will implement it on the main.py file due to the monte carlo simulation
def compare_imputation_methods(data, methods_dict, n_splits=5, missing_percentage=0.2,
                               stratify_column=None, metrics=['rmse', 'mae'], 
                               random_state=53):
    """
    Compare multiple imputation methods using cross-validation.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Complete dataset
    methods_dict : dict
        Dictionary of {method_name: imputation_method} to compare
    n_splits : int
        Number of folds for cross-validation
    missing_percentage : float
        Percentage of values to artificially remove
    stratify_column : str, optional
        Column name for stratified k-fold
    metrics : list
        List of metrics to compute
    random_state : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Comparison table of methods and their performance
    """
    print("\n" + "="*60)
    print("COMPARING IMPUTATION METHODS")
    print("="*60)
    
    comparison_results = []
    
    for method_name, method in methods_dict.items():
        print(f"\n{'*'*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'*'*60}")
        
        results = cross_validate_imputation(
            data=data,
            imputation_method=method,
            n_splits=n_splits,
            missing_percentage=missing_percentage,
            stratify_column=stratify_column,
            metrics=metrics,
            random_state=random_state,
            verbose=True
        )
        
        # Extract aggregate metrics
        method_results = {'Method': method_name}
        for metric in metrics:
            if metric in results['aggregate_metrics']:
                method_results[f'{metric.upper()}_mean'] = results['aggregate_metrics'][metric]['mean']
                method_results[f'{metric.upper()}_std'] = results['aggregate_metrics'][metric]['std']
        
        method_results['Time_mean'] = results['mean_fold_time']
        method_results['Time_total'] = results['total_time']
        
        comparison_results.append(method_results)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def evaluate_downstream_task(df_imputed, target_col, cv=5):
    """
    Treina um Random Forest para classificar a coluna alvo, e depois mede a 'Certeza da Decisão' em cenários reais.
    """
    if target_col not in df_imputed.columns:
        return np.nan
        
    # remove lines where the target is still NaN
    df = df_imputed.dropna(subset=[target_col]).copy()
    
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # ver se é classificação
    if pd.api.types.is_numeric_dtype(y):
        y = y.round().astype(int)
    
    if y.nunique() < 2:
        return  
    
    classifiers = {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1, random_state=53)),
        'RandomForest': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=53),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=53, max_depth=3)
    }

    results = {}

    for classifier_name, classifier in classifiers.items():
        try:
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
            results[classifier_name] = scores.mean()
        except Exception as e:
            results[classifier_name] = np.nan

    return results

if __name__ == "__main__":
    print("Evaluation module for imputation methods")
    print("This module provides cross-validation functionality.")
    print("\nExample usage:")
    print("""
    from evaluation import cross_validate_imputation
    from imputation_techniques import SimpleImputer
    from sklearn.impute import SimpleImputer
    
    # Load your data
    data = pd.read_csv('data.csv')
    
    # Create imputation method
    imputer = SimpleImputer(strategy='mean')
    
    # Run cross-validation
    results = cross_validate_imputation(
        data=data,
        imputation_method=imputer,
        n_splits=5,
        missing_percentage=0.2,
        metrics=['rmse', 'mae', 'r2']
    )
    """)
