import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # to use for the logistic regression 
from sklearn.pipeline import make_pipeline # to use for the logistic regression as well
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer


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
    
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)  # ✅ PROTECT TARGET!

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
        kfold = StratifiedKFold(n_splits=min(5, len(np.unique(y)) * 2), shuffle=True, random_state=random_state)
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


def simple_imputation_comparison(data, imputation_functions, missing_percentage=0.2, random_state=42):
    """
    Simple comparison of imputation methods without cross-validation.
    
    Creates artificial missing values, imputes them with different methods,
    and calculates MSE/RMSE compared to the real values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Complete dataset (without missing values, or will use complete cases)
    imputation_functions : dict
        Dictionary of {method_name: imputation_function}
        Each function should take a DataFrame and return an imputed DataFrame
    missing_percentage : float
        Percentage of values to artificially remove (0-1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with MSE and RMSE for each method
        
    Example:
    --------
    >>> from imputation_techniques import impute_mean, impute_knn, impute_mice, pool_mice_results
    >>> 
    >>> methods = {
    >>>     'Mean': impute_mean,
    >>>     'KNN_5': lambda df: impute_knn(df, n_neighbors=5),
    >>>     'MICE': lambda df: pool_mice_results(impute_mice(df, n_imputations=3))
    >>> }
    >>> results = simple_imputation_comparison(data, methods, missing_percentage=0.2)
    """
    print("\n" + "="*100)
    print("SIMPLE IMPUTATION COMPARISON")
    print("="*100)
    
    # Use only complete cases
    data_complete = data.dropna()
    print(f"Using {len(data_complete)} complete rows from dataset")
    
    # Create artificial missing values
    print(f"\nCreating {missing_percentage*100}% artificial missing values...")
    data_with_missing, missing_mask, original_values = create_missing_mask(
        data_complete, 
        missing_percentage=missing_percentage,
        random_state=random_state
    )
    
    total_missing = sum(len(v) for v in original_values.values())
    print(f"Created {total_missing} artificial missing values across {len(original_values)} columns")
    
    # Test each imputation method
    comparison_results = []
    
    for method_name, imputation_func in imputation_functions.items():
        print(f"\n{'-'*100}")
        print(f"Testing: {method_name}")
        print(f"{'-'*100}")
        
        try:
            import time
            start_time = time.time()
            
            # Apply imputation
            data_imputed = imputation_func(data_with_missing.copy())
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            mse = evaluate_imputation(original_values, data_imputed, metric='rmse') ** 2  # Square to get MSE
            rmse = np.sqrt(mse)
            mae = evaluate_imputation(original_values, data_imputed, metric='mae')
            
            # Check if all missing values were imputed
            remaining_missing = data_imputed.isnull().sum().sum()
            
            print(f"MSE:  {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Remaining missing values: {remaining_missing}")
            
            comparison_results.append({
                'Method': method_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Time (s)': elapsed_time,
                'Missing_After': remaining_missing
            })
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            comparison_results.append({
                'Method': method_name,
                'MSE': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'Time (s)': np.nan,
                'Missing_After': np.nan
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    print("\n" + "="*100)
    print("FINAL COMPARISON")
    print("="*100)
    print(comparison_df.to_string(index=False))
    
    # Rank methods by RMSE
    comparison_df_sorted = comparison_df.sort_values('RMSE')
    print("\n" + "="*100)
    print("RANKING (Best to Worst by RMSE)")
    print("="*100)
    print(comparison_df_sorted.to_string(index=False))
    
    return comparison_df


def evaluate_downstream_task(df_imputed, target_col, cv=5, metrics=['accuracy', 'precision', 'recall', 'f1']):
    """
    Evaluate imputation quality by training classifiers on imputed data.
    
    Trains multiple classifiers and evaluates them using various metrics
    to assess the practical utility of the imputed data.
    
    Parameters:
    -----------
    df_imputed : pd.DataFrame
        Imputed dataset
    target_col : str
        Name of the target column for classification
    cv : int
        Number of cross-validation folds
    metrics : list
        List of metrics to compute: 'accuracy', 'precision', 'recall', 'f1'
        
    Returns:
    --------
    dict
        Nested dictionary: {classifier_name: {metric: score}}
    """
    if target_col not in df_imputed.columns:
        return {}

    # Remove rows where the target is missing
    df = df_imputed.dropna(subset=[target_col]).copy()
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    if pd.api.types.is_numeric_dtype(y):
        y = y.round().astype(int)
    if y.nunique() < 2:
        return {}

    # Impute the entire datase=t first
    X_imputed = imputation_func(X)
    from xgboost import XGBClassifier
    classifiers = {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1, random_state=53)),
        'RandomForest': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=53),
        'XGBoost': XGBClassifier(n_estimators=50, random_state=53, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
    }
    results = {}
    for classifier_name, classifier in classifiers.items():
        try:
            y_encoded = pd.Categorical(y, categories=sorted(y.unique())).codes
            scores = cross_val_score(classifier, X_imputed, y_encoded, cv=cv, scoring='accuracy')
            results[classifier_name] = {'accuracy': scores.mean()}
        except Exception as e:
            results[classifier_name] = {'accuracy': np.nan}
    return results

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from imputation_techniques import impute_mean, impute_median, impute_knn, impute_missForest, impute_mice, pool_mice_results
    
    # ✅ ALL IMPORTS
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    print("="*100)
    print("EVALUATION MODULE - ACCURACY + F1 ONLY (10 ITERATIONS)")
    print("="*100)
    
    # Dataset selection
    print("\n1. Heart Disease UCI")
    print("2. Wine Quality")
    print("3. Both datasets")
    
    dataset_choice = input("\nSelect dataset (1, 2, or 3): ").strip()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_to_process = []
    
    if dataset_choice == "1":
        datasets_to_process = [(
            "Heart Disease",
            os.path.join(script_dir, "..", "data", "processed", "preprocessed_heart_disease_train_data.csv"),
            "num"
        )]
    elif dataset_choice == "2":
        datasets_to_process = [(
            "Wine Quality",
            os.path.join(script_dir, "..", "data", "processed", "preprocessed_wine_quality_train_data.csv"),
            "quality"
        )]
    elif dataset_choice == "3":
        datasets_to_process = [
            ("Heart Disease", os.path.join(script_dir, "..", "data", "processed", "preprocessed_heart_disease_train_data.csv"), "num"),
            ("Wine Quality", os.path.join(script_dir, "..", "data", "processed", "preprocessed_wine_quality_train_data.csv"), "quality")
        ]
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    # Define imputation methods
    methods = {
        'Mean': impute_mean,
        'Median': impute_median,
        'KNN (k=5)': lambda df: impute_knn(df, n_neighbors=5),
        'KNN (k=10)': lambda df: impute_knn(df, n_neighbors=10),
        'MissForest': lambda df: impute_missForest(df, max_iter=10, n_estimators=50),
        'MICE': lambda df: pool_mice_results(impute_mice(df, n_imputations=3, max_iter=10))
    }
    
    # F1 SCORER
    f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)
    
    # Process each dataset
    for dataset_name, data_path, target_col in datasets_to_process:
        print("\n\n" + "="*100)
        print(f"PROCESSING: {dataset_name.upper()}")
        print("="*100)
        
        # Load data
        try:
            data = pd.read_csv(data_path)
            print(f"\nLoaded data from: {data_path}")
            print(f"Shape: {data.shape}")
            
            total_missing = data.isnull().sum().sum()
            print(f"Missing values: {total_missing}")
            
            if total_missing == 0:
                print(f"\n{'!'*100}")
                print(f"NOTE: {dataset_name} has NO natural missing values!")
                print(f"Creating artificial missing values for evaluation...")
                print(f"{'!'*100}")
                
                data, _, _ = create_missing_mask(
                    data=data,
                    missing_percentage=0.15,
                    random_state=42
                )
                print(f"Created artificial missing values: {data.isnull().sum().sum()} missing cells")
                
        except FileNotFoundError:
            print(f"\nERROR: File not found at {data_path}")
            continue

        # Part 1: Simple imputation comparison
        print("\n" + "#"*100)
        print("PART 1: IMPUTATION QUALITY EVALUATION (MSE/RMSE/MAE)")
        print("#"*100)
        
        imputation_results_1 = simple_imputation_comparison(
            data=data,
            imputation_functions=methods,
            missing_percentage=0.1,
            random_state=42
        )

        classifiers = {
            'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1, random_state=53)),
            'RandomForest': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=53),
            'XGBoost': XGBClassifier(n_estimators=50, random_state=53, n_jobs=-1, eval_metric='logloss')
        }

        # ✅ TRUE CV PIPELINE (replace entire Part 2)
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        print("\n\n" + "#"*100)
        print("PART 2: TRUE CROSS-VALIDATED PIPELINES")
        print("#"*100)

        # Create pipelines for EACH method
        pipelines = {}

        # 1. No Handling → just dropna in preprocessing
        pipelines['No Handling'] = Pipeline([
            ('dropna', FunctionTransformer(lambda X: X.dropna())),
            ('clf', classifiers['XGBoost'])
        ])

        # 2. XGBoost Native → raw data
        pipelines['XGBoost Native'] = Pipeline([
            ('clf', classifiers['XGBoost'])
        ])

        # 3. Imputation pipelines
        for method_name, imputation_func in methods.items():
            pipelines[method_name] = Pipeline([
                ('impute', FunctionTransformer(imputation_func)),
                ('clf', classifiers['XGBoost'])
            ])

        # TRUE 10-FOLD CV (no leakage!)
        results = {}
        for name, pipe in pipelines.items():
            print(f"CV {name}...")
            
            # Encode y ONCE (outside CV)
            y_encoded = data[target_col]
            if len(np.unique(y_encoded)) > 2:
                unique_classes = sorted(np.unique(y_encoded))
                y_encoded = pd.Categorical(y_encoded, categories=unique_classes).codes
            
            X = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
            
            # 10-fold stratified CV
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = cross_validate(pipe, X, y_encoded, cv=skf,
                                    scoring={'accuracy': 'accuracy', 'f1': f1_scorer},
                                    return_train_score=False)
            
            results[name] = {
                'accuracy_mean': np.mean(cv_scores['test_accuracy']),
                'accuracy_std': np.nanstd(cv_scores['test_accuracy']),
                'f1_mean': np.mean(cv_scores['test_f1']),
                'f1_std': np.nanstd(cv_scores['test_f1'])
            }

        # PERFECT TABLE
        results_df = pd.DataFrame(results).T.round(4)
        print("\n🏆 TRUE CROSS-VALIDATED RESULTS:")
        print(results_df[['accuracy_mean', 'accuracy_std', 'f1_mean', 'f1_std']])

        # Save
        results_df.to_csv(os.path.join(script_dir, '..', 'results', f"{dataset_name}_true_cv_results.csv"))