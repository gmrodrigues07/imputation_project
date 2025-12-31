import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# NOTE: not using detect_outliers, split_train_test nor save_processed_data 

def load_data(filepath):
    """
    Load the  dataset from CSV file.
    
    Parameters:
    filepath : str
        Path to the CSV file
        
    Returns:
    pd.DataFrame
        Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}")
        print(f"Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def explore_data(data):
    """
    Perform initial data exploration to understand the dataset structure,
    missing values, and data types.
    
    Parameters:
    data : pd.DataFrame
        Input dataset
        
    Returns:
    dict
        Dictionary containing exploration results
    """
    if data is None:
        print("Error: No data provided")
        return None
    
    exploration_results = {}
    
    # Basic information
    exploration_results['dtypes'] = data.dtypes
    
    # Missing values analysis
    print("\n" + "="*100)
    print("\t\t MISSING VALUES ANALYSIS")
    print("="*100)
    missing_counts = data.isnull().sum()
    missing_percentages = (data.isnull().sum() / len(data) * 100).round(2)
    
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    if len(missing_summary) > 0:
        print(f"\nColumns with missing values:")
        print(missing_summary)
        exploration_results['missing_values'] = missing_summary
    else:
        print("\nNo missing values found in the dataset.")
        exploration_results['missing_values'] = None
    
    # Basic statistics for numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        #print(data[numerical_cols].describe())
        exploration_results['numerical_stats'] = data[numerical_cols].describe()
    else:
        #print("No numerical columns found.")
        exploration_results['numerical_stats'] = None
    
    # Categorical columns
    print("\n" + "="*100)
    print("\t\t CATEGORICAL COLUMNS")
    print("="*100)
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Categorical columns: {categorical_cols}")
        print("\nUnique values per categorical column:")
        for col in categorical_cols:
            print(f"{col}: {data[col].nunique()} unique values")
        exploration_results['categorical_cols'] = categorical_cols
    else:
        print("No categorical columns found.")
        exploration_results['categorical_cols'] = []
    
    # Sample of the data
    exploration_results['shape'] = data.shape
    exploration_results['columns'] = list(data.columns)
    
    return exploration_results


def identify_missing_patterns(data):
    """
    Identify patterns in missing data (which columns/rows have missing values).
    
    Parameters:
    data : pd.DataFrame
        Input dataset
        
    Returns:
    dict
        Dictionary containing missing data patterns
    """

    if data is None:
        print("Error: No data provided")
        return None
    
    patterns = {}
    
    # Rows with missing values
    rows_with_missing = data.isnull().any(axis=1).sum()
    patterns['rows_with_missing'] = rows_with_missing
    patterns['rows_with_missing_pct'] = (rows_with_missing / len(data) * 100).round(2)
    
    print("\n" + "="*100)
    print("\t\t MISSING DATA PATTERNS")
    print("="*100)
    print(f"Rows with at least one missing value: {rows_with_missing} ({patterns['rows_with_missing_pct']}%)")
    print(f"Rows with complete data: {len(data) - rows_with_missing} ({(100 - patterns['rows_with_missing_pct']):.2f}%)")
    
    # Columns with missing values
    cols_with_missing = data.isnull().any(axis=0).sum()
    patterns['cols_with_missing'] = cols_with_missing
    
    # Total missing values
    total_missing = data.isnull().sum().sum()
    total_cells = data.shape[0] * data.shape[1]
    patterns['total_missing'] = total_missing
    patterns['total_missing_pct'] = (total_missing / total_cells * 100).round(2)
    
    print(f"Total missing values: {total_missing} out of {total_cells} ({patterns['total_missing_pct']}%)")
    
    return patterns


def analyze_missing_data_type(data, alpha=0.05):
    """
    Analyze the type of missing data mechanism (MCAR, MAR, MNAR).
    
    This function performs statistical tests to help identify the missingness mechanism:
    - Little's MCAR test (approximation using chi-square)
    - Correlation analysis between missingness indicators and observed values
    - Tests comparing observed values between complete and incomplete cases
    
    Parameters:
    data : pd.DataFrame
        Input dataset
    alpha : float, default=0.05
        Significance level for statistical tests
        
    Returns:
    dict
        Dictionary containing analysis results and interpretations
    """
    if data is None:
        print("Error: No data provided")
        return None
    
    print("\n" + "="*100)
    print("\t\t MISSING DATA MECHANISM ANALYSIS")
    print("="*100)
    
    analysis_results = {}
    
    # Get columns with missing values
    cols_with_missing = data.columns[data.isnull().any()].tolist()
    
    if not cols_with_missing:
        print("\nNo missing values found. Analysis not needed.")
        return None
        
    # Create missingness indicator variables
    
    missing_indicators = pd.DataFrame()
    for col in cols_with_missing:
        missing_indicators[f'{col}_missing'] = data[col].isnull().astype(int)
        
    # Correlation between missingness and observed values
    print("\n\t\t CORRELATION ANALYSIS")
    print("\nTesting if missingness in one variable is related to observed values in other variables")
    
    correlation_results = []
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for missing_col in cols_with_missing:
        if missing_col not in numerical_cols:
            continue
            
        for obs_col in numerical_cols:
            if obs_col == missing_col or obs_col not in data.columns:
                continue
            
            # Get complete cases for the observed column
            valid_data = data[obs_col].notna()
            
            if valid_data.sum() < 2:
                continue
            
            # Calculate correlation between missingness indicator and observed values
            missing_indicator = missing_indicators[f'{missing_col}_missing']
            
            # Only consider rows where obs_col has values
            mask = valid_data
            if mask.sum() < 2:
                continue
                
            # to avoid constant array error
            v1 = missing_indicator[mask]
            v2 = data.loc[mask, obs_col]
            
            if v1.std() == 0 or v2.std() == 0:
                continue

            try:
                corr, p_value = stats.pointbiserialr(v1, v2)
                
                if abs(corr) > 0.1 and p_value < alpha:  # Significant correlation
                    correlation_results.append({
                        'Missing_Variable': missing_col,
                        'Observed_Variable': obs_col,
                        'Correlation': round(corr, 4),
                        'P_Value': round(p_value, 4),
                        'Significant': 'Yes' if p_value < alpha else 'No'
                    })
            except:
                continue
    
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results).sort_values('P_Value')
        print("\nSignificant correlations found (suggests MAR or MNAR):")
        print(corr_df.to_string(index=False))
        analysis_results['correlations'] = corr_df
    else:
        print("\nNo significant correlations found between missingness and observed values")
        analysis_results['correlations'] = None
    
    # Compare distributions: complete vs incomplete cases
    print("\n\t\t DISTRIBUTION COMPARISON ")
    print("\nTesting if observed values differ between complete and incomplete cases")
    
    ttest_results = []
    
    for missing_col in cols_with_missing:
        if missing_col not in numerical_cols:
            continue
            
        for obs_col in numerical_cols:
            if obs_col == missing_col:
                continue
            
            # Split data into complete and incomplete cases
            complete_cases = data[data[missing_col].notna()][obs_col].dropna()
            incomplete_cases = data[data[missing_col].isna()][obs_col].dropna()
            
            if len(complete_cases) < 2 or len(incomplete_cases) < 2:
                continue
            
            try:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(complete_cases, incomplete_cases)
                
                if p_value < alpha:
                    ttest_results.append({
                        'Missing_In': missing_col,
                        'Observed_Variable': obs_col,
                        'Complete_Mean': round(complete_cases.mean(), 2),
                        'Incomplete_Mean': round(incomplete_cases.mean(), 2),
                        'T_Statistic': round(t_stat, 4),
                        'P_Value': round(p_value, 4),
                        'Significant': 'Yes' if p_value < alpha else 'No'
                    })
            except:
                continue
    
    if ttest_results:
        ttest_df = pd.DataFrame(ttest_results).sort_values('P_Value')
        print("\nSignificant differences found (suggests MAR or MNAR):")
        print(ttest_df.to_string(index=False))
        analysis_results['ttests'] = ttest_df
    else:
        print("\nNo significant differences in distributions (suggests MCAR)")
        analysis_results['ttests'] = None
    
    # Provide interpretation
    print("\n\t\t INTERPRETATION\n")
    
    interpretation = []
    
    if not correlation_results and not ttest_results:
        interpretation.append("• No significant relationships detected between missingness and observed values")
        interpretation.append("• This suggests MCAR (Missing Completely At Random)")
        interpretation.append("• Simple imputation methods like mean/median may be appropriate")
        analysis_results['likely_mechanism'] = 'MCAR'
    else:
        if correlation_results:
            interpretation.append(f"• Found {len(correlation_results)} significant correlations between missingness and other variables")
        if ttest_results:
            interpretation.append(f"• Found {len(ttest_results)} significant differences in distributions")
        
        interpretation.append("• This suggests MAR (Missing At Random) or MNAR (Missing Not At Random)")
        interpretation.append("• MAR: Missingness depends on observed data (can be modeled)")
        interpretation.append("• MNAR: Missingness depends on unobserved data (more complex)")
        interpretation.append("• Recommended: Use advanced imputation (MICE, KNN) or consider model-based approaches")
        
        # Try to distinguish MAR from MNAR
        if correlation_results and len(correlation_results) > len(cols_with_missing):
            interpretation.append("• Strong evidence of MAR (multiple observed predictors of missingness)")
            analysis_results['likely_mechanism'] = 'MAR'
        else:
            interpretation.append("• Could be MAR or MNAR - domain knowledge required for confirmation")
            analysis_results['likely_mechanism'] = 'MAR/MNAR'
    
    print("\n".join(interpretation))
    analysis_results['interpretation'] = interpretation
    
    return analysis_results


def clean_data(data):
    """
    Clean the dataset by handling duplicates, standardizing missing values,and basic data quality checks.
    
    Parameters:
    data : pd.DataFrame
        Input dataset
        
    Returns:
    pd.DataFrame
        Cleaned dataset
    """
    if data is None:
        print("Error: No data provided")
        return None
    
    
    data_clean = data.copy()
    
    # Check for duplicate rows
    n_duplicates = data_clean.duplicated().sum()
    if n_duplicates > 0:
        data_clean = data_clean.drop_duplicates()
    
    # Standardize missing value representations
    
    # Replace common missing value representations with NaN
    missing_values = ['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'None', 'none', 'NULL', 'null', '?', '-', '--', 'missing', 'MISSING']
    
    for col in data_clean.columns:
        if data_clean[col].dtype == 'object':
            # Strip whitespace from string values only
            try:
                data_clean[col] = data_clean[col].astype(str).str.strip()
            except:
                pass
            # Replace empty strings and common missing value representations
            data_clean[col] = data_clean[col].replace('', np.nan)
            data_clean[col] = data_clean[col].replace(missing_values, np.nan)
            # Replace 'nan' string back to NaN
            data_clean[col] = data_clean[col].replace('nan', np.nan)
        
    # Remove ID and dataset columns (not useful for modeling)
    cols_to_remove = []
    if ('id' or 'Id') in data_clean.columns.str.lower():
        id_cols = [col for col in data_clean.columns if col.lower() == 'id']
        cols_to_remove.extend(id_cols)
    if 'dataset' in data_clean.columns:
        cols_to_remove.append('dataset')
    
    if cols_to_remove:
        data_clean = data_clean.drop(columns=cols_to_remove)
    
    # Check for constant columns 
    constant_cols = []
    for col in data_clean.columns:
        if data_clean[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        data_clean = data_clean.drop(columns=constant_cols)
    
    # Check data types and inconsistencies
    for col in data_clean.columns:
        if data_clean[col].dtype == 'object':
            # Try to convert to numeric if possible
            try:
                numeric_col = pd.to_numeric(data_clean[col], errors='coerce')
                # If more than 50% can be converted, it's likely a numeric column
                if numeric_col.notna().sum() / len(data_clean) > 0.5:
                    data_clean[col] = numeric_col
            except:
                pass
    
    
    return data_clean


def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers in numerical columns using IQR or Z-score method.
    
    Parameters:
    data : pd.DataFrame
        Input dataset
    method : str
        Method to use: 'iqr' (Interquartile Range) or 'zscore'
    threshold : float
        Threshold multiplier (1.5 for IQR, 3.0 for Z-score typically)
        
    Returns:
    dict
        Dictionary containing outlier information per column
    """

    if data is None:
        print("Error: No data provided")
        return None
    
    print("\n" + "="*100)
    print("OUTLIER DETECTION")
    print("="*100)
    print(f"Method: {method.upper()}, Threshold: {threshold}")
    
    outlier_info = {}
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        col_data = data[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            outliers_mask = z_scores > threshold
            outliers = col_data[outliers_mask]
        else:
            print(f"Unknown method: {method}")
            continue
        
        if len(outliers) > 0:
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(data) * 100, 2),
                'indices': outliers.index.tolist()
            }
    
    if outlier_info:
        print("\nOutliers detected:")
        for col, info in outlier_info.items():
            print(f"  {col}: {info['count']} outliers ({info['percentage']}%)")
    else:
        print("\nNo outliers detected")
    
    return outlier_info


def encode_categorical_variables(data, encoding_type='label', drop_first=True):
    """
    Encode categorical variables using Label Encoding or One-Hot Encoding.
    
    Parameters:
    data : pd.DataFrame
        Input dataset
    encoding_type : str
        'label' for Label Encoding (default), 'onehot' for One-Hot Encoding, 
        'auto' to automatically choose based on cardinality
    drop_first : bool
        For one-hot encoding, whether to drop first category to avoid multicollinearity
        
    Returns:
    tuple
        (encoded_data, encoding_info)
    """
    if data is None:
        print("Error: No data provided")
        return None, None
    
    print("\n" + "="*100)
    print("\t\t CATEGORICAL VARIABLE ENCODING")
    print("="*100)
    
    data_encoded = data.copy()
    encoding_info = {}
    
    categorical_cols = data_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found")
        return data_encoded, encoding_info
    
    print(f"\nFound {len(categorical_cols)} categorical columns!")
    
    for col in categorical_cols:
        n_unique = data_encoded[col].nunique()
        
        # Decide encoding strategy
        if encoding_type == 'auto':
            # Use label encoding for high cardinality 
            if n_unique > 10:
                strategy = 'label'
            else:
                strategy = 'onehot'
        else:
            strategy = encoding_type
        
        print(f"\n  {col}: {n_unique} unique values -> {strategy} encoding")
        
        if strategy == 'label':
            # Label Encoding
            le = LabelEncoder()
            # Handle NaN values
            mask = data_encoded[col].notna()
            data_encoded.loc[mask, col] = le.fit_transform(data_encoded.loc[mask, col])
            
            encoding_info[col] = {
                'type': 'label',
                'classes': le.classes_.tolist()
            }
            
        elif strategy == 'onehot':
            dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=drop_first, dtype=int)
            
            # Drop original column and add dummy columns
            data_encoded = data_encoded.drop(columns=[col])
            data_encoded = pd.concat([data_encoded, dummies], axis=1)
            
            encoding_info[col] = {
                'type': 'onehot',
                'columns': dummies.columns.tolist()
            }
    
    print(f"\nEncoding complete. New shape: {data_encoded.shape}")
    
    return data_encoded, encoding_info


def split_train_test(data, target_column=None, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and testing sets. This is done before imputation to avoid data leakage.
    
    Parameters:
    data : pd.DataFrame
        Input dataset
    target_column : str, optional
        Name of target column for stratified split
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to use stratified split (only if target_column provided)
        
    Returns:
    tuple
        (train_data, test_data)
    """

    if data is None:
        print("Error: No data provided")
        return None, None
    
    print("\n" + "="*100)
    print("TRAIN-TEST SPLIT")
    print("="*100)
    print(f"Test size: {test_size*100}%")
    print(f"Random state: {random_state}")
    
    stratify_col = None
    if target_column and stratify and target_column in data.columns:
        stratify_col = data[target_column]
        print(f"Stratified split on: {target_column}")
    
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_col
    )
    
    print(f"\nTrain set size: {train_data.shape}")
    print(f"Test set size: {test_data.shape}")
    
    if target_column and target_column in data.columns:
        print(f"\nTarget distribution:")
        print("Train:")
        print(train_data[target_column].value_counts(normalize=True).round(3))
        print("\nTest:")
        print(test_data[target_column].value_counts(normalize=True).round(3))
    
    return train_data, test_data


def save_processed_data(train_data, test_data, output_dir='../data/processed', prefix=''):
    """
    Save processed training and testing data to CSV files.
    
    Parameters:
    train_data : pd.DataFrame
        Training dataset
    test_data : pd.DataFrame
        Testing dataset
    output_dir : str
        Directory to save the files
    prefix : str
        Prefix for the output filenames
        
    Returns:
    tuple
        (train_filepath, test_filepath)
    """
    
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Generate filenames
    train_filename = f"{prefix}train_data.csv" if prefix else "train_data.csv"
    test_filename = f"{prefix}test_data.csv" if prefix else "test_data.csv"
    
    train_filepath = os.path.join(full_output_dir, train_filename)
    test_filepath = os.path.join(full_output_dir, test_filename)
    
    # Save files
    train_data.to_csv(train_filepath, index=False)
    test_data.to_csv(test_filepath, index=False)
    
    print("\n" + "="*100)
    print("DATA SAVED")
    print("="*100)
    print(f"Train data saved to: {train_filepath}")
    print(f"Test data saved to: {test_filepath}")
    
    return train_filepath, test_filepath


def add_random_missing_values(data, missing_percentage, random_state=42, exclude_columns=None):
    """
    Add random missing values to a dataset. this function introduces NaNs randomly to simulate missing data.
    
    Parameters:
    data : pd.DataFrame
        Input dataset to add missing values to
    missing_percentage : float
        Percentage of values to make missing (0-100)
        Example: 10.0 means 10% of values will be set to NaN
    random_state : int, default=42
        Random seed for reproducible results
    exclude_columns : list, optional
        List of column names to exclude from having missing values added
        
    Returns:
    pd.DataFrame
        Dataset with randomly added missing values
    """

    if data is None:
        print("Error: No data provided")
        return None
    
    if not 0 <= missing_percentage <= 100:
        print("Error: missing_percentage must be between 0 and 100")
        return None
    
    # Create a copy to avoid modifying the original data
    data_with_missing = data.copy()
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get columns to modify 
    columns_to_modify = data.columns.tolist()
    if exclude_columns:
        columns_to_modify = [col for col in columns_to_modify if col not in exclude_columns]
    
    if not columns_to_modify:
        print("Warning: No columns available to add missing values (all excluded)")
        return data_with_missing
    
    print(f"\nAdding {missing_percentage}% random missing values...")
    print(f"Columns to modify: {columns_to_modify}")
    if exclude_columns:
        print(f"Excluded columns: {exclude_columns}")
    
    # Calculate number of values to make missing
    total_cells_to_modify = len(data_with_missing) * len(columns_to_modify)
    n_missing_to_add = int(total_cells_to_modify * (missing_percentage / 100))
    
    print(f"Total cells in target columns: {total_cells_to_modify}")
    print(f"Number of values to make missing: {n_missing_to_add}")
    
    # Generate random positions for missing values
    missing_positions = []
    
    for _ in range(n_missing_to_add):
        # Randomly select a row
        row_idx = np.random.randint(0, len(data_with_missing))
        # Randomly select a column from the modifiable columns
        col_idx = np.random.randint(0, len(columns_to_modify))
        col_name = columns_to_modify[col_idx]
        
        missing_positions.append((row_idx, col_name))
    
    # Add missing values at the selected positions
    missing_count_by_column = {}
    for row_idx, col_name in missing_positions:
        data_with_missing.iloc[row_idx, data_with_missing.columns.get_loc(col_name)] = np.nan
        missing_count_by_column[col_name] = missing_count_by_column.get(col_name, 0) + 1
    
    # Report results
    print("\nMissing values added successfully!")
    print("Missing values per column:")
    for col in columns_to_modify:
        count = missing_count_by_column.get(col, 0)
        percentage = (count / len(data_with_missing) * 100) if len(data_with_missing) > 0 else 0
        print(f"  {col}: {count} values ({percentage:.2f}%)")
    
    # Overall missing values summary
    total_missing_before = data.isnull().sum().sum()
    total_missing_after = data_with_missing.isnull().sum().sum()
    added_missing = total_missing_after - total_missing_before
    
    print(f"\nSummary:")
    print(f"  Missing values before: {total_missing_before}")
    print(f"  Missing values after: {total_missing_after}")
    print(f"  Missing values added: {added_missing}")
    
    total_cells = data_with_missing.shape[0] * data_with_missing.shape[1]
    overall_missing_pct = (total_missing_after / total_cells * 100) if total_cells > 0 else 0
    print(f"  Overall missing percentage: {overall_missing_pct:.2f}%")
    
    return data_with_missing


if __name__ == "__main__":
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    heart_disease_path = os.path.join(script_dir, "..", "data", "raw", "heart_disease_uci.csv")
    wineqt_path = os.path.join(script_dir, "..", "data", "raw", "WineQT.csv")
    
    # Choose which dataset to process
    print("="*100)
    print("DATASET SELECTION")
    print("="*100)
    print("1. Heart Disease UCI")
    print("2. Wine Quality")
    print("3. Both datasets")
    
    dataset_choice = input("\nSelect dataset (1, 2, or 3): ").strip()
    
    datasets_to_process = []
    if dataset_choice == "1":
        datasets_to_process = [("Heart Disease", heart_disease_path, "num")]
    elif dataset_choice == "2":
        datasets_to_process = [("Wine Quality", wineqt_path, "quality")]
    elif dataset_choice == "3":
        datasets_to_process = [
            ("Heart Disease", heart_disease_path, "num"),
            ("Wine Quality", wineqt_path, "quality")
        ]
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    # Process each dataset
    for dataset_name, data_path, target_col in datasets_to_process:
        print("\n\n" + "="*100)
        print(f"PROCESSING: {dataset_name.upper()}")
        print("="*100)
        
        # Load data
        print(f"\nLoading {dataset_name} dataset...")
        df = load_data(data_path)
        
        if df is not None:
            # Explore data
            exploration_results = explore_data(df)
            
            # Identify missing patterns
            missing_patterns = identify_missing_patterns(df)
            
            # Analyze missing data type (MCAR, MAR, MNAR)
            missing_type_analysis = analyze_missing_data_type(df)
            
            # Clean data
            df_clean = clean_data(df)
            
            # Detect outliers
            outlier_info = detect_outliers(df_clean, method='iqr', threshold=1.5)
            
            # Encode categorical variables
            df_encoded, encoding_info = encode_categorical_variables(df_clean, encoding_type='label')
            
            # Split into train/test before imputation
            train_data, test_data = split_train_test(
                df_encoded, 
                target_column=target_col,
                test_size=0.2,
                random_state=42,
                stratify=True
            )
            
            # Save processed data
            if train_data is not None and test_data is not None:
                save_processed_data(
                    train_data, 
                    test_data, 
                    output_dir='../data/processed',
                    prefix=f'preprocessed_{dataset_name.lower().replace(" ", "_")}_'
                )
            
            # Test add_random_missing_values function (only for Wine Quality dataset)
            if dataset_name == "Wine Quality":
                print("\n" + "="*100)
                print(f"TESTING ADD_RANDOM_MISSING_VALUES FUNCTION ON {dataset_name.upper()}")
                print("="*100)
                
                # Get complete data
                df_complete = df_clean.dropna()
                print(f"\nComplete data shape (no missing values): {df_complete.shape}")
                
                # Test with 10% missing values
                print("\n" + "-"*100)
                print("Test 1: Adding 10% random missing values (excluding target column)")
                print("-"*100)
                df_10pct_missing = add_random_missing_values(
                    data=df_complete,
                    missing_percentage=10.0,
                    random_state=42,
                    exclude_columns=[target_col]
                )
                
                # Test with 20% missing values
                print("\n" + "-"*100)
                print("Test 2: Adding 20% random missing values (excluding target column)")
                print("-"*100)
                df_20pct_missing = add_random_missing_values(
                    data=df_complete,
                    missing_percentage=20.0,
                    random_state=123,
                    exclude_columns=[target_col]
                )
                
                # Analyze the artificial missing data
                print("\n" + "-"*100)
                print("Analyzing 20% artificial missing data:")
                print("-"*100)
                identify_missing_patterns(df_20pct_missing)
            
            print("\n" + "="*100)
            print(f"PREPROCESSING COMPLETE FOR {dataset_name.upper()}")
            print("="*100)
            print("\nNext steps:")
            print("1. Apply imputation techniques to train and test sets separately")
            print("2. Fit imputer on train_data only")
            print("3. Transform both train_data and test_data")
            print("4. Proceed with model training")
            print("5. Test imputation quality using artificially created missing data")
        else:
            print(f"Failed to load {dataset_name} dataset. Skipping...")
