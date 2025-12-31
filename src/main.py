import os
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection") # Silences warnings about classes with few samples
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute") to stop the conversion warnings from MICE if we want 
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model") # Silences future warnings from sklearn in this case of n_jobs = -1

# local imports 
from imputation_techniques import impute_mean, impute_knn, impute_mice, impute_missForest, pool_mice_results
from evaluation import create_missing_mask, evaluate_imputation, evaluate_downstream_task
from visualization import ImputationVisualizer
from preprocessing import load_data, clean_data, encode_categorical_variables, explore_data, identify_missing_patterns, analyze_missing_data_type

# directories 
RAW_DIR = '../data/raw'
PROCESSED_DIR = '../data/processed'
RESULTS_DIR = '../results'

# ddefine iterations and missing rate
MC_TEST_ITER = 2   # do 20 test simulations
MC_REAL_ITER = 2   # do 20 real simulations
MISSING_RATE = 0.2

# methods
METHODS = {
    'Mean': lambda df: impute_mean(df),
    'KNN': lambda df: impute_knn(df, n_neighbors=5),
    'MICE': lambda df: pool_mice_results(impute_mice(df, n_imputations=5)),
    'MissForest': lambda df: impute_missForest(df)
}

STOCHASTIC = ['MICE', 'MissForest']

# PROGRESS BAR FUNCTION 

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    
    if iteration == total: 
        print()

def get_best_scatter_pair(df):
    """Finds the pair of most correlated numerical columns."""
    try:
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        if corr_matrix.empty: return None, None
        
        max_val = corr_matrix.max().max()
        row, col = np.where(corr_matrix == max_val)
        if len(row) > 0:
            return corr_matrix.columns[row[0]], corr_matrix.columns[col[0]]
    except:
        pass
    return None, None

def run_pipeline(file_path):
    filename = os.path.basename(file_path)
    dataset_name = os.path.splitext(filename)[0]

    print(f"\n{'-'*20}Switched to new Dataset{'-'*20}")
    print(f"\nPROCESSING: {filename}")

    df_loaded = load_data(file_path)

    if df_loaded is None:
        print(f"Skipping {filename} due to error.")
        return
    
    dataset_processed_dir = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(dataset_processed_dir, exist_ok=True)

    print(f"\n\t\t INITIAL DATA ANALYSIS")
    explore_data(df_loaded)
    identify_missing_patterns(df_loaded)
    analyze_missing_data_type(df_loaded)
    

    df_clean = clean_data(df_loaded)
    df_raw, _ = encode_categorical_variables(df_clean, encoding_type='label')
    prep_path = os.path.join(dataset_processed_dir, f"{dataset_name}_preprocessed.csv")
    df_raw.to_csv(prep_path, index=False)
    print(f"Preprocessed data saved to: {prep_path}")

    # Detect the Target
    target_col = None
    possible_targets = ['target', 'class', 'diagnosis', 'output', 'num']
    for c in df_raw.columns:
        if c.lower() in possible_targets:
            target_col = c; 
            break
    
    if target_col:
        print(f"Target detected: '{target_col}'")
    else:
        print("No target detected. Skipping accuracy analysis.")

    # PHASE 1: MONTE CARLO - TEST

    print(f"\n Entering Phase 1: Test Monte Carlo ({MC_TEST_ITER} runs)")
    test_results = []
    df_complete = df_raw.dropna()
    
    if len(df_complete) > 50:
        # initialize empty progress bar (0 of the total)
        print_progress_bar(0, MC_TEST_ITER, prefix='Progress:', suffix='Complete', length=40)

        for i in range(MC_TEST_ITER):
            df_art, _, true_vals = create_missing_mask(df_complete, MISSING_RATE, random_state=i)
            
            for m_name, m_func in METHODS.items():
                try:
                    t0 = time.time()
                    df_fill = m_func(df_art)
                    dt = time.time() - t0
                    
                    rmse = evaluate_imputation(true_vals, df_fill, 'rmse')
                    mae = evaluate_imputation(true_vals, df_fill, 'mae')
                    
                    acc = {}
                    if target_col:
                        acc = evaluate_downstream_task(df_fill, target_col)

                    register = {
                        'method': m_name,
                        'iteration': i,
                        'rmse': rmse,
                        'mae': mae,
                        'time': dt,
                        'accuracy': acc
                    }

                    for model_name, score in acc.items():
                        register[f'accuracy_{model_name}'] = score
                        
                    test_results.append(register    )
                except Exception as e:
                    pass
            
            # looks more or less like  Progress: |████------| 40.0% Iteration: X out of Y
            print_progress_bar(i + 1, MC_TEST_ITER, prefix='Testing:', suffix=f'Iteration: {i+1} out of {MC_TEST_ITER}', length=40)
            
    else:
        print("Not enough complete data for Phase 1. Skipping.")
    
    df_test = pd.DataFrame(test_results)
    
    metrics_summary = None
    if not df_test.empty:
        metrics_summary = df_test.groupby('method').mean(numeric_only=True).reset_index()
        
        metrics_summary = metrics_summary.rename(columns={
            'method': 'Method', 
            'rmse': 'RMSE_mean',
            'mae': 'MAE_mean',
            'time': 'Time_mean'
        })

    # PHASE 2: MONTE CARLO - REAL

    print(f"\n Entering Phase 2: Real Imputation ({MC_REAL_ITER} runs)")
    real_results = []
    final_dfs = {} 
    
    # initialize progress bar for phase 2
    print_progress_bar(0, MC_REAL_ITER, prefix='Real Sim:', suffix='Complete', length=40)

    for i in range(MC_REAL_ITER):
        for m_name, m_func in METHODS.items():
            if m_name not in STOCHASTIC and i > 0:
                prev = [r for r in real_results if r['method'] == m_name]
                if prev:
                    
                    prev_record = prev[0]
                    new_record = {'method': m_name}

                    for key, value in prev_record.items():
                        if key.startswith('accuracy_'):
                            new_record[key] = value

                    real_results.append(new_record)
                continue

            try:
                df_fill = m_func(df_raw)
                
                final_dfs[m_name] = df_fill
                
                acc = {}
                if target_col: 
                    acc = evaluate_downstream_task(df_fill, target_col)
                
                register = {'method': m_name}
                for model_name, score in acc.items():
                    register[f'accuracy_{model_name}'] = score
                real_results.append(register)

            except Exception as e:
                pass
        
        # update the progress bar
        print_progress_bar(i + 1, MC_REAL_ITER, prefix='Real Sim:', suffix=f'Iter {i+1}/{MC_REAL_ITER}', length=40)

    df_real = pd.DataFrame(real_results)
    
    # save the final CSVs
    for m, df in final_dfs.items():
        out_name = f"{dataset_name}_{m}.csv"
        df.to_csv(os.path.join(dataset_processed_dir, out_name), index=False)

    # PHASE 3: VISUALIZATIONS

    print("\n Entering Phase 3: Generating Visualizations")
    
    viz = ImputationVisualizer(filename, metrics_summary, df_test)
    
    if metrics_summary is not None:
        viz.plot_bar_comparison('rmse')
        viz.plot_bar_comparison('mae')
        viz.plot_computation_time()
        viz.plot_monte_carlo_stability('rmse')
        viz.plot_error_density('rmse')
        viz.plot_convergence('rmse')
        viz.plot_imputation_vs_downstream()
        
    if not df_real.empty and target_col:
        viz.mc_df = df_real 
        viz.plot_downstream_uncertainty()
        
    num_cols = df_raw.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0 and final_dfs:
        col_x, col_y = get_best_scatter_pair(df_raw)
        if col_x and col_y:
            viz.plot_scatter_comparison(df_raw, final_dfs, col_x, col_y)
            
        # Find the numerical column with the most missing values for the distribution plot
        cols_with_missing = df_loaded.select_dtypes(include=[np.number]).isna().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0].sort_values(ascending=False)
        
        if not cols_with_missing.empty:
            target_dist_col = cols_with_missing.index[0]
            viz.plot_distribution_comparison(df_raw, final_dfs, target_dist_col)

    print(f"\nSUCCESS: all graphs were created and inserted in {RESULTS_DIR}/{dataset_name}/.")
    print(f"Pipeline Finished for {dataset_name}.")

if __name__ == "__main__":
    for d in [PROCESSED_DIR, RESULTS_DIR]:
        if not os.path.exists(d): os.makedirs(d)
        
    try:
        all_files = os.listdir(RAW_DIR)
        files = [os.path.join(RAW_DIR, f) for f in all_files if f.endswith('.csv')]
    except FileNotFoundError:
        files = []
    
    if not files: 
        print(f"No CSV files found in {RAW_DIR}")
    else:
        print(f"Found {len(files)} datasets: {[os.path.basename(f) for f in files]}")
        for f in files: run_pipeline(f)