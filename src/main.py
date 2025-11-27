import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from imputation_techniques import impute_mean, impute_knn

# Load Raw and Processed datasets for training and end result 

# Dataset 1   NOTE: INSERT REAL NAMES OF DATASETS ON THE PATHS

# Load datasets from the other directory (from ../data/raw)
original_path_1 = '../data/raw/heart_disease_uci.csv' # Path to the original dataset create one for each dataset  
print("dataset path:", original_path_1)  
df_original_1 = pd.read_csv(original_path_1) 
# Load dataset without missing values (from ../data/processed)
processed_path_1 = '../data/processed/heart_disease_uci.csv'
df_processed_1 = pd.read_csv(processed_path_1)

df_with_missing_1 = df_processed_1.copy()
print("Dataset 1 loaded!")

# Dataset 2

original_path_2 = '../data/raw/2.csv'     
df_original_2 = pd.read_csv(original_path_2) 
processed_path_2 = '../data/processed/2.csv'
df_processed_2 = pd.read_csv(processed_path_2)

df_with_missing_2 = df_processed_2.copy()
print("Dataset 2 loaded!")

# Dataset 3 

original_path_3 = '../data/raw/3.csv'   
df_original_3 = pd.read_csv(original_path_3) 
processed_path_3 = '../data/processed/3.csv'
df_processed_3 = pd.read_csv(processed_path_3)

df_with_missing_3 = df_processed_3.copy()
print("Dataset 3 loaded!")

#  Introduce Missing Values Artificially 

# Introduce missing values artificially pseudorandomlly for training purpose and all techniques have the same cells missing in each dataset because of the seed 
# Dataset 1

np.random.seed(42)
missing_frac_1 = 0.2  # % of the values to be set as NaN
for col in df_with_missing_1.select_dtypes(include=[np.number]).columns:
    missing_indices = df_with_missing_1.sample(frac=missing_frac_1).index
    df_with_missing_1.loc[missing_indices, col] = np.nan

# Dataset 2

np.random.seed(235)
missing_frac_2 = 0.15  # % of the values to be set as NaN
for col in df_with_missing_2.select_dtypes(include=[np.number]).columns:
    missing_indices = df_with_missing_2.sample(frac=missing_frac_2).index
    df_with_missing_2.loc[missing_indices, col] = np.nan    

# Dataset 3

np.random.seed(123)
missing_frac_3 = 0.1  # % of the values to be set as NaN
for col in df_with_missing_3.select_dtypes(include=[np.number]).columns:
    missing_indices = df_with_missing_3.sample(frac=missing_frac_3).index
    df_with_missing_3.loc[missing_indices, col] = np.nan

print("Missing values introduced!!")

#  Impute missing values using the techniques from imputation_techniques.py wanted 

# Dataset 1

# Mean imputation
df_mean_imputed_1 = impute_mean(df_with_missing_1)
# KNN imputation (select n_neighbors as wanted maybe make one big and one small and one medium?)
df_knn_imputed_1 = impute_knn(df_with_missing_1, n_neighbors=5) 
print("Dataset 1 imputation completed!!!!!!!!!!!!!!!")
# index=False to avoid writing the numeration of the rows | used to save new datasets 
df_with_missing_1.to_csv('../data/processed/with_missing/dataset1_with_missing.csv', index = False)
df_mean_imputed_1.to_csv('../data/processed/results/dataset1_mean_imputed.csv', index = False)
df_knn_imputed_1.to_csv('../data/processed/results/dataset1_knn_imputed.csv', index = False)  

# Dataset 2     

# Mean imputation
df_mean_imputed_2 = impute_mean(df_with_missing_2)
# KNN imputation
df_knn_imputed_2 = impute_knn(df_with_missing_2, n_neighbors=3) 
print("Dataset 2 imputation completed!!!!!!!!!!!!!!!")
df_with_missing_2.to_csv('../data/processed/with_missing/dataset2_with_missing.csv', index = False)
df_mean_imputed_2.to_csv('../data/processed/results/dataset2_mean_imputed.csv', index = False)
df_knn_imputed_2.to_csv('../data/processed/results/dataset2_knn_imputed.csv', index = False)

# Dataset 3

# Mean imputation
df_mean_imputed_3 = impute_mean(df_with_missing_3)
# KNN imputation
df_knn_imputed_3 = impute_knn(df_with_missing_3, n_neighbors=7) 
print("Dataset 3 imputation completed!!!!!!!!!!!!!!!")
df_with_missing_3.to_csv('../data/processed/with_missing/dataset3_with_missing.csv', index = False)
df_mean_imputed_3.to_csv('../data/processed/results/dataset3_mean_imputed.csv', index = False)
df_knn_imputed_3.to_csv('../data/processed/results/dataset3_knn_imputed.csv', index = False)

print("All datasets created were saved!")

# plot the comparison between the methods using the metrics (call the metrics (other file) and then plot the results here)

