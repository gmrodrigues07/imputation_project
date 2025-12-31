# imputation_project
**APC Project:**
This project conducts a comprehensive comparative analysis of imputation strategies, contrasting traditional methods (Mean, Median, KNN) against advanced algorithms (MICE, MissForest).
It is Possible to run main.py and evaluation.py. The main has a pipeline including the monte carlo simulations and generates all final datasets, as for the evaluation, it iterates 10 times using CV and creates artificial missing data and calculates performance metrics (RMSE, MAE, R², Accuracy) used by the main analysis, also generates one graph.

**Authors:**

Lucas Aparicio (up202206594)

Guilherme Rodrigues (up202208878)

Rúben Oliveira (up202205106)

**Installation:**

To run this code, you need Python installed along with the following libraries that you can install them using pip.

pip install pandas numpy matplotlib scikit-learn xgboost scipy

