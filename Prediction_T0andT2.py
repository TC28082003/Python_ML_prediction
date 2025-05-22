import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge # Add Ridge for regularization
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Read and prepare data (Same as previous steps) ---
file_path = 'Pain_python.csv'
df = pd.read_csv(file_path)
df.rename(columns={'Q28. JointP': 'Joint_Pain', 'Pain rating': 'Pain_Rating',
                   'CD4%': 'CD4_Percent', '%Trans sat': 'Trans_Sat_Percent',
                   'Transf': 'Transferrin_Actual'},
          inplace=True)
cols_to_numeric = ['Lymphocytes', 'Age', 'Pain_Rating', 'Neutrophils', 'WCC', 'CRP', 'HgB', 'Platelets',
                   'CD3Total', 'CD8-Suppr', 'CD4_Percent', 'CD4-Helper', 'CD19 Bcell', 'Ferritin', 'CK', 'Iron',
                   'Trans_Sat_Percent', 'Transferrin_Actual']
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Pair data and calculate _Change (Same as before) ---
vars_for_delta = ['Pain_Rating', 'Transferrin_Actual', 'CD4_Percent',
                  'Platelets', 'Neutrophils', 'Trans_Sat_Percent']
df_pivoted_list = []
for var in vars_for_delta:
    if var in df.columns:
        df_pivot_var = df.pivot_table(index='Number', columns='Timepoint', values=var)
        df_pivot_var.columns = [f'{var}_T0', f'{var}_T2']
        df_pivoted_list.append(df_pivot_var)
df_paired = df_pivoted_list[0]
for i in range(1, len(df_pivoted_list)):
    df_paired = pd.merge(df_paired, df_pivoted_list[i], on='Number', how='inner')
df_paired.reset_index(inplace=True)
for var in vars_for_delta:
    col_t0 = f'{var}_T0'; col_t2 = f'{var}_T2'
    if col_t0 in df_paired.columns and col_t2 in df_paired.columns:
        df_paired[f'{var}_Change'] = df_paired[col_t2] - df_paired[col_t0]

# --- PART A: PREDICT PAIN_RATING_CHANGE ---
print("--- PART A: PREDICTING THE CHANGE IN PAIN LEVEL (Pain_Rating_Change) ---")
# Select variables and remove NaN
cols_for_prediction_A = ['Pain_Rating_Change', 'Transferrin_Actual_Change', 'CD4_Percent_Change',
                         'Platelets_Change', 'Neutrophils_Change', 'Trans_Sat_Percent_Change']
data_A = df_paired.dropna(subset=cols_for_prediction_A).copy()

if not data_A.empty and len(data_A) > 10: # Need enough data to split
    y_A = data_A['Pain_Rating_Change']
    X_A = data_A[['Transferrin_Actual_Change', 'CD4_Percent_Change',
                  'Platelets_Change', 'Neutrophils_Change', 'Trans_Sat_Percent_Change']]

    # Split data into training and test sets
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

    # 1. Use the trained RLM model (version with constant as it's more general)
    # Retrain on X_train_A to get coefficients for prediction
    X_train_A_const = sm.add_constant(X_train_A)
    rlm_model_final_A = sm.RLM(y_train_A, X_train_A_const, M=sm.robust.norms.HuberT()).fit()
    print("\nRLM Model (with constant) Summary (trained on training set):")
    print(rlm_model_final_A.summary())

    X_test_A_const = sm.add_constant(X_test_A)
    y_pred_rlm_A = rlm_model_final_A.predict(X_test_A_const)
    mse_rlm_A = mean_squared_error(y_test_A, y_pred_rlm_A)
    r2_rlm_A = r2_score(y_test_A, y_pred_rlm_A)
    print(f"\nRLM - Evaluation on test set:")
    print(f"  Mean Squared Error (MSE): {mse_rlm_A:.4f}")
    print(f"  R-squared (R2): {r2_rlm_A:.4f}") # R2 can be negative if the model is worse than predicting the mean

    # 2. Use Ordinary Linear Regression (LinearRegression from scikit-learn)
    lr_model_A = LinearRegression()
    lr_model_A.fit(X_train_A, y_train_A)
    y_pred_lr_A = lr_model_A.predict(X_test_A)
    mse_lr_A = mean_squared_error(y_test_A, y_pred_lr_A)
    r2_lr_A = r2_score(y_test_A, y_pred_lr_A)
    print(f"\nLinear Regression (scikit-learn) - Evaluation on test set:")
    print(f"  Mean Squared Error (MSE): {mse_lr_A:.4f}")
    print(f"  R-squared (R2): {r2_lr_A:.4f}")
    print(f"  Intercept: {lr_model_A.intercept_:.4f}")
    print(f"  Coefficients: {lr_model_A.coef_}")

    # Visualize predictions vs. actual values (for the better model if available)
    plt.figure(figsize=(8,6))
    plt.scatter(y_test_A, y_pred_lr_A, alpha=0.7, label='Linear Regression') # Or y_pred_rlm_A
    plt.plot([y_test_A.min(), y_test_A.max()], [y_test_A.min(), y_test_A.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual value of Pain_Rating_Change")
    plt.ylabel("Predicted value of Pain_Rating_Change")
    plt.title("Prediction vs. Actual for Pain_Rating_Change")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Not enough data for Part A after processing.")

# --- PART B: PREDICT PAIN_RATING AT T2 ---
print("\n\n--- PART B: PREDICTING PAIN LEVEL AT T2 (Pain_Rating_T2) ---")
# Use biomarkers at T0 and T2, and Pain_Rating at T0 to predict Pain_Rating at T2
cols_for_prediction_B = ['Pain_Rating_T2', 'Pain_Rating_T0', # Pain_Rating_T0 is an important feature
                         'Transferrin_Actual_T0', 'CD4_Percent_T0', 'Platelets_T0', 'Neutrophils_T0', 'Trans_Sat_Percent_T0',
                         'Transferrin_Actual_T2', 'CD4_Percent_T2', 'Platelets_T2', 'Neutrophils_T2', 'Trans_Sat_Percent_T2']
# Or just use _Change:
# cols_for_prediction_B = ['Pain_Rating_T2', 'Pain_Rating_T0',
# 'Transferrin_Actual_Change', 'CD4_Percent_Change', 'Platelets_Change',
# 'Neutrophils_Change', 'Trans_Sat_Percent_Change']

data_B = df_paired.dropna(subset=cols_for_prediction_B).copy()

if not data_B.empty and len(data_B) > 10:
    y_B = data_B['Pain_Rating_T2']
    X_B_feature_cols = [col for col in cols_for_prediction_B if col != 'Pain_Rating_T2']
    X_B = data_B[X_B_feature_cols]

    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

    # Use Ridge Regression (Linear regression with L2 regularization to prevent overfitting)
    ridge_model_B = Ridge(alpha=1.0) # alpha is the regularization parameter
    ridge_model_B.fit(X_train_B, y_train_B)
    y_pred_ridge_B = ridge_model_B.predict(X_test_B)
    mse_ridge_B = mean_squared_error(y_test_B, y_pred_ridge_B)
    r2_ridge_B = r2_score(y_test_B, y_pred_ridge_B)
    print(f"\nRidge Regression - Evaluation on test set:")
    print(f"  Mean Squared Error (MSE): {mse_ridge_B:.4f}")
    print(f"  R-squared (R2): {r2_ridge_B:.4f}")
    print(f"  Intercept: {ridge_model_B.intercept_:.4f}")
    print(f"  Coefficients: {ridge_model_B.coef_}")

    plt.figure(figsize=(8,6))
    plt.scatter(y_test_B, y_pred_ridge_B, alpha=0.7)
    plt.plot([y_test_B.min(), y_test_B.max()], [y_test_B.min(), y_test_B.max()], 'k--', lw=2)
    plt.xlabel("Actual value of Pain_Rating_T2")
    plt.ylabel("Predicted value of Pain_Rating_T2")
    plt.title("Prediction vs. Actual for Pain_Rating_T2 (Ridge Regression)")
    plt.grid(True)
    plt.show()
else:
    print("Not enough data for Part B after processing.")