import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge # Add Ridge for regularization
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- 1. DATA LOADING AND PREPARATION ---
# ==============================================================================
print("--- 1. LOADING AND PREPARING DATA ---")
file_path = 'Pain_python.csv'
df = pd.read_csv(file_path)
df.rename(columns={'Q28. JointP': 'Joint_Pain', 'Pain rating': 'Pain_Rating',
                   'CD4%': 'CD4_Percent', '%Trans sat': 'Trans_Sat_Percent',
                   'Transf': 'Transferrin_Actual'},
          inplace=True)
# Convert relevant columns to numeric, coercing errors
cols_to_numeric = ['Lymphocytes', 'Age', 'Pain_Rating', 'Neutrophils', 'WCC', 'CRP', 'HgB', 'Platelets',
                   'CD3Total', 'CD8-Suppr', 'CD4_Percent', 'CD4-Helper', 'CD19 Bcell', 'Ferritin', 'CK', 'Iron',
                   'Trans_Sat_Percent', 'Transferrin_Actual']
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Pivot data to pair T0 and T2 values for selected variables
vars_for_delta = ['Pain_Rating', 'Transferrin_Actual', 'CD4_Percent',
                  'Platelets', 'Neutrophils', 'Trans_Sat_Percent']
df_pivoted_list = []
for var in vars_for_delta:
    if var in df.columns:
        df_pivot_var = df.pivot_table(index='Number', columns='Timepoint', values=var)
        df_pivot_var.columns = [f'{var}_T0', f'{var}_T2'] # Rename columns to specify timepoint
        df_pivoted_list.append(df_pivot_var)

# Merge pivoted tables
df_paired = df_pivoted_list[0]
for i in range(1, len(df_pivoted_list)):
    df_paired = pd.merge(df_paired, df_pivoted_list[i], on='Number', how='inner')
df_paired.reset_index(inplace=True)

# Calculate the change (T2 - T0) for each variable
for var in vars_for_delta:
    col_t0 = f'{var}_T0'; col_t2 = f'{var}_T2'
    if col_t0 in df_paired.columns and col_t2 in df_paired.columns:
        df_paired[f'{var}_Change'] = df_paired[col_t2] - df_paired[col_t0]
print("--- Data Preparation Complete ---")

# ==============================================================================
# --- PART A: PREDICTING THE CHANGE IN PAIN LEVEL (Pain_Rating_Change) ---
# ==============================================================================
print("\n--- PART A: PREDICTING THE CHANGE IN PAIN LEVEL (Pain_Rating_Change) ---")
# Define features (X) and target (y) for prediction
cols_for_prediction_A = ['Pain_Rating_Change', 'Transferrin_Actual_Change', 'CD4_Percent_Change',
                         'Platelets_Change', 'Neutrophils_Change', 'Trans_Sat_Percent_Change']
data_A = df_paired.dropna(subset=cols_for_prediction_A).copy()

if not data_A.empty and len(data_A) > 10: # Need enough data to split
    y_A = data_A['Pain_Rating_Change']
    X_A = data_A[['Transferrin_Actual_Change', 'CD4_Percent_Change',
                  'Platelets_Change', 'Neutrophils_Change', 'Trans_Sat_Percent_Change']]

    # Split data into training and test sets
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

    # --- Part A.1: Robust Linear Model (RLM) ---
    print("\n--- Part A.1: Robust Linear Model (RLM) ---")
    X_train_A_const = sm.add_constant(X_train_A) # Add constant for intercept
    rlm_model_final_A = sm.RLM(y_train_A, X_train_A_const, M=sm.robust.norms.HuberT()).fit()
    print("\nRLM Model (with constant) Summary (trained on training set):")
    print(rlm_model_final_A.summary())

    # Evaluate RLM on the test set
    X_test_A_const = sm.add_constant(X_test_A)
    y_pred_rlm_A = rlm_model_final_A.predict(X_test_A_const)
    mse_rlm_A = mean_squared_error(y_test_A, y_pred_rlm_A)
    r2_rlm_A = r2_score(y_test_A, y_pred_rlm_A)
    print(f"\nRLM - Evaluation on test set:")
    print(f"  Mean Squared Error (MSE): {mse_rlm_A:.4f}")
    print(f"  R-squared (R2): {r2_rlm_A:.4f}") # R2 can be negative

    # --- Part A.2: Ordinary Linear Regression (Scikit-learn) ---
    print("\n--- Part A.2: Ordinary Linear Regression (Scikit-learn) ---")
    lr_model_A = LinearRegression()
    lr_model_A.fit(X_train_A, y_train_A)
    y_pred_lr_A = lr_model_A.predict(X_test_A) # Predictions on the test set

    # Evaluate Linear Regression on the test set
    mse_lr_A = mean_squared_error(y_test_A, y_pred_lr_A)
    r2_lr_A = r2_score(y_test_A, y_pred_lr_A)
    print(f"\nLinear Regression (scikit-learn) - Evaluation on test set:")
    print(f"  Mean Squared Error (MSE): {mse_lr_A:.4f}")
    print(f"  R-squared (R2): {r2_lr_A:.4f}")
    print(f"  Intercept: {lr_model_A.intercept_:.4f}")
    print(f"  Coefficients: {lr_model_A.coef_}")

    # --- Part A.3: Linear Regression - Detailed Prediction Analysis (Task 1 User Feedback) ---
    print("\n--- Part A.3: Linear Regression - Detailed Prediction Analysis ---")
    # Calculate absolute error for the accuracy criterion
    abs_error_A = np.abs(y_pred_lr_A - y_test_A)

    # Create a DataFrame for comparing actual vs. predicted values and the error criterion
    comparison_df_A = pd.DataFrame({
        'Actual Pain Change': y_test_A,
        'Predicted Pain Change': y_pred_lr_A,
        'Absolute Error': abs_error_A,
        'Error < 1.0': abs_error_A < 1.0
    })
    print("\nComparison of Actual vs. Predicted Pain_Rating_Change (Linear Regression):")
    print(comparison_df_A.to_string())

    # Report on the accuracy criterion (absolute error < 1.0)
    count_within_threshold = np.sum(abs_error_A < 1.0)
    percentage_within_threshold = (count_within_threshold / len(y_test_A)) * 100
    print(f"\nNumber of test samples where abs(Predicted Change - Actual Change) < 1.0: {count_within_threshold}")
    print(f"Percentage of test samples meeting this criterion: {percentage_within_threshold:.2f}%")

    # --- Part A.4: Linear Regression - Prediction Error Analysis (Task 2) ---
    print("\n--- Part A.4: Linear Regression - Prediction Error Analysis (Task 2) ---")
    errors_A = y_test_A - y_pred_lr_A # Prediction error (Actual - Predicted)

    # 1. Visualize Error Distribution
    print("\nGenerating plot: Distribution of Prediction Errors...")
    plt.figure(figsize=(10, 6)) # Slightly wider for better title spacing
    sns.histplot(errors_A, kde=True)
    plt.title("Distribution of Prediction Errors (Pain_Rating_Change, Linear Regression)")
    plt.xlabel("Prediction Error (Actual Value - Predicted Value)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 2. Analyze Error Relationship with Biomarkers
    biomarkers_A_cols = X_A.columns # Using X_A.columns as X_test_A has the same columns
    print("\nAnalyzing relationship between prediction errors and biomarkers:")
    for biomarker_name in biomarkers_A_cols:
        print(f"\nGenerating plot: Prediction Error vs. {biomarker_name}...")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test_A[biomarker_name], errors_A, alpha=0.7)
        plt.title(f"Prediction Error vs. {biomarker_name} (Pain_Rating_Change)")
        plt.xlabel(f"Change in {biomarker_name}")
        plt.ylabel("Prediction Error (Actual - Predicted)")
        plt.grid(True)
        plt.show()

        # Calculate and print Pearson correlation coefficient
        correlation = errors_A.corr(X_test_A[biomarker_name]) # errors_A is a Series, X_test_A[biomarker_name] is a Series
        print(f"Pearson Correlation between error and {biomarker_name}: {correlation:.4f}")

    # --- Part A.5: Linear Regression - Visualize Overall Predictions vs. Actuals ---
    print("\nGenerating plot: Actual vs. Predicted Pain_Rating_Change...")
    plt.figure(figsize=(10, 6)) # Slightly wider
    plt.scatter(y_test_A, y_pred_lr_A, alpha=0.7, label='Linear Regression Predictions')
    plt.plot([y_test_A.min(), y_test_A.max()], [y_test_A.min(), y_test_A.max()], 'k--', lw=2, label='Perfect Prediction Line')
    plt.xlabel("Actual Value of Pain_Rating_Change")
    plt.ylabel("Predicted Value of Pain_Rating_Change")
    plt.title("Actual vs. Predicted Values for Pain_Rating_Change (Linear Regression)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Not enough data available for Part A model training and evaluation after processing.")

# ==============================================================================
# --- PART B: PREDICT PAIN_RATING AT T2 (Commented Out) ---
# ==============================================================================
# print("\n\n--- PART B: PREDICTING PAIN LEVEL AT T2 (Pain_Rating_T2) ---")
# # Use biomarkers at T0 and T2, and Pain_Rating at T0 to predict Pain_Rating at T2
# cols_for_prediction_B = ['Pain_Rating_T2', 'Pain_Rating_T0', # Pain_Rating_T0 is an important feature
#                          'Transferrin_Actual_T0', 'CD4_Percent_T0', 'Platelets_T0', 'Neutrophils_T0', 'Trans_Sat_Percent_T0',
#                          'Transferrin_Actual_T2', 'CD4_Percent_T2', 'Platelets_T2', 'Neutrophils_T2', 'Trans_Sat_Percent_T2']
# # Or just use _Change:
# # cols_for_prediction_B = ['Pain_Rating_T2', 'Pain_Rating_T0',
# # 'Transferrin_Actual_Change', 'CD4_Percent_Change', 'Platelets_Change',
# # 'Neutrophils_Change', 'Trans_Sat_Percent_Change']
#
# data_B = df_paired.dropna(subset=cols_for_prediction_B).copy()
#
# if not data_B.empty and len(data_B) > 10:
#     y_B = data_B['Pain_Rating_T2']
#     X_B_feature_cols = [col for col in cols_for_prediction_B if col != 'Pain_Rating_T2']
#     X_B = data_B[X_B_feature_cols]
#
#     X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
#
#     # Use Ridge Regression (Linear regression with L2 regularization to prevent overfitting)
#     ridge_model_B = Ridge(alpha=1.0) # alpha is the regularization parameter
#     ridge_model_B.fit(X_train_B, y_train_B)
#     y_pred_ridge_B = ridge_model_B.predict(X_test_B)
#     mse_ridge_B = mean_squared_error(y_test_B, y_pred_ridge_B)
#     r2_ridge_B = r2_score(y_test_B, y_pred_ridge_B)
#     print(f"\nRidge Regression - Evaluation on test set:")
#     print(f"  Mean Squared Error (MSE): {mse_ridge_B:.4f}")
#     print(f"  R-squared (R2): {r2_ridge_B:.4f}")
#     print(f"  Intercept: {ridge_model_B.intercept_:.4f}")
#     print(f"  Coefficients: {ridge_model_B.coef_}")
#
#     plt.figure(figsize=(8,6))
#     plt.scatter(y_test_B, y_pred_ridge_B, alpha=0.7)
#     plt.plot([y_test_B.min(), y_test_B.max()], [y_test_B.min(), y_test_B.max()], 'k--', lw=2)
#     plt.xlabel("Actual value of Pain_Rating_T2")
#     plt.ylabel("Predicted value of Pain_Rating_T2")
#     plt.title("Prediction vs. Actual for Pain_Rating_T2 (Ridge Regression)")
#     plt.grid(True)
#     plt.show()
# else:
#     print("Not enough data for Part B after processing.")