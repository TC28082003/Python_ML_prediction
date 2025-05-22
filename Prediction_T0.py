import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # Add if you want detailed assumption checking and VIF

# --- Step 1: Data Preprocessing ---
print("--- Step 1: Data Preprocessing (Focused on RLM) ---")

# 1. Load and Filter Initial Data
df_raw = pd.read_csv("Pain_python.csv")
df_t0 = df_raw[df_raw['Timepoint'] == 'T0'].copy()

# 2. Remove columns with >50% missing values
threshold_missing = 0.5
missing_frac = df_t0.isnull().mean()
cols_to_drop_missing = missing_frac[missing_frac > threshold_missing].index
df_t0.drop(columns=cols_to_drop_missing, inplace=True)

TARGET_COL = "Pain rating"

# 4. Impute missing values and Ensure data types
num_cols = df_t0.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_original = df_t0.select_dtypes(include=['object']).columns.tolist()

# Handle target variable
if TARGET_COL in num_cols: # If initially numeric
    if df_t0[TARGET_COL].isnull().any():
        df_t0[[TARGET_COL]] = SimpleImputer(strategy='median').fit_transform(df_t0[[TARGET_COL]])
elif TARGET_COL in cat_cols_original or TARGET_COL not in df_t0.columns: # If object or does not exist
    print(f"Warning: '{TARGET_COL}' is not numeric or does not exist. Attempting to convert and impute.")
    df_t0[TARGET_COL] = pd.to_numeric(df_t0[TARGET_COL], errors='coerce')
    df_t0[[TARGET_COL]] = SimpleImputer(strategy='median').fit_transform(df_t0[[TARGET_COL]])
else: # Other cases
     df_t0[TARGET_COL] = pd.to_numeric(df_t0[TARGET_COL], errors='coerce')
     df_t0[[TARGET_COL]] = SimpleImputer(strategy='median').fit_transform(df_t0[[TARGET_COL]])


if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)

if num_cols:
    num_imputer = SimpleImputer(strategy='median')
    df_t0[num_cols] = num_imputer.fit_transform(df_t0[num_cols])

if cat_cols_original:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_t0[cat_cols_original] = cat_imputer.fit_transform(df_t0[cat_cols_original])

# 5. Encode Categorical Variables
if 'Gender' in df_t0.columns and 'Gender' in cat_cols_original:
    gender_map = {'M': 0, 'F': 1}; df_t0['Gender'] = df_t0['Gender'].map(gender_map)
    if df_t0['Gender'].isnull().any(): df_t0[['Gender']] = SimpleImputer(strategy='most_frequent').fit_transform(df_t0[['Gender']])
    df_t0['Gender'] = df_t0['Gender'].astype(int)

if 'Q28. JointP' in df_t0.columns and 'Q28. JointP' in cat_cols_original:
    jointp_map = {'N': 0, 'Y': 1}
    unique_jointp = df_t0['Q28. JointP'].unique()
    if all(val in jointp_map for val in unique_jointp):
        df_t0['Q28. JointP'] = df_t0['Q28. JointP'].map(jointp_map)
        if df_t0['Q28. JointP'].isnull().any(): df_t0[['Q28. JointP']] = SimpleImputer(strategy='most_frequent').fit_transform(df_t0[['Q28. JointP']])
        df_t0['Q28. JointP'] = df_t0['Q28. JointP'].astype(int)

cols_to_dummify = [c for c in cat_cols_original if c not in ['Gender', 'Q28. JointP'] or \
                   (c == 'Q28. JointP' and df_t0['Q28. JointP'].dtype == 'object')]
if cols_to_dummify:
    df_t0 = pd.get_dummies(df_t0, columns=cols_to_dummify, drop_first=True)

X_processed = df_t0.drop(columns=[TARGET_COL, 'Timepoint', 'Number'], errors='ignore')
y_processed = df_t0[TARGET_COL]

# Ensure all columns in X_processed are numeric
for col in X_processed.columns:
    if not pd.api.types.is_numeric_dtype(X_processed[col]):
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        if X_processed[col].isnull().any():
            X_processed[[col]] = SimpleImputer(strategy='median').fit_transform(X_processed[[col]])

# --- Step 2: Feature Selection (Based on initial correlation) ---
print("\n--- Step 2: Feature Selection for RLM ---")
df_for_corr = X_processed.copy()
df_for_corr[TARGET_COL] = y_processed
correlations_with_target = df_for_corr.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL, errors='ignore')
sorted_correlations_abs = correlations_with_target.abs().sort_values(ascending=False)

print(f"\nCorrelation table of features with '{TARGET_COL}' (sorted by absolute magnitude):")
correlation_display_df = pd.DataFrame({
    'Feature': sorted_correlations_abs.index,
    'Correlation_Absolute': sorted_correlations_abs.values,
    'Correlation_Original': correlations_with_target[sorted_correlations_abs.index].values
})
print(correlation_display_df.to_string())

correlation_threshold_rlm = 0.10 # This threshold can be adjusted
features_for_rlm = sorted_correlations_abs[sorted_correlations_abs >= correlation_threshold_rlm].index.tolist()

if not features_for_rlm:
    print(f"\nWARNING: No features have a correlation >= {correlation_threshold_rlm} with '{TARGET_COL}'.")
    print("RLM will not be trained. Consider lowering the threshold or checking the data.")
    exit()

print(f"\nFeatures selected for RLM (correlation >= {correlation_threshold_rlm}): {features_for_rlm}")
X_rlm = X_processed[features_for_rlm]
y_rlm = y_processed

# Train-Test Split
X_rlm_train, X_rlm_test, y_rlm_train, y_rlm_test = train_test_split(X_rlm, y_rlm, test_size=0.2, random_state=42)

# Standardize Features (Very important for RLM coefficient interpretation)
print("\nStandardizing features for RLM...")
scaler_rlm = StandardScaler()
X_rlm_train_scaled = scaler_rlm.fit_transform(X_rlm_train)
X_rlm_test_scaled = scaler_rlm.transform(X_rlm_test)

# Convert back to DataFrame to keep column names
X_rlm_train_scaled_df = pd.DataFrame(X_rlm_train_scaled, columns=features_for_rlm, index=X_rlm_train.index)
X_rlm_test_scaled_df = pd.DataFrame(X_rlm_test_scaled, columns=features_for_rlm, index=X_rlm_test.index)


# --- Step 3: Train RLM Model ---
print("\n--- Step 3: Train RLM Model ---")
# Using LinearRegression for simplicity here, as RLM (Robust Linear Model) from statsmodels
# would require a slightly different setup if strictly following robust regression.
# If the intention was statsmodels.RLM, the fitting process would be:
# X_rlm_train_scaled_df_const = sm.add_constant(X_rlm_train_scaled_df)
# rlm_model_sm = sm.RLM(y_rlm_train, X_rlm_train_scaled_df_const, M=sm.robust.norms.HuberT()).fit()
# For scikit-learn's LinearRegression:
rlm_model = LinearRegression()
rlm_model.fit(X_rlm_train_scaled_df, y_rlm_train)
print("Linear Regression model trained.") # Clarified this is Linear Regression

# --- Step 4: Evaluate RLM Model ---
print("\n--- Step 4: Evaluate RLM Model ---")

# Evaluate on training set
preds_rlm_train = rlm_model.predict(X_rlm_train_scaled_df)
r2_rlm_train = r2_score(y_rlm_train, preds_rlm_train)
rmse_rlm_train = np.sqrt(mean_squared_error(y_rlm_train, preds_rlm_train))
n_train = X_rlm_train_scaled_df.shape[0]
p_train = X_rlm_train_scaled_df.shape[1]
adj_r2_rlm_train = 1 - (1 - r2_rlm_train) * (n_train - 1) / (n_train - p_train - 1)
print(f"Results on Training Set:")
print(f"  R2 Score: {r2_rlm_train:.4f}")
print(f"  Adjusted R2 Score: {adj_r2_rlm_train:.4f}")
print(f"  RMSE: {rmse_rlm_train:.4f}")

# Evaluate on test set
preds_rlm_test = rlm_model.predict(X_rlm_test_scaled_df)
r2_rlm_test = r2_score(y_rlm_test, preds_rlm_test)
rmse_rlm_test = np.sqrt(mean_squared_error(y_rlm_test, preds_rlm_test))
n_test = X_rlm_test_scaled_df.shape[0]
p_test = X_rlm_test_scaled_df.shape[1] # p_test will be equal to p_train
adj_r2_rlm_test = 1 - (1 - r2_rlm_test) * (n_test - 1) / (n_test - p_test - 1)
print(f"\nResults on Test Set:")
print(f"  R2 Score: {r2_rlm_test:.4f}")
print(f"  Adjusted R2 Score: {adj_r2_rlm_test:.4f}")
print(f"  RMSE: {rmse_rlm_test:.4f}")

print("\nPredicted vs. Actual Pain Ratings (Test Set):")
# y_rlm_test là một pandas Series với index gốc. preds_rlm_test là một mảng NumPy.
# Tạo DataFrame để so sánh. Index của y_rlm_test sẽ được sử dụng cho DataFrame mới.
comparison_df = pd.DataFrame({
    'Actual Pain Rating': y_rlm_test,
    'Predicted Pain Rating': preds_rlm_test
})
# Thêm cột chênh lệch để dễ phân tích
comparison_df['Difference (Predicted - Actual)'] = comparison_df['Predicted Pain Rating'] - comparison_df['Actual Pain Rating']

# In toàn bộ DataFrame (tất cả các hàng)
print(comparison_df.to_string())


# --- Step 5: Analyze Coefficients ---
print("\n--- Step 5: Analyze RLM Coefficients ---")
coefficients = rlm_model.coef_
intercept = rlm_model.intercept_

# Create DataFrame to display coefficients
coef_df = pd.DataFrame(coefficients, index=features_for_rlm, columns=['Coefficient'])
# Sort by absolute value of coefficient to see the magnitude of influence
coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs()
coef_df_sorted = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)

print(f"\nIntercept: {intercept:.4f}")
print("\nCoefficients of Variables (sorted by absolute influence):")
print(coef_df_sorted.to_string())

print("\nExample interpretation (for the variable with the largest coefficient):")
if not coef_df_sorted.empty:
    top_feature = coef_df_sorted.index[0]
    top_coef = coef_df_sorted['Coefficient'].iloc[0]
    print(f"  When '{top_feature}' increases by 1 standard deviation (as data is standardized),")
    print(f"  and other variables are held constant, '{TARGET_COL}' is predicted to {'increase' if top_coef > 0 else 'decrease'} by approximately {abs(top_coef):.4f} units.")


# --- (Optional) Step 6: Check RLM assumptions with statsmodels ---
print("\n--- (Optional) Step 6: Check RLM Assumptions with statsmodels ---")
# Add constant (intercept) to X_train for statsmodels
X_rlm_train_sm = sm.add_constant(X_rlm_train_scaled_df) # Use scaled data for consistency
model_sm = sm.OLS(y_rlm_train, X_rlm_train_sm).fit()
print(model_sm.summary())

# Check VIF (multicollinearity)
from statsmodels.stats.outliers_influence import variance_inflation_factor
if X_rlm_train_scaled_df.shape[1] > 1: # Need at least 2 features
     vif_data = pd.DataFrame()
     vif_data["feature"] = X_rlm_train_scaled_df.columns
     vif_data["VIF"] = [variance_inflation_factor(X_rlm_train_scaled_df.values, i) for i in range(X_rlm_train_scaled_df.shape[1])]
     print("\nVariance Inflation Factor (VIF):")
     print(vif_data.sort_values(by='VIF', ascending=False).to_string())
     print(" (VIF > 5 or 10 may indicate significant multicollinearity)")
else:
     print("\nNot enough features to calculate VIF.")

print("\n--- Program finished ---")