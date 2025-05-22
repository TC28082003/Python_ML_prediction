import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# --- Hàm MAPE ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return 0.0 if np.all(y_pred == 0) else np.inf
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    if len(y_true_filtered) == 0:
        return np.nan
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


# --- Bước 1: Tiền xử lý dữ liệu ---
print("--- Bước 1: Tiền xử lý dữ liệu ---")
df_raw = pd.read_csv("Pain_python.csv")  # Sử dụng tên tệp bạn đã cung cấp
df_t0 = df_raw[df_raw['Timepoint'] == 'T0'].copy()
print(f"Kích thước ban đầu của df_t0: {df_t0.shape}")

TARGET_COL = "Pain rating"

df_t0[TARGET_COL] = pd.to_numeric(df_t0[TARGET_COL], errors='coerce')
nan_target_rows_count = df_t0[TARGET_COL].isnull().sum()
if nan_target_rows_count > 0:
    print(f"CẢNH BÁO: '{TARGET_COL}' có {nan_target_rows_count} NaNs. Xóa các hàng này.")
    df_t0.dropna(subset=[TARGET_COL], inplace=True)
    if df_t0.empty:
        print("LỖI: DataFrame trống sau khi xóa các giá trị NaN của biến mục tiêu. Dừng lại.")
        exit()
df_t0[TARGET_COL] = df_t0[TARGET_COL].astype(float)

# Xử lý đặc trưng: Xóa các cột có tỷ lệ thiếu cao
features_to_check_missing = [col for col in df_t0.columns if col != TARGET_COL]
missing_frac_features = df_t0[features_to_check_missing].isnull().mean()
threshold_missing = 0.5
cols_to_drop_high_missing = missing_frac_features[missing_frac_features > threshold_missing].index.tolist()

if TARGET_COL in cols_to_drop_high_missing:  # Mặc dù không nên xảy ra ở đây
    cols_to_drop_high_missing.remove(TARGET_COL)

if cols_to_drop_high_missing:
    df_t0.drop(columns=cols_to_drop_high_missing, inplace=True)
    print(f"Đã xóa các cột có tỷ lệ thiếu >50%: {cols_to_drop_high_missing}")
else:
    print("Không có cột đặc trưng nào có >50% giá trị bị thiếu.")
print(f"Kích thước sau khi xóa các cột có tỷ lệ thiếu cao: {df_t0.shape}")

# Xác định loại đặc trưng
num_feature_cols = df_t0.select_dtypes(include=np.number).columns.tolist()
if TARGET_COL in num_feature_cols: num_feature_cols.remove(TARGET_COL)
if 'Number' in num_feature_cols: num_feature_cols.remove('Number')

cat_feature_cols = df_t0.select_dtypes(include='object').columns.tolist()
if 'Timepoint' in cat_feature_cols: cat_feature_cols.remove('Timepoint')

# Thay thế giá trị thiếu (Imputation)
if num_feature_cols:
    num_imputer = SimpleImputer(strategy='median')
    df_t0[num_feature_cols] = num_imputer.fit_transform(df_t0[num_feature_cols])
    print(f"Đã thay thế giá trị thiếu cho {len(num_feature_cols)} cột số.")
if cat_feature_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_t0[cat_feature_cols] = cat_imputer.fit_transform(df_t0[cat_feature_cols])
    print(f"Đã thay thế giá trị thiếu cho {len(cat_feature_cols)} cột phân loại.")

# Mã hóa các cột phân loại cụ thể
if 'Gender' in df_t0.columns and df_t0['Gender'].dtype == 'object':
    df_t0['Gender'] = df_t0['Gender'].map({'M': 0, 'F': 1})
    if df_t0['Gender'].isnull().any(): df_t0[['Gender']] = SimpleImputer(strategy='most_frequent').fit_transform(
        df_t0[['Gender']])
    df_t0['Gender'] = df_t0['Gender'].astype(int)
    print("Đã mã hóa 'Gender'")

if 'Q28. JointP' in df_t0.columns and df_t0['Q28. JointP'].dtype == 'object':
    df_t0['Q28. JointP'] = df_t0['Q28. JointP'].map({'N': 0, 'Y': 1})
    if df_t0['Q28. JointP'].isnull().any(): df_t0[['Q28. JointP']] = SimpleImputer(
        strategy='most_frequent').fit_transform(df_t0[['Q28. JointP']])
    df_t0['Q28. JointP'] = df_t0['Q28. JointP'].astype(int)
    print("Đã mã hóa 'Q28. JointP'")

# Tạo biến giả cho các cột object còn lại
cat_cols_to_dummify = df_t0.select_dtypes(include='object').columns.tolist()
if 'Timepoint' in cat_cols_to_dummify: cat_cols_to_dummify.remove('Timepoint')
if cat_cols_to_dummify:
    df_t0 = pd.get_dummies(df_t0, columns=cat_cols_to_dummify, drop_first=True, dummy_na=False)
    print(f"Đã tạo biến giả cho các cột object còn lại: {cat_cols_to_dummify}")
else:
    print("Không còn cột phân loại nào để tạo biến giả.")

# Chuẩn bị X và y
cols_to_drop_from_X = ['Timepoint', 'Number']
X = df_t0.drop(columns=[TARGET_COL] + [col for col in cols_to_drop_from_X if col in df_t0.columns], errors='ignore')
y = df_t0[TARGET_COL]

if X.isnull().sum().sum() > 0:
    print(f"Cảnh báo: X có {X.isnull().sum().sum()} NaNs trước khi imputation cuối cùng. Thực hiện imputation...")
    for col in X.columns[X.isnull().any()]: X[[col]] = SimpleImputer(strategy='median').fit_transform(X[[col]])
if y.isnull().sum().sum() > 0 or X.isnull().sum().sum() > 0:
    print("LỖI NGHIÊM TRỌNG: NaNs tồn tại trong X hoặc y sau tất cả quá trình xử lý. Dừng lại.")
    exit()
print(f"Kích thước cuối cùng của X: {X.shape}, y: {y.shape}")
X_columns_list = X.columns.tolist()
print(f"Các đặc trưng để huấn luyện mô hình: {X_columns_list}")

# Phân chia train-test
KFold_N_SPLITS = 5
# Kiểm tra xem có thể phân tầng cho y không
can_stratify_split = y.nunique() > 1 and all(y.value_counts() >= 2)
if not can_stratify_split:
    print(
        "Cảnh báo: Phân tầng cho train_test_split có thể không khả thi hoặc không hiệu quả. Sử dụng phân chia không phân tầng.")
    stratify_option = None
else:
    stratify_option = y

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_option)
except ValueError:
    print("Phân chia phân tầng thất bại mặc dù đã kiểm tra, sử dụng phân chia không phân tầng.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train), columns=X_columns_list, index=X_train.index)
X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X_columns_list, index=X_test.index)

print(f"Kích thước sau khi phân chia/chuẩn hóa: X_train={X_train_scaled_df.shape}, X_test={X_test_scaled_df.shape}")
print("--- Tiền xử lý và Phân chia hoàn tất ---")

# --- Bước 2: Huấn luyện và Đánh giá các Mô hình Cơ sở ---
print("\n--- Bước 2: Huấn luyện và Đánh giá các Mô hình Cơ sở ---")
models_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=5000),
    "ElasticNet": ElasticNet(random_state=42, max_iter=5000),
    "KNeighbors Regressor": KNeighborsRegressor(),
    "SVR (Linear)": SVR(kernel='linear'),
    "SVR (RBF)": SVR(kernel='rbf'),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "XGBoost Regressor": xgb.XGBRegressor(random_state=42, n_jobs=-1, early_stopping_rounds=10),
    # early_stopping_rounds trong constructor
    "LightGBM Regressor": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
}

kf = KFold(n_splits=KFold_N_SPLITS, shuffle=True, random_state=42)
base_results = []
for model_name, model_template in models_dict.items():
    cv_r2, cv_rmse, cv_mape = [], [], []
    print(f"Chạy CV cơ sở cho: {model_name}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled_df, y_train)):
        X_f_train, X_f_val = X_train_scaled_df.iloc[train_idx], X_train_scaled_df.iloc[val_idx]
        y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = clone(model_template)

        if model_name == "XGBoost Regressor":
            # early_stopping_rounds đã được set trong model_template, chỉ cần eval_set
            model.fit(X_f_train, y_f_train, eval_set=[(X_f_val, y_f_val)], verbose=False)
        elif model_name == "LightGBM Regressor":
            model.fit(X_f_train, y_f_train, eval_set=[(X_f_val, y_f_val)],
                      callbacks=[lgb.early_stopping(10, verbose=False)])
        else:
            model.fit(X_f_train, y_f_train)

        y_f_pred = model.predict(X_f_val)
        cv_r2.append(r2_score(y_f_val, y_f_pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y_f_val, y_f_pred)))
        cv_mape.append(mean_absolute_percentage_error(y_f_val, y_f_pred))

    base_results.append({"Model": model_name, "Avg CV R2": np.mean(cv_r2),
                         "Avg CV RMSE": np.mean(cv_rmse), "Avg CV MAPE (%)": np.nanmean(cv_mape)})
base_results_df = pd.DataFrame(base_results).sort_values(by="Avg CV MAPE (%)")
print("\n--- Kết quả CV Cơ sở (Sắp xếp theo MAPE) ---")
print(base_results_df.to_string(index=False))

# --- Bước 3: Tinh chỉnh Siêu Tham Số ---
print("\n--- Bước 3: Tinh chỉnh Siêu Tham Số ---")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

param_grid_gbr = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
    # 'early_stopping_rounds' không cần trong grid nếu đã cố định trong constructor
}

param_grid_lgbm = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [-1, 7, 15],
    'num_leaves': [31, 50, 70],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

models_to_tune = {
    "Random Forest Regressor": (RandomForestRegressor(random_state=42, n_jobs=-1), param_grid_rf),
    "Gradient Boosting Regressor": (GradientBoostingRegressor(random_state=42), param_grid_gbr),
    "XGBoost Regressor": (xgb.XGBRegressor(random_state=42, n_jobs=-1, early_stopping_rounds=10), param_grid_xgb),
    # early_stopping_rounds trong constructor
    "LightGBM Regressor": (lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), param_grid_lgbm)
}

tuned_results_list = []
overall_best_tuned_model = None
overall_best_tuned_model_name = ""
overall_lowest_mape = float('inf')

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(X_train_scaled_df, y_train, test_size=0.2,
                                                                      random_state=123)

for model_name, (model_template, param_grid) in models_to_tune.items():
    print(f"\nTinh chỉnh: {model_name}")
    start_time = time.time()

    fit_params_gs = {}  # GridSearchCV fit_params
    current_estimator = clone(model_template)  # Estimator này đã có early_stopping_rounds cho XGB

    if model_name == "XGBoost Regressor":
        # Chỉ cần eval_set, vì early_stopping_rounds đã ở trong estimator
        fit_params_gs = {"eval_set": [(X_tune_val, y_tune_val)], "verbose": False}
    elif model_name == "LightGBM Regressor":
        fit_params_gs = {"eval_set": [(X_tune_val, y_tune_val)],
                         "callbacks": [lgb.early_stopping(10, verbose=False)]}

    grid_search = GridSearchCV(estimator=current_estimator,
                               param_grid=param_grid, cv=kf,
                               scoring=mape_scorer,
                               n_jobs=-1, verbose=0)

    if fit_params_gs:
        grid_search.fit(X_train_scaled_df, y_train, **fit_params_gs)
    else:
        grid_search.fit(X_train_scaled_df, y_train)

    tuning_duration = time.time() - start_time
    print(f"Tinh chỉnh {model_name} mất {tuning_duration:.2f} giây. Tham số tốt nhất: {grid_search.best_params_}")
    best_tuned_model_instance = grid_search.best_estimator_

    y_test_pred_tuned = best_tuned_model_instance.predict(X_test_scaled_df)
    test_r2_tuned = r2_score(y_test, y_test_pred_tuned)
    test_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_test_pred_tuned))
    test_mape_tuned = mean_absolute_percentage_error(y_test, y_test_pred_tuned)

    print(
        f"  Mô hình {model_name} đã tinh chỉnh trên tập Test: R2={test_r2_tuned:.4f}, RMSE={test_rmse_tuned:.4f}, MAPE={test_mape_tuned:.2f}%")
    tuned_results_list.append({"Model": model_name + " (Tuned)", "Test R2": test_r2_tuned,
                               "Test RMSE": test_rmse_tuned, "Test MAPE (%)": test_mape_tuned,
                               "Best Params": grid_search.best_params_,
                               "Tuning Duration (s)": tuning_duration})
    if not np.isnan(test_mape_tuned) and test_mape_tuned < overall_lowest_mape:
        overall_lowest_mape = test_mape_tuned
        overall_best_tuned_model = best_tuned_model_instance
        overall_best_tuned_model_name = model_name + " (Tuned)"

tuned_summary_df = pd.DataFrame(tuned_results_list).sort_values(by="Test MAPE (%)")
print("\n--- Hiệu Suất Mô Hình Đã Tinh Chỉnh trên Tập Test (Sắp xếp theo MAPE) ---")
print(tuned_summary_df.to_string(index=False))

if overall_best_tuned_model:
    print(
        f"\nMô hình tốt nhất tổng thể đã tinh chỉnh: {overall_best_tuned_model_name} (Test MAPE: {overall_lowest_mape:.2f}%)")
    y_pred_best_overall = overall_best_tuned_model.predict(X_test_scaled_df)

    comparison_df_overall = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_best_overall.round(2),
                                          'Difference': (y_pred_best_overall - y_test.values).round(2)})
    comparison_df_overall['% Error'] = comparison_df_overall.apply(
        lambda r: (r['Difference'] / r['Actual'] * 100).round(2) if r['Actual'] != 0 else np.nan, axis=1)
    print("\nSo sánh Dự đoán vs Thực tế (Mô hình Tốt nhất Tổng thể trên Tập Test - 10 hàng đầu):")
    print(comparison_df_overall.head(10).to_string())

    if hasattr(overall_best_tuned_model, 'feature_importances_'):
        importances = overall_best_tuned_model.feature_importances_
        feature_names_final = X_train_scaled_df.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names_final, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        print("\n--- Độ Quan trọng của Đặc trưng (Mô hình Tốt nhất Tổng thể Đã Tinh Chỉnh) ---")
        print(feature_importance_df.head(15).to_string(index=False))

        plt.figure(figsize=(12, 8))
        plt.title(f'Độ Quan trọng của Đặc trưng cho {overall_best_tuned_model_name}')
        top_n = 15
        plot_df = feature_importance_df.head(top_n)

        plt.barh(plot_df['Feature'], plot_df['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel('Độ quan trọng')
        plt.ylabel('Đặc trưng')
        plt.tight_layout()
        plot_filename = "feature_importances_final.png"
        plt.savefig(plot_filename)
        print(f"\nBiểu đồ độ quan trọng của đặc trưng đã được lưu vào {plot_filename}")

    achieved_mape_target = not np.isnan(overall_lowest_mape) and overall_lowest_mape <= 10.0
    best_rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred_best_overall))
    achieved_rmse_target = best_rmse_overall <= 1.0

    if achieved_mape_target:
        print(f"\nTHÀNH CÔNG: Đã đạt mục tiêu MAPE <= 10% bởi {overall_best_tuned_model_name}!")
    elif achieved_rmse_target:
        print(
            f"\nTHÀNH CÔNG (dựa trên RMSE): Đã đạt mục tiêu RMSE <= 1.0 bởi {overall_best_tuned_model_name} (RMSE: {best_rmse_overall:.4f})!")
    else:
        print(
            f"\nLƯU Ý: Chưa đạt mục tiêu MAPE/RMSE bởi mô hình tốt nhất ({overall_best_tuned_model_name}). MAPE: {overall_lowest_mape:.2f}%, RMSE: {best_rmse_overall:.4f}.")
        print(
            "Có thể cần cải thiện thêm. Xem xét kỹ thuật đặc trưng nâng cao, các loại mô hình khác, hoặc thêm dữ liệu.")
else:
    print(
        "\nKhông có mô hình nào được tinh chỉnh thành công hoặc không có mô hình nào cho kết quả MAPE hợp lệ để so sánh.")

print("\n--- Toàn bộ quá trình hoàn tất ---")