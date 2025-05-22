import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC  # Thay thế cho RandomForestClassifier hoặc LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import warnings

# Suppress general UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Step 0: Load Data ---
file_path = 'Pain_python.csv'
try:
    df_original = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

print("--- Original Data (First 5 rows) ---")
print(df_original.head())
print("\n--- Original Data Info ---")
df_original.info()

# --- Step 1: Classification (Predict Pain Group: A, B, C, D, E) ---
print("\n\n--- STEP 1: CLASSIFICATION (Pain Group) USING SUPPORT VECTOR MACHINES (SVM) ---")
df_pain_clf_input = df_original.copy()

# Create 'Pain_Group' column
pain_bins = [0, 2, 4, 6, 8, 10]
pain_labels = ['E (1-2)', 'D (3-4)', 'C (5-6)', 'B (7-8)', 'A (9-10)']

df_pain_clf_input['Pain rating'] = pd.to_numeric(df_pain_clf_input['Pain rating'], errors='coerce')
df_pain_clf_input.dropna(subset=['Pain rating'], inplace=True)
df_pain_clf_input['Pain rating'] = df_pain_clf_input['Pain rating'].astype(int)

df_pain_clf_input['Pain_Group'] = pd.cut(df_pain_clf_input['Pain rating'],
                                         bins=pain_bins,
                                         labels=pain_labels,
                                         right=True,
                                         include_lowest=True)
df_pain_clf_input.dropna(subset=['Pain_Group'], inplace=True)
df_pain_clf_input['Pain_Group'] = pd.Categorical(df_pain_clf_input['Pain_Group'], categories=pain_labels, ordered=True)

print("\n--- Value Counts for Pain_Group ---")
print(df_pain_clf_input['Pain_Group'].value_counts().sort_index())

target_column_pain_clf = 'Pain_Group'
features_to_drop_for_pain_clf = ['Number', 'Pain rating', target_column_pain_clf]
X_pain_clf = df_pain_clf_input.drop(columns=features_to_drop_for_pain_clf, axis=1, errors='ignore')
y_pain_clf = df_pain_clf_input[target_column_pain_clf]

if X_pain_clf.empty or y_pain_clf.empty:
    print("Error: No data left after creating Pain_Group. Cannot train classifier.")
    exit()

min_samples_for_stratify = 2
value_counts_y = y_pain_clf.value_counts()
classes_to_remove = value_counts_y[value_counts_y < min_samples_for_stratify].index

if not classes_to_remove.empty:
    print(
        f"\nWarning: Removing classes with less than {min_samples_for_stratify} samples for stratification: {list(classes_to_remove)}")
    df_pain_clf_input = df_pain_clf_input[~df_pain_clf_input[target_column_pain_clf].isin(classes_to_remove)]
    X_pain_clf = df_pain_clf_input.drop(columns=features_to_drop_for_pain_clf, axis=1, errors='ignore')
    y_pain_clf = df_pain_clf_input[target_column_pain_clf]
    print("\n--- Value Counts for Pain_Group (After Filtering Small Classes) ---")
    print(y_pain_clf.value_counts().sort_index())

if X_pain_clf.empty or y_pain_clf.empty or len(y_pain_clf.unique()) < 2:
    print(
        "Error: Not enough data or classes left after filtering small classes for Pain Group. Cannot train classifier.")
    exit()

label_encoder_y_pain_clf = LabelEncoder()
label_encoder_y_pain_clf.fit(y_pain_clf)
y_pain_clf_encoded = label_encoder_y_pain_clf.transform(y_pain_clf)

print("\n--- Encoded Target Classes (Pain Group Classification) ---")
encoded_class_mapping = {}
for label in pain_labels:
    if label in label_encoder_y_pain_clf.classes_:
        encoded_class_mapping[label] = label_encoder_y_pain_clf.transform([label])[0]
for class_name, encoded_value in encoded_class_mapping.items():
    print(f"   {class_name} -> {encoded_value}")

numerical_cols_pain_clf = X_pain_clf.select_dtypes(include=np.number).columns.tolist()
categorical_cols_pain_clf = X_pain_clf.select_dtypes(include='object').columns.tolist()

numerical_transformer_pain_clf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # StandardScaler is very important for SVM
])
categorical_transformer_pain_clf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor_pain_clf = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_pain_clf, numerical_cols_pain_clf),
        ('cat', categorical_transformer_pain_clf, categorical_cols_pain_clf)
    ],
    remainder='passthrough'
)

X_pain_clf_train, X_pain_clf_test, y_pain_clf_train_encoded, y_pain_clf_test_encoded = train_test_split(
    X_pain_clf, y_pain_clf_encoded, test_size=0.2, random_state=42, stratify=y_pain_clf_encoded
)

# --- Hiển thị mẫu Training/Test Set ---
print("\n--- Training Set Sample (Pain Group Classification) ---")
print("X_train (first 5 rows):")
print(X_pain_clf_train.head())
print("\ny_train (first 5 values, encoded):")
print(y_pain_clf_train_encoded[:5])
if len(y_pain_clf_train_encoded) > 0:
    y_train_labels_sample = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_train_encoded[:5])
    print("\ny_train (first 5 values, original labels):")
    print(y_train_labels_sample)

print("\n--- Test Set Sample (Pain Group Classification) ---")
print("X_test (first 5 rows):")
print(X_pain_clf_test.head())
print("\ny_test (first 5 values, encoded):")
print(y_pain_clf_test_encoded[:5])
if len(y_pain_clf_test_encoded) > 0:
    y_test_labels_sample = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_test_encoded[:5])
    print("\ny_test (first 5 values, original labels):")
    print(y_test_labels_sample)
# --- Kết thúc hiển thị mẫu ---


if len(np.unique(y_pain_clf_train_encoded)) < 2 or len(X_pain_clf_train) < 5:
    print("Error: Not enough samples or classes in the training data for Pain Group classification. Exiting.")
    exit()

# --- SVM Specifics ---
param_grid_pain_clf_svm = {
    'classifier__C': [0.1, 1, 10, 50],  # Regularization parameter
    'classifier__kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'classifier__gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf' and 'poly'
    'classifier__degree': [2, 3]  # Degree for 'poly' kernel (only used if kernel is 'poly')
}

# probability=True is needed for predict_proba, but can slow down training.
# If predict_proba is not strictly needed for final output, can set to False for faster GridSearchCV.
pain_classification_pipeline_base_svm = Pipeline(steps=[
    ('preprocessor', preprocessor_pain_clf),
    ('classifier', SVC(random_state=42, class_weight='balanced', probability=True))
])

min_samples_per_class_train = np.min(np.bincount(y_pain_clf_train_encoded)) if y_pain_clf_train_encoded.size > 0 else 0
n_splits_cv_pain = int(min(3, min_samples_per_class_train)) if min_samples_per_class_train > 1 else 2

best_pain_classification_pipeline_svm = None
grid_search_success_svm = False

if n_splits_cv_pain < 2 and len(X_pain_clf_train) >= 1:
    print(
        f"Warning: Not enough samples in the smallest class ({min_samples_per_class_train}) for Stratified CV (need at least {n_splits_cv_pain} for {n_splits_cv_pain}-fold CV). Fitting SVM model directly.")
    try:
        # For direct fitting, we might choose simpler SVC parameters if GridSearchCV is skipped
        # However, using the base pipeline is a reasonable default
        best_pain_classification_pipeline_svm = pain_classification_pipeline_base_svm.fit(X_pain_clf_train,
                                                                                          y_pain_clf_train_encoded)
        grid_search_success_svm = True
    except Exception as e:
        print(f"Error fitting SVM model directly: {e}")
        exit()
elif len(X_pain_clf_train) == 0:
    print("Error: Training data is empty. Cannot train Pain Group classifier (SVM).")
    exit()
else:
    cv_stratified_pain = StratifiedKFold(n_splits=n_splits_cv_pain, shuffle=True, random_state=42)
    grid_search_pain_clf_svm = GridSearchCV(pain_classification_pipeline_base_svm, param_grid_pain_clf_svm,
                                            cv=cv_stratified_pain, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
    try:
        grid_search_pain_clf_svm.fit(X_pain_clf_train, y_pain_clf_train_encoded)
        best_pain_classification_pipeline_svm = grid_search_pain_clf_svm.best_estimator_
        print("\nBest Pain Group classification parameters (SVM):", grid_search_pain_clf_svm.best_params_)
        grid_search_success_svm = True
    except ValueError as e:
        print(f"Error during GridSearchCV for Pain Group (SVM): {e}. Trying to fit base model directly.")
        try:
            best_pain_classification_pipeline_svm = pain_classification_pipeline_base_svm.fit(X_pain_clf_train,
                                                                                              y_pain_clf_train_encoded)
            grid_search_success_svm = True
        except Exception as e_fit:
            print(f"Error fitting base SVM model directly after GridSearchCV failed: {e_fit}")
            exit()

if not best_pain_classification_pipeline_svm or not grid_search_success_svm:
    print("Error: Pain Group classification model (SVM) could not be trained. Exiting.")
    exit()

# --- Predictions on Training Set (Pain Group - SVM) ---
if len(X_pain_clf_train) > 0 and best_pain_classification_pipeline_svm:
    print("\n\n--- PREDICTIONS ON TRAINING SET (Pain Group - SVM) ---")
    y_pred_pain_clf_on_train_encoded = best_pain_classification_pipeline_svm.predict(X_pain_clf_train)
    y_train_labels_actual = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_train_encoded)
    y_pred_train_labels = label_encoder_y_pain_clf.inverse_transform(y_pred_pain_clf_on_train_encoded)

    train_predictions_df = pd.DataFrame({
        'Patient_Index_in_Train_X': X_pain_clf_train.index,
        'Real Pain Group': y_train_labels_actual,
        'Predicted Pain Group (SVM)': y_pred_train_labels
    })
    print("First 10 predictions on Training Set (SVM):")
    print(train_predictions_df.head(10))

    accuracy_train_svm = accuracy_score(y_train_labels_actual, y_pred_train_labels)
    balanced_accuracy_train_svm = balanced_accuracy_score(y_train_labels_actual, y_pred_train_labels)
    print(f"\nAccuracy on Training Set (SVM): {accuracy_train_svm * 100:.2f}%")
    print(f"Balanced Accuracy on Training Set (SVM): {balanced_accuracy_train_svm * 100:.2f}%")
else:
    print("Warning: Training set is empty or SVM model not trained. Cannot make predictions on training data.")

# --- Predictions on Test Set (Pain Group - SVM) ---
if len(X_pain_clf_test) > 0 and best_pain_classification_pipeline_svm:
    print("\n\n--- PREDICTIONS ON TEST SET (Pain Group - SVM) ---")
    y_pred_pain_clf_on_test_encoded = best_pain_classification_pipeline_svm.predict(X_pain_clf_test)

    y_test_labels_actual = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_test_encoded)
    y_pred_test_labels = label_encoder_y_pain_clf.inverse_transform(y_pred_pain_clf_on_test_encoded)

    test_predictions_df = pd.DataFrame({
        'Patient_Index_in_Test_X': X_pain_clf_test.index,
        'Real Pain Group': y_test_labels_actual,
        'Predicted Pain Group (SVM)': y_pred_test_labels
    })
    print("First 10 predictions on Test Set (SVM):")
    print(test_predictions_df.head(10))

    accuracy_pain_clf_on_test_svm = accuracy_score(y_test_labels_actual, y_pred_test_labels)
    balanced_accuracy_test_svm = balanced_accuracy_score(y_test_labels_actual, y_pred_test_labels)

    print("\n--- Pain Group Classification Results (Test Set - SVM) ---")
    print(f"Test Accuracy (SVM): {accuracy_pain_clf_on_test_svm * 100:.2f}%")
    print(f"Test Balanced Accuracy (SVM): {balanced_accuracy_test_svm * 100:.2f}%")

    print("Classification Report (Pain Group - SVM):")
    report_target_names_ordered = [label for label in pain_labels if label in label_encoder_y_pain_clf.classes_]
    print(classification_report(y_test_labels_actual, y_pred_test_labels,
                                labels=report_target_names_ordered,
                                target_names=report_target_names_ordered,
                                zero_division=0))
else:
    print("Warning: Test set is empty or SVM model not trained. Cannot evaluate Pain Group classifier.")

# Feature Importance for SVM (only for linear kernel)
if best_pain_classification_pipeline_svm:
    try:
        classifier_model_svm = best_pain_classification_pipeline_svm.named_steps['classifier']

        # Feature importance is typically straightforward only for linear SVM
        if hasattr(classifier_model_svm, 'coef_') and classifier_model_svm.kernel == 'linear':
            preprocessor_fitted_svm = best_pain_classification_pipeline_svm.named_steps['preprocessor']
            try:
                feature_names_transformed_svm = preprocessor_fitted_svm.get_feature_names_out()
            except AttributeError:  # Fallback
                feature_names_transformed_svm = []
                if 'num' in preprocessor_fitted_svm.named_transformers_:
                    feature_names_transformed_svm.extend(numerical_cols_pain_clf)
                if 'cat' in preprocessor_fitted_svm.named_transformers_:
                    cat_pipeline_svm = preprocessor_fitted_svm.named_transformers_['cat']
                    if hasattr(cat_pipeline_svm.named_steps['onehot'], 'get_feature_names_out'):
                        feature_names_transformed_svm.extend(
                            cat_pipeline_svm.named_steps['onehot'].get_feature_names_out(categorical_cols_pain_clf))
                    else:
                        feature_names_transformed_svm.extend(
                            [f"cat_col_svm_{i}" for i in range(len(categorical_cols_pain_clf))])
                if preprocessor_fitted_svm.remainder == 'passthrough':
                    remainder_cols_svm = [X_pain_clf.columns[i] for i in preprocessor_fitted_svm._remainder[2]]
                    feature_names_transformed_svm.extend(remainder_cols_svm)

            coefficients_svm = classifier_model_svm.coef_

            if coefficients_svm.shape[1] == len(feature_names_transformed_svm):
                if coefficients_svm.shape[0] == 1:  # Binary
                    importances_svm = np.abs(coefficients_svm[0])
                else:  # Multiclass (OvO or OvR, coef_ shape depends on strategy)
                    # For OvR, shape is (n_classes, n_features)
                    # For OvO, shape is (n_classes * (n_classes - 1) / 2, n_features)
                    # We'll take mean of absolute values if multiclass for simplicity,
                    # acknowledging this is a simplification for OvO.
                    importances_svm = np.mean(np.abs(coefficients_svm), axis=0)

                feature_importance_df_svm = pd.DataFrame(
                    {'feature': feature_names_transformed_svm, 'importance_abs_coef': importances_svm})
                feature_importance_df_svm = feature_importance_df_svm.sort_values(by='importance_abs_coef',
                                                                                  ascending=False)

                plt.figure(figsize=(12, 10))
                plt.barh(feature_importance_df_svm['feature'][:15],
                         feature_importance_df_svm['importance_abs_coef'][:15])
                plt.gca().invert_yaxis()
                plt.xlabel("Mean Absolute Coefficient Value (Linear SVM Importance)")
                plt.ylabel("Feature (SVM - Linear Kernel)")
                plt.title("Top 15 Important Features (SVM - Linear Kernel)")
                plt.tight_layout()
                plt.show()
            else:
                print(
                    f"Warning: Mismatch in SVM (linear) feature names ({len(feature_names_transformed_svm)}) and coefficients ({coefficients_svm.shape[1]}). Skipping SVM feature importance.")
        elif classifier_model_svm.kernel != 'linear':
            print(
                f"\nFeature importance plot is typically shown for SVM with 'linear' kernel. Current kernel is '{classifier_model_svm.kernel}'.")
            print(
                "For non-linear kernels (like RBF or Poly), feature importances are not directly available as coefficients.")
            print("Techniques like permutation importance can be used but are more computationally intensive.")
        else:
            print(
                "SVM model does not have 'coef_' attribute or kernel is not linear. Cannot plot feature importances directly.")
    except Exception as e:
        print(f"Error plotting feature importance for SVM classifier: {e}")
print("\n--- End of Script (SVM Version) ---")