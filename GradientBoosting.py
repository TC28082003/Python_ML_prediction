import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier  # Thay thế các model khác
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
print("\n\n--- STEP 1: CLASSIFICATION (Pain Group) USING GRADIENT BOOSTING CLASSIFIER ---")
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

# Scaling numerical features can sometimes help Gradient Boosting, though it's less critical than for SVM/KNN
numerical_transformer_pain_clf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
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

# --- Gradient Boosting Specifics ---
param_grid_pain_clf_gb = {
    'classifier__n_estimators': [250],
    'classifier__learning_rate': [0.1],
    'classifier__max_depth': [4],
    'classifier__subsample': [0.8],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [3]
}

pain_classification_pipeline_base_gb = Pipeline(steps=[
    ('preprocessor', preprocessor_pain_clf),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

min_samples_per_class_train = np.min(np.bincount(y_pain_clf_train_encoded)) if y_pain_clf_train_encoded.size > 0 else 0
if min_samples_per_class_train >= 5:
    n_splits_cv_pain = 5
elif min_samples_per_class_train >= 3:
    n_splits_cv_pain = 3
else:
    n_splits_cv_pain = 2

best_pain_classification_pipeline_gb = None
grid_search_success_gb = False

if n_splits_cv_pain < 2 and len(X_pain_clf_train) >= 1:
    print(
        f"Warning: Not enough samples in the smallest class ({min_samples_per_class_train}) for Stratified CV. Fitting Gradient Boosting model directly.")
    try:
        best_pain_classification_pipeline_gb = pain_classification_pipeline_base_gb.fit(X_pain_clf_train,
                                                                                        y_pain_clf_train_encoded)
        grid_search_success_gb = True
    except Exception as e:
        print(f"Error fitting Gradient Boosting model directly: {e}")
        exit()
elif len(X_pain_clf_train) == 0:
    print("Error: Training data is empty. Cannot train Pain Group classifier (Gradient Boosting).")
    exit()
else:
    cv_stratified_pain = StratifiedKFold(n_splits=n_splits_cv_pain, shuffle=True, random_state=42)
    grid_search_pain_clf_gb = GridSearchCV(pain_classification_pipeline_base_gb, param_grid_pain_clf_gb,
                                           cv=cv_stratified_pain, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
    try:
        grid_search_pain_clf_gb.fit(X_pain_clf_train, y_pain_clf_train_encoded)
        best_pain_classification_pipeline_gb = grid_search_pain_clf_gb.best_estimator_
        print("\nBest Pain Group classification parameters (Gradient Boosting):", grid_search_pain_clf_gb.best_params_)
        grid_search_success_gb = True
    except ValueError as e:
        print(f"Error during GridSearchCV for Pain Group (Gradient Boosting): {e}. Trying to fit base model directly.")
        try:
            best_pain_classification_pipeline_gb = pain_classification_pipeline_base_gb.fit(X_pain_clf_train,
                                                                                            y_pain_clf_train_encoded)
            grid_search_success_gb = True
        except Exception as e_fit:
            print(f"Error fitting base Gradient Boosting model directly after GridSearchCV failed: {e_fit}")
            exit()

if not best_pain_classification_pipeline_gb or not grid_search_success_gb:
    print("Error: Pain Group classification model (Gradient Boosting) could not be trained. Exiting.")
    exit()

# --- Predictions on Training Set (Pain Group - Gradient Boosting) ---
if len(X_pain_clf_train) > 0 and best_pain_classification_pipeline_gb:
    print("\n\n--- PREDICTIONS ON TRAINING SET (Pain Group - Gradient Boosting) ---")
    y_pred_pain_clf_on_train_encoded = best_pain_classification_pipeline_gb.predict(X_pain_clf_train)
    y_train_labels_actual = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_train_encoded)
    y_pred_train_labels = label_encoder_y_pain_clf.inverse_transform(y_pred_pain_clf_on_train_encoded)

    train_predictions_df = pd.DataFrame({
        'Patient_Index_in_Train_X': X_pain_clf_train.index,
        'Real Pain Group': y_train_labels_actual,
        'Predicted Pain Group (GB)': y_pred_train_labels
    })
    print("First 10 predictions on Training Set (GB):")
    print(train_predictions_df.head(10))

    accuracy_train_gb = accuracy_score(y_train_labels_actual, y_pred_train_labels)
    balanced_accuracy_train_gb = balanced_accuracy_score(y_train_labels_actual, y_pred_train_labels)
    print(f"\nAccuracy on Training Set (GB): {accuracy_train_gb * 100:.2f}%")
    print(f"Balanced Accuracy on Training Set (GB): {balanced_accuracy_train_gb * 100:.2f}%")
else:
    print("Warning: Training set is empty or GB model not trained. Cannot make predictions on training data.")

# --- Predictions on Test Set (Pain Group - Gradient Boosting) ---
if len(X_pain_clf_test) > 0 and best_pain_classification_pipeline_gb:
    print("\n\n--- PREDICTIONS ON TEST SET (Pain Group - Gradient Boosting) ---")
    y_pred_pain_clf_on_test_encoded = best_pain_classification_pipeline_gb.predict(X_pain_clf_test)

    y_test_labels_actual = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_test_encoded)
    y_pred_test_labels = label_encoder_y_pain_clf.inverse_transform(y_pred_pain_clf_on_test_encoded)

    test_predictions_df = pd.DataFrame({
        'Patient_Index_in_Test_X': X_pain_clf_test.index,
        'Real Pain Group': y_test_labels_actual,
        'Predicted Pain Group (GB)': y_pred_test_labels
    })
    print("First 10 predictions on Test Set (GB):")
    print(test_predictions_df.head(10))

    accuracy_pain_clf_on_test_gb = accuracy_score(y_test_labels_actual, y_pred_test_labels)
    balanced_accuracy_test_gb = balanced_accuracy_score(y_test_labels_actual, y_pred_test_labels)

    print("\n--- Pain Group Classification Results (Test Set - Gradient Boosting) ---")
    print(f"Test Accuracy (GB): {accuracy_pain_clf_on_test_gb * 100:.2f}%")
    print(f"Test Balanced Accuracy (GB): {balanced_accuracy_test_gb * 100:.2f}%")

    print("Classification Report (Pain Group - GB):")
    report_target_names_ordered = [label for label in pain_labels if label in label_encoder_y_pain_clf.classes_]
    print(classification_report(y_test_labels_actual, y_pred_test_labels,
                                labels=report_target_names_ordered,
                                target_names=report_target_names_ordered,
                                zero_division=0))
else:
    print("Warning: Test set is empty or GB model not trained. Cannot evaluate Pain Group classifier.")

# Plot feature importance for Gradient Boosting classifier
if best_pain_classification_pipeline_gb:
    try:
        classifier_model_gb = best_pain_classification_pipeline_gb.named_steps['classifier']
        preprocessor_fitted_gb = best_pain_classification_pipeline_gb.named_steps['preprocessor']

        if hasattr(classifier_model_gb, 'feature_importances_'):
            try:
                feature_names_transformed_gb = preprocessor_fitted_gb.get_feature_names_out()
            except AttributeError:  # Fallback
                feature_names_transformed_gb = []
                if 'num' in preprocessor_fitted_gb.named_transformers_:
                    feature_names_transformed_gb.extend(numerical_cols_pain_clf)
                if 'cat' in preprocessor_fitted_gb.named_transformers_:
                    cat_pipeline_gb = preprocessor_fitted_gb.named_transformers_['cat']
                    if hasattr(cat_pipeline_gb.named_steps['onehot'], 'get_feature_names_out'):
                        feature_names_transformed_gb.extend(
                            cat_pipeline_gb.named_steps['onehot'].get_feature_names_out(categorical_cols_pain_clf))
                    else:
                        feature_names_transformed_gb.extend(
                            [f"cat_col_gb_{i}" for i in range(len(categorical_cols_pain_clf))])
                if preprocessor_fitted_gb.remainder == 'passthrough':
                    remainder_cols_gb = [X_pain_clf.columns[i] for i in preprocessor_fitted_gb._remainder[2]]
                    feature_names_transformed_gb.extend(remainder_cols_gb)

            importances_gb = classifier_model_gb.feature_importances_

            if len(feature_names_transformed_gb) == len(importances_gb):
                feature_importance_df_gb = pd.DataFrame(
                    {'feature': feature_names_transformed_gb, 'importance': importances_gb})
                feature_importance_df_gb = feature_importance_df_gb.sort_values(by='importance',
                                                                                ascending=False)

                plt.figure(figsize=(12, 10))
                plt.barh(feature_importance_df_gb['feature'][:15],
                         feature_importance_df_gb['importance'][:15])
                plt.gca().invert_yaxis()
                plt.xlabel("Importance")
                plt.ylabel("Feature (Gradient Boosting Classification)")
                plt.title("Top 15 Important Features (Gradient Boosting Classification)")
                plt.tight_layout()
                plt.show()
            else:
                print(
                    f"Warning: Mismatch between number of transformed feature names ({len(feature_names_transformed_gb)}) "
                    f"and importances ({len(importances_gb)}) for Gradient Boosting. Skipping feature importance plot.")
        else:
            print("Gradient Boosting model does not have feature_importances_ attribute.")
    except Exception as e:
        print(f"Error plotting feature importance for Gradient Boosting classifier: {e}")

print("\n--- End of Script (Gradient Boosting Version) ---")