import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression  # Thay thế RandomForestClassifier
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
print("\n\n--- STEP 1: CLASSIFICATION (Pain Group) USING LOGISTIC REGRESSION ---")
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

# --- Logistic Regression Specifics ---
param_grid_pain_clf_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']  # 'liblinear' is good for smaller datasets and supports L1/L2
}

pain_classification_pipeline_base_lr = Pipeline(steps=[
    ('preprocessor', preprocessor_pain_clf),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])

min_samples_per_class_train = np.min(np.bincount(y_pain_clf_train_encoded)) if y_pain_clf_train_encoded.size > 0 else 0
n_splits_cv_pain = int(min(3, min_samples_per_class_train)) if min_samples_per_class_train > 1 else 2

best_pain_classification_pipeline_lr = None
grid_search_success = False

if n_splits_cv_pain < 2 and len(X_pain_clf_train) >= 1:
    print(
        f"Warning: Not enough samples in the smallest class ({min_samples_per_class_train}) for Stratified CV (need at least {n_splits_cv_pain} for {n_splits_cv_pain}-fold CV). Fitting Logistic Regression model directly.")
    try:
        best_pain_classification_pipeline_lr = pain_classification_pipeline_base_lr.fit(X_pain_clf_train,
                                                                                        y_pain_clf_train_encoded)
        grid_search_success = True
    except Exception as e:
        print(f"Error fitting Logistic Regression model directly: {e}")
        exit()
elif len(X_pain_clf_train) == 0:
    print("Error: Training data is empty. Cannot train Pain Group classifier (Logistic Regression).")
    exit()
else:
    cv_stratified_pain = StratifiedKFold(n_splits=n_splits_cv_pain, shuffle=True, random_state=42)
    grid_search_pain_clf_lr = GridSearchCV(pain_classification_pipeline_base_lr, param_grid_pain_clf_lr,
                                           cv=cv_stratified_pain, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
    try:
        grid_search_pain_clf_lr.fit(X_pain_clf_train, y_pain_clf_train_encoded)
        best_pain_classification_pipeline_lr = grid_search_pain_clf_lr.best_estimator_
        print("\nBest Pain Group classification parameters (Logistic Regression):",
              grid_search_pain_clf_lr.best_params_)
        grid_search_success = True
    except ValueError as e:
        print(
            f"Error during GridSearchCV for Pain Group (Logistic Regression): {e}. Trying to fit base model directly.")
        try:
            best_pain_classification_pipeline_lr = pain_classification_pipeline_base_lr.fit(X_pain_clf_train,
                                                                                            y_pain_clf_train_encoded)
            grid_search_success = True
        except Exception as e_fit:
            print(f"Error fitting base Logistic Regression model directly after GridSearchCV failed: {e_fit}")
            exit()

if not best_pain_classification_pipeline_lr or not grid_search_success:
    print("Error: Pain Group classification model (Logistic Regression) could not be trained. Exiting.")
    exit()

# --- Predictions on Training Set (Pain Group - Logistic Regression) ---
if len(X_pain_clf_train) > 0 and best_pain_classification_pipeline_lr:
    print("\n\n--- PREDICTIONS ON TRAINING SET (Pain Group - Logistic Regression) ---")
    y_pred_pain_clf_on_train_encoded = best_pain_classification_pipeline_lr.predict(X_pain_clf_train)
    y_train_labels_actual = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_train_encoded)
    y_pred_train_labels = label_encoder_y_pain_clf.inverse_transform(y_pred_pain_clf_on_train_encoded)

    train_predictions_df = pd.DataFrame({
        'Patient_Index_in_Train_X': X_pain_clf_train.index,
        'Real Pain Group': y_train_labels_actual,
        'Predicted Pain Group (LR)': y_pred_train_labels
    })
    print("First 10 predictions on Training Set (LR):")
    print(train_predictions_df.head(10))  # Display more if needed, e.g. head(300)

    accuracy_train_lr = accuracy_score(y_train_labels_actual, y_pred_train_labels)
    balanced_accuracy_train_lr = balanced_accuracy_score(y_train_labels_actual, y_pred_train_labels)
    print(f"\nAccuracy on Training Set (LR): {accuracy_train_lr * 100:.2f}%")
    print(f"Balanced Accuracy on Training Set (LR): {balanced_accuracy_train_lr * 100:.2f}%")

else:
    print("Warning: Training set is empty or LR model not trained. Cannot make predictions on training data.")

# --- Predictions on Test Set (Pain Group - Logistic Regression) ---
if len(X_pain_clf_test) > 0 and best_pain_classification_pipeline_lr:
    print("\n\n--- PREDICTIONS ON TEST SET (Pain Group - Logistic Regression) ---")
    y_pred_pain_clf_on_test_encoded = best_pain_classification_pipeline_lr.predict(X_pain_clf_test)

    y_test_labels_actual = label_encoder_y_pain_clf.inverse_transform(y_pain_clf_test_encoded)
    y_pred_test_labels = label_encoder_y_pain_clf.inverse_transform(y_pred_pain_clf_on_test_encoded)

    test_predictions_df = pd.DataFrame({
        'Patient_Index_in_Test_X': X_pain_clf_test.index,
        'Real Pain Group': y_test_labels_actual,
        'Predicted Pain Group (LR)': y_pred_test_labels
    })
    print("First 10 predictions on Test Set (LR):")
    print(test_predictions_df.head(10))  # Display more if needed, e.g. head(100)

    accuracy_pain_clf_on_test_lr = accuracy_score(y_test_labels_actual, y_pred_test_labels)
    balanced_accuracy_test_lr = balanced_accuracy_score(y_test_labels_actual, y_pred_test_labels)

    print("\n--- Pain Group Classification Results (Test Set - Logistic Regression) ---")
    print(f"Test Accuracy (LR): {accuracy_pain_clf_on_test_lr * 100:.2f}%")
    print(f"Test Balanced Accuracy (LR): {balanced_accuracy_test_lr * 100:.2f}%")

    print("Classification Report (Pain Group - LR):")
    report_target_names_ordered = [label for label in pain_labels if label in label_encoder_y_pain_clf.classes_]
    print(classification_report(y_test_labels_actual, y_pred_test_labels,
                                labels=report_target_names_ordered,
                                target_names=report_target_names_ordered,
                                zero_division=0))
else:
    print("Warning: Test set is empty or LR model not trained. Cannot evaluate Pain Group classifier.")

# Plot feature importance for Logistic Regression classifier
if best_pain_classification_pipeline_lr:
    try:
        classifier_model_lr = best_pain_classification_pipeline_lr.named_steps['classifier']
        preprocessor_fitted_lr = best_pain_classification_pipeline_lr.named_steps['preprocessor']

        if hasattr(classifier_model_lr, 'coef_'):
            try:
                feature_names_transformed_lr = preprocessor_fitted_lr.get_feature_names_out()
            except AttributeError:  # Fallback for older scikit-learn
                feature_names_transformed_lr = []
                if 'num' in preprocessor_fitted_lr.named_transformers_:
                    feature_names_transformed_lr.extend(numerical_cols_pain_clf)
                if 'cat' in preprocessor_fitted_lr.named_transformers_:
                    cat_pipeline = preprocessor_fitted_lr.named_transformers_['cat']
                    if hasattr(cat_pipeline.named_steps['onehot'], 'get_feature_names_out'):
                        feature_names_transformed_lr.extend(
                            cat_pipeline.named_steps['onehot'].get_feature_names_out(categorical_cols_pain_clf))
                    else:
                        feature_names_transformed_lr.extend(
                            [f"cat_col_{i}" for i in range(len(categorical_cols_pain_clf))])
                if preprocessor_fitted_lr.remainder == 'passthrough':
                    remainder_cols = [X_pain_clf.columns[i] for i in preprocessor_fitted_lr._remainder[2]]
                    feature_names_transformed_lr.extend(remainder_cols)

            coefficients = classifier_model_lr.coef_

            if coefficients.shape[1] == len(feature_names_transformed_lr):
                if coefficients.shape[0] == 1:  # Binary or OvR where coef_ is (1, n_features)
                    importances_lr = np.abs(coefficients[0])
                else:  # Multiclass OvR, coef_ is (n_classes, n_features)
                    # Use mean of absolute coefficient values across classes as importance
                    importances_lr = np.mean(np.abs(coefficients), axis=0)

                feature_importance_df_lr = pd.DataFrame(
                    {'feature': feature_names_transformed_lr, 'importance_abs_coef': importances_lr})
                feature_importance_df_lr = feature_importance_df_lr.sort_values(by='importance_abs_coef',
                                                                                ascending=False)

                plt.figure(figsize=(12, 10))
                plt.barh(feature_importance_df_lr['feature'][:15],
                         feature_importance_df_lr['importance_abs_coef'][:15])
                plt.gca().invert_yaxis()
                plt.xlabel("Mean Absolute Coefficient Value (Importance)")
                plt.ylabel("Feature (Logistic Regression)")
                plt.title("Top 15 Important Features (Logistic Regression)")
                plt.tight_layout()
                plt.show()
            else:
                print(
                    f"Warning: Mismatch between number of transformed feature names ({len(feature_names_transformed_lr)}) "
                    f"and coefficients ({coefficients.shape[1]}) for Logistic Regression. Skipping feature importance plot.")
                print("Transformed feature names attempt:", feature_names_transformed_lr)
                print("Coefficients array shape:", coefficients.shape)
        else:
            print("Logistic Regression model does not have coef_ attribute (e.g., if fitting failed).")
    except Exception as e:
        print(f"Error plotting feature importance for Logistic Regression classifier: {e}")

# --- Step 2: Predict on New Patient Data (Pain Group Only - Logistic Regression) ---
print("\n\n--- STEP 2: PREDICT PAIN GROUP ON NEW/TEST PATIENT DATA (Logistic Regression) ---")

patients_to_predict_on = [
    {
        'Age': 50, 'Gender': 'M', 'Q28. JointP': 'Y', 'Timepoint': 'T0',
        'CD3%': 65, 'CD3Total': 1300, 'CD8%': 25, 'CD8-Suppr': 450, 'CD4%': 40, 'CD4-Helper': 800,
        'CD19 Bcell': 180, 'CD19%': 12, 'H/S-ratio': 1.8, 'CD57+NK cells': 90, 'HgB': 14.5,
        'Platelets': 270, 'Neutrophils': 4.2, 'Lymphocytes': 2.1, 'WCC': 6.5, 'CRP': 3, 'Iron': 19,
        'Transf': 2.1, '%Trans sat': 29, 'Ferritin': 105, 'CK': 110, 'FT4': 13.5, 'TSH1': 1.6,
        'Patient_ID_Ref': 'New_Patient_1_Pain_Example_Rating_6'
    },
    {
        'Age': 36, 'Gender': 'F', 'Q28. JointP': 'Y', 'Timepoint': 'T2',
        'CD3%': 72, 'CD3Total': 1667, 'CD8%': 23, 'CD8-Suppr': 543, 'CD4%': 50, 'CD4-Helper': 1169,
        'CD19 Bcell': 239, 'CD19%': 11, 'H/S-ratio': 2.15, 'CD57+NK cells': np.nan, 'HgB': 14.2,
        'Platelets': 321, 'Neutrophils': 2.87, 'Lymphocytes': 2.24, 'WCC': 5.76, 'CRP': 1, 'Iron': 16.8,
        'Transf': 2.75, '%Trans sat': 23.3, 'Ferritin': 33, 'CK': 48, 'FT4': 13.3, 'TSH1': 1.88,
        'Patient_ID_Ref': 'New_Patient_2_Pain_Example_Rating_9'
    }
]

if 'Number' in df_original.columns:
    available_patients_for_pred = df_pain_clf_input[~df_pain_clf_input['Number'].duplicated(keep='first')]
    if not available_patients_for_pred.empty:
        num_to_sample = min(2, len(available_patients_for_pred))
        sample_indices_from_X = np.random.choice(X_pain_clf.index, size=num_to_sample, replace=False)
        patients_from_csv_for_pred = df_pain_clf_input.loc[sample_indices_from_X]

        print(
            f"\nWill also test Pain Group prediction with {num_to_sample} patients from CSV (Numbers: {patients_from_csv_for_pred['Number'].tolist()})")

        for idx, patient_row_csv in patients_from_csv_for_pred.iterrows():
            patient_dict_csv = patient_row_csv.drop(labels=['Pain_Group'], errors='ignore').to_dict()
            patient_dict_csv[
                'Patient_ID_Ref'] = f"CSV_Patient_{patient_row_csv.get('Number', 'Unknown')}_ActualPainRating_{patient_row_csv.get('Pain rating', 'Unknown')}"
            patients_to_predict_on.append(patient_dict_csv)

new_patients_df_full = pd.DataFrame(patients_to_predict_on)
patient_ids_for_output = new_patients_df_full['Patient_ID_Ref'].tolist()

new_patients_input_df_pain_clf = pd.DataFrame(columns=X_pain_clf.columns)
for col in X_pain_clf.columns:
    if col in new_patients_df_full.columns:
        new_patients_input_df_pain_clf[col] = new_patients_df_full[col]
    else:
        new_patients_input_df_pain_clf[col] = np.nan
        if col in numerical_cols_pain_clf:
            new_patients_input_df_pain_clf[col] = pd.to_numeric(new_patients_input_df_pain_clf[col], errors='coerce')

if best_pain_classification_pipeline_lr:
    for col in new_patients_input_df_pain_clf.columns:
        if col in X_pain_clf_train.columns:
            new_patients_input_df_pain_clf[col] = new_patients_input_df_pain_clf[col].astype(
                X_pain_clf_train[col].dtype, errors='ignore')

    pred_pain_group_encoded = best_pain_classification_pipeline_lr.predict(new_patients_input_df_pain_clf)
    pred_pain_group_labels = label_encoder_y_pain_clf.inverse_transform(pred_pain_group_encoded)

    if hasattr(best_pain_classification_pipeline_lr, "predict_proba"):
        pred_pain_group_proba = best_pain_classification_pipeline_lr.predict_proba(new_patients_input_df_pain_clf)
    else:
        pred_pain_group_proba = None  # Some models might not have predict_proba or it might be configured off

    for i, patient_id_ref in enumerate(patient_ids_for_output):
        print(f"\n--- Predictions for Patient: {patient_id_ref} (Using Logistic Regression) ---")
        current_pred_pain_label = pred_pain_group_labels[i]

        if pred_pain_group_proba is not None:
            current_pain_probas = pred_pain_group_proba[i]
            class_names_proba = label_encoder_y_pain_clf.classes_
            proba_str_pain = ", ".join([f"{class_names_proba[j]}={prob * 100:.1f}%"
                                        for j, prob in enumerate(current_pain_probas)])
            print(f"  Predicted Pain Group: {current_pred_pain_label} (Probabilities: {proba_str_pain})")
        else:
            print(f"  Predicted Pain Group: {current_pred_pain_label} (Probabilities not available)")
else:
    print(
        "Pain Group classification model (Logistic Regression) not trained. Cannot predict Pain Group for new patients.")

print("\n--- End of Script (Logistic Regression Version) ---")