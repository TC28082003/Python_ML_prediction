import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # Using sklearn's Pipeline as SMOTE is not applied in this step
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, \
    roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Read and prepare basic data (Keep as is) ---
file_path = 'Pain_python.csv'
df_original = pd.read_csv(file_path)
df = df_original.copy()
df.rename(columns={
    'Pain rating': 'Pain_Rating',
    'Q28. JointP': 'Joint_Pain',
    'CD4%': 'CD4_Percent',
    '%Trans sat': 'Trans_Sat_Percent',
    'Transf': 'Transferrin_Actual'
}, inplace=True)

# --- 2. Prepare data for 2-CLASS PAIN CLASSIFICATION at T0 ---
print("\n--- CLASSIFY PATIENTS INTO 2 PAIN GROUPS (Not High vs. High) AT T0 ---")

df_t0_clf2 = df[df['Timepoint'] == 'T0'].copy()
df_t0_clf2['Pain_Rating'] = pd.to_numeric(df_t0_clf2['Pain_Rating'], errors='coerce')
df_t0_clf2.dropna(subset=['Pain_Rating'], inplace=True)
df_t0_clf2['Pain_Rating'] = df_t0_clf2['Pain_Rating'].astype(int)

# Define 2 pain groups
# Group 0: Not High Pain (1-6)
# Group 1: High Pain (7-10)
pain_bins_2class = [0, 6, 10]  # Thresholds: (0,6], (6,10]
pain_labels_2class = ['Not High Pain (1-6)', 'High Pain (7-10)']

df_t0_clf2['Pain_Category_2class_T0'] = pd.cut(df_t0_clf2['Pain_Rating'],
                                               bins=pain_bins_2class,
                                               labels=pain_labels_2class,
                                               right=True,
                                               include_lowest=True)
df_t0_clf2.dropna(subset=['Pain_Category_2class_T0'], inplace=True)

print("\nDistribution of 2 pain groups at T0:")
print(df_t0_clf2['Pain_Category_2class_T0'].value_counts().sort_index())

target_clf2_t0 = 'Pain_Category_2class_T0'

potential_features_clf2_t0 = [  # Can reuse the optimized feature set
    'Transferrin_Actual',  # Renamed from 'Transf'
    'CD4_Percent',  # Renamed from 'CD4%'
    'Platelets',

    # Add other potentially important features from your previous analysis:
    'Joint_Pain',  # Categorical, very important
    'Iron',  # Often associated with the transferrin group, very important
    'Ferritin',  # Similar to Iron
]
feature_columns_clf2_t0 = [col for col in potential_features_clf2_t0 if col in df_t0_clf2.columns]
if 'Pain_Rating' in feature_columns_clf2_t0:
    feature_columns_clf2_t0.remove('Pain_Rating')

X_clf2_t0 = df_t0_clf2[feature_columns_clf2_t0].copy()
y_clf2_t0_labels = df_t0_clf2[target_clf2_t0].copy()

if X_clf2_t0.empty or y_clf2_t0_labels.empty:
    print("Not enough data after creating 2 pain groups at T0.")
    exit()

label_encoder_2class_t0 = LabelEncoder()
y_clf2_t0_encoded = label_encoder_2class_t0.fit_transform(y_clf2_t0_labels)
print("\nEncoded target classes (2 classes):")
for i, class_name in enumerate(label_encoder_2class_t0.classes_):
    print(f"  {class_name} -> {i}")  # Will be 0 and 1

numerical_cols_clf2_t0 = X_clf2_t0.select_dtypes(include=np.number).columns.tolist()
categorical_cols_clf2_t0 = X_clf2_t0.select_dtypes(include='object').columns.tolist()

# --- 3. Create Preprocessor Pipeline (Keep as is) ---
numerical_transformer_clf2_t0 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer_clf2_t0 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor_clf2_t0 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_clf2_t0, numerical_cols_clf2_t0),
        ('cat', categorical_transformer_clf2_t0, categorical_cols_clf2_t0)
    ],
    remainder='drop'
)

# --- 4. Split Train/Test data (Stratified) ---
X_train_clf2_t0, X_test_clf2_t0, y_train_clf2_t0, y_test_clf2_t0 = train_test_split(
    X_clf2_t0, y_clf2_t0_encoded, test_size=0.25, random_state=42, stratify=y_clf2_t0_encoded
)
print(f"\nTraining set size (2 classes) T0: X={X_train_clf2_t0.shape}, y={y_train_clf2_t0.shape}")
print(f"Test set size (2 classes) T0: X={X_test_clf2_t0.shape}, y={y_test_clf2_t0.shape}")
print("Class distribution (2 classes) in training set:", np.bincount(y_train_clf2_t0))
print("Class distribution (2 classes) in test set:", np.bincount(y_test_clf2_t0))

# --- 5. Build, Train, and Evaluate Classification Models (2 classes) ---
# Can remove class_weight='balanced' if the 2 classes are already fairly balanced, or keeping it is also fine
classifiers_2class_t0 = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
    "SVC": SVC(random_state=42, probability=True),  # probability=True to calculate ROC AUC
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_jobs=-1)
}

# Keep param_grids as is or adjust if needed
param_grids_2class_t0 = {
    "Logistic Regression": {'classifier__C': [0.01, 0.1, 1, 10]},
    "SVC": {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']},
    "Random Forest": {'classifier__n_estimators': [100, 150, 200], 'classifier__max_depth': [None, 10, 20],
                      'classifier__min_samples_split': [2, 5]},
    "Gradient Boosting": {'classifier__n_estimators': [100, 150], 'classifier__learning_rate': [0.05, 0.1],
                          'classifier__max_depth': [3, 4]},
    "KNN": {'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
}

trained_classifiers_2class_t0 = {}
results_clf2_t0 = {}

# Determine the number of folds for StratifiedKFold
n_splits_cv_2class = 5  # Can be increased to 5 as data might be more balanced
min_class_count_train_2class = np.bincount(y_train_clf2_t0).min() if len(y_train_clf2_t0) > 0 else 0

if min_class_count_train_2class < n_splits_cv_2class and min_class_count_train_2class > 1:
    n_splits_cv_2class = min_class_count_train_2class
    print(
        f"Warning: The number of samples in the smallest class in the training set is {min_class_count_train_2class}, reducing n_splits for CV to {n_splits_cv_2class}")
elif min_class_count_train_2class <= 1:  # Should not happen with 2 classes if original data is sufficient
    n_splits_cv_2class = 2
    print(
        f"Warning: Smallest class in training set has <=1 samples. CV may not work well. Setting n_splits={n_splits_cv_2class}")

for clf_name, clf_instance in classifiers_2class_t0.items():
    print(f"\n--- Training and Evaluation: {clf_name} (2-class Classification T0) ---")
    # Using sklearn's Pipeline (SMOTE is not applied in this step)
    pipeline_clf2_t0 = Pipeline(steps=[
        ('preprocessor', preprocessor_clf2_t0),
        ('classifier', clf_instance)
    ])

    best_clf_pipeline_2class_t0 = None  # This will store the best pipeline for the current classifier
    if clf_name in param_grids_2class_t0 and min_class_count_train_2class >= n_splits_cv_2class and len(
            np.unique(y_train_clf2_t0)) > 1:
        print("Performing GridSearchCV...")
        cv_stratified_2class = StratifiedKFold(n_splits=n_splits_cv_2class, shuffle=True, random_state=42)
        # Use 'roc_auc' for 2-class problem, or 'f1_weighted' / 'balanced_accuracy'
        grid_search_clf2_t0 = GridSearchCV(pipeline_clf2_t0, param_grids_2class_t0[clf_name], cv=cv_stratified_2class,
                                           scoring='roc_auc', n_jobs=-1, verbose=0)
        try:
            grid_search_clf2_t0.fit(X_train_clf2_t0, y_train_clf2_t0)
            best_clf_pipeline_2class_t0 = grid_search_clf2_t0.best_estimator_
            print(f"Best params for {clf_name}: {grid_search_clf2_t0.best_params_}")
        except Exception as e:
            print(f"Error in GridSearchCV for {clf_name}: {e}. Training with default parameters.")
            pipeline_clf2_t0.fit(X_train_clf2_t0, y_train_clf2_t0)
            best_clf_pipeline_2class_t0 = pipeline_clf2_t0
    else:
        if not (len(np.unique(y_train_clf2_t0)) > 1):
            print(f"Only 1 class in the training set. Cannot train classification model for {clf_name}.")
            continue  # Skip to the next classifier
        print(
            f"Not enough samples in the smallest class ({min_class_count_train_2class}) for {n_splits_cv_2class}-fold CV or no param_grid for {clf_name}. Training with default parameters.")
        pipeline_clf2_t0.fit(X_train_clf2_t0, y_train_clf2_t0)
        best_clf_pipeline_2class_t0 = pipeline_clf2_t0

    if best_clf_pipeline_2class_t0 is None:  # Should not happen if continue is used above, but as a safeguard
        print(f"Could not train or select a model for {clf_name}. Skipping evaluation.")
        continue

    trained_classifiers_2class_t0[clf_name] = best_clf_pipeline_2class_t0
    y_pred_clf2_t0 = best_clf_pipeline_2class_t0.predict(X_test_clf2_t0)

    # Ensure y_test_clf2_t0 has at least two classes for roc_auc_score if predict_proba is used
    if len(np.unique(y_test_clf2_t0)) > 1 and hasattr(best_clf_pipeline_2class_t0, "predict_proba"):
        y_pred_proba_clf2_t0 = best_clf_pipeline_2class_t0.predict_proba(X_test_clf2_t0)[:, 1]  # Probability of class 1
        roc_auc_2class_t0 = roc_auc_score(y_test_clf2_t0, y_pred_proba_clf2_t0)
    else:
        y_pred_proba_clf2_t0 = None  # Cannot calculate ROC AUC if only one class or no predict_proba
        roc_auc_2class_t0 = np.nan  # Assign NaN or handle as appropriate
        if not hasattr(best_clf_pipeline_2class_t0, "predict_proba"):
            print(f"Warning: {clf_name} does not have predict_proba method. ROC AUC cannot be calculated.")
        elif len(np.unique(y_test_clf2_t0)) <= 1:
            print(f"Warning: Only one class present in y_test_clf2_t0. ROC AUC cannot be calculated for {clf_name}.")

    acc_2class_t0 = accuracy_score(y_test_clf2_t0, y_pred_clf2_t0)
    bal_acc_2class_t0 = balanced_accuracy_score(y_test_clf2_t0, y_pred_clf2_t0)
    report_2class_t0 = classification_report(y_test_clf2_t0, y_pred_clf2_t0,
                                             target_names=label_encoder_2class_t0.classes_, zero_division=0)
    cm_2class_t0 = confusion_matrix(y_test_clf2_t0, y_pred_clf2_t0, labels=range(len(label_encoder_2class_t0.classes_)))

    results_clf2_t0[clf_name] = {'Accuracy': acc_2class_t0, 'Balanced Accuracy': bal_acc_2class_t0,
                                 'ROC AUC': roc_auc_2class_t0, 'Report': report_2class_t0, 'CM': cm_2class_t0}

    print(f"Evaluation results for {clf_name} on test set (2-class Classification T0):")
    print(f"  Accuracy: {acc_2class_t0:.4f}")
    print(f"  Balanced Accuracy: {bal_acc_2class_t0:.4f}")
    if not np.isnan(roc_auc_2class_t0):
        print(f"  ROC AUC Score: {roc_auc_2class_t0:.4f}")
    else:
        print(f"  ROC AUC Score: Not calculable")
    print("  Classification Report:\n", report_2class_t0)

    plt.figure(figsize=(5, 3.5))  # Adjust size for better fit
    sns.heatmap(cm_2class_t0, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder_2class_t0.classes_, yticklabels=label_encoder_2class_t0.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {clf_name} (2 Classes T0)')
    plt.show()

    # --- Feature Importance (for the current classifier's best pipeline) ---
    print(f"\n--- Feature Importance for {clf_name} (2-class Classification T0) ---")
    # Ensure best_clf_pipeline_2class_t0 is the fitted pipeline
    if best_clf_pipeline_2class_t0 is not None:
        fitted_preprocessor = best_clf_pipeline_2class_t0.named_steps['preprocessor']
        fitted_classifier = best_clf_pipeline_2class_t0.named_steps['classifier']

        try:
            # Get feature names after preprocessing
            feature_names = fitted_preprocessor.get_feature_names_out()

            importances = None
            importance_type = ""
            if hasattr(fitted_classifier, 'feature_importances_'):
                importances = fitted_classifier.feature_importances_
                importance_type = "Feature Importances"
            elif hasattr(fitted_classifier, 'coef_'):
                if fitted_classifier.coef_.ndim > 1:  # e.g. LogisticRegression, coef_ is (1, n_features) for binary
                    importances = np.abs(fitted_classifier.coef_[0])
                else:  # e.g. LinearSVC with coef_ as (n_features,)
                    importances = np.abs(fitted_classifier.coef_)
                importance_type = "Absolute Coefficients"

            if importances is not None:
                if len(feature_names) != len(importances):
                    print(
                        f"Warning: Mismatch in length of feature names ({len(feature_names)}) and importances ({len(importances)}) for {clf_name}.")
                    # Attempt to proceed if lengths are close, or log error

                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(
                    15)  # Display top 15

                plt.figure(figsize=(10, max(4, len(feature_importance_df) * 0.4)))  # Adjust height
                sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
                plt.title(f'{importance_type} for {clf_name} (Top {len(feature_importance_df)}) - 2 Classes T0')
                plt.xlabel(importance_type)
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.show()
            else:
                print(
                    f"Feature importance/coefficients not directly available for {clf_name} (e.g., KNN or non-linear SVC without specific method).")

        except AttributeError as e_attr:
            print(
                f"Could not retrieve feature names or importances for {clf_name} due to AttributeError: {e_attr}. This might happen with older scikit-learn versions or specific transformer types.")
        except Exception as e:
            print(f"Could not plot feature importances for {clf_name}: {e}")
    else:
        print(f"Skipping feature importance for {clf_name} as no model was trained/selected.")

# --- (Optional: Overall best model selection and its feature importance if desired) ---
# This would involve comparing models in 'results_clf2_t0' based on a metric like ROC AUC or Balanced Accuracy
# and then re-plotting feature importance for that single overall best model.

print("\n--- Completed 2-group pain classification at T0 ---")