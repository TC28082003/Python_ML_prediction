import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # New import
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Suppress specific warnings for cleaner output if desired
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


def evaluate_model_on_full_dataset(
        model,
        preprocessor,
        original_X_for_eval,
        original_y_for_eval,
        top_5_processed_feature_names,
        all_processed_feature_names,  # Names of all columns after preprocessor.transform
        dataset_name="Dataset",      # New argument for clarity
        show_all_predictions=True
):
    """
    Evaluates the trained model on the provided dataset and shows example predictions.
    Args:
        model: The trained prediction model (expecting top 5 processed features).
        preprocessor: The fitted preprocessor.
        original_X_for_eval: The original feature DataFrame for evaluation.
        original_y_for_eval: The original target Series for evaluation.
        top_5_processed_feature_names: List of the names of the top 5 processed features.
        all_processed_feature_names: List of all feature names after preprocessing.
        dataset_name: Name of the dataset being evaluated (e.g., "Training", "Test").
        show_all_predictions: If True, shows actual vs predicted for all patients in the set.
    """
    print(f"\n--- Evaluating Model on {dataset_name} ---")

    if original_X_for_eval.empty or original_y_for_eval.empty:
        print(f"No data to evaluate on for {dataset_name}.")
        return

    # 1. Preprocess the original_X_for_eval dataset
    try:
        # The preprocessor was FIT on training data, here we ONLY TRANSFORM
        X_eval_processed_array = preprocessor.transform(original_X_for_eval)
        X_eval_processed_df = pd.DataFrame(
            X_eval_processed_array,
            columns=all_processed_feature_names,
            index=original_X_for_eval.index
        )
    except Exception as e:
        print(f"Error during preprocessing of {dataset_name}: {e}")
        return

    # 2. Select only the top 5 processed features
    missing_cols = [col for col in top_5_processed_feature_names if col not in X_eval_processed_df.columns]
    if missing_cols:
        print(f"Error: The following top 5 features are missing from the processed {dataset_name} data: {missing_cols}")
        print(f"Available columns: {X_eval_processed_df.columns.tolist()}")
        return

    X_eval_top5_features = X_eval_processed_df[top_5_processed_feature_names]

    # 3. Make predictions
    y_pred_eval = model.predict(X_eval_top5_features)

    # 4. Calculate and print metrics
    mse = mean_squared_error(original_y_for_eval, y_pred_eval)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_y_for_eval, y_pred_eval)
    r2 = r2_score(original_y_for_eval, y_pred_eval)

    errors = np.abs(original_y_for_eval.values - y_pred_eval)
    max_err_idx = np.argmax(errors)
    print(f"  Patient with Maximum Absolute Error (Index: {original_y_for_eval.index[max_err_idx]}):")
    print(
        f"    Actual: {original_y_for_eval.iloc[max_err_idx]:.2f}, Predicted: {y_pred_eval[max_err_idx]:.2f}, Error: {errors[max_err_idx]:.2f}")

    print(f"Performance on {dataset_name} using top 5 features model:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R-squared (R²): {r2:.4f}")
    if dataset_name.lower() == "training":
        print("  Note: If this evaluation is on the training data, R² might be optimistic compared to test data.")

    # 5. Show predictions for all patients if requested
    if show_all_predictions:
        print(f"\n--- All Predictions (Actual vs. Predicted) from the {dataset_name} ---")
        if len(original_y_for_eval) > 0:
            results_df = pd.DataFrame({
                'Actual Pain Rating': original_y_for_eval.values,
                'Predicted Pain Rating': y_pred_eval,
                'Error': abs(original_y_for_eval.values - y_pred_eval), # Corrected column name to 'Error'
            }, index=original_y_for_eval.index)

            print(f"{'Original Index':<15} {'Actual Pain':<15} {'Predicted Pain':<18} {'Error':<15}") # Adjusted spacing
            print("-" * 63) # Adjusted line length
            results_df = results_df.sort_values(by='Error', ascending=False)

            for idx, row in results_df.iterrows():
                print(f"{idx:<15} {row['Actual Pain Rating']:<15.2f} {row['Predicted Pain Rating']:<18.2f} {row['Error']:<15.2f}")
        else:
            print("Not enough data points to show examples.")
    else:
        print(f"\n(Individual predictions for all patients in {dataset_name} not shown based on settings)")


def solve():
    # --- 1. Load Data ---
    try:
        df_loaded = pd.read_csv("Pain_python.csv") # Corrected filename back to Pain_python.csv
    except FileNotFoundError:
        print("Error: Pain_python.csv not found. Make sure it's in the same directory.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    df = df_loaded.copy()

    # --- 2. Basic Data Cleaning & Preparation ---
    target = 'Pain rating'

    if df[target].isnull().any():
        print(f"Dropping {df[target].isnull().sum()} rows with missing '{target}' values.")
        df.dropna(subset=[target], inplace=True)

    df[target] = pd.to_numeric(df[target], errors='coerce')
    df.dropna(subset=[target], inplace=True) # Drop rows where conversion to numeric failed

    if df.empty:
        print("Error: No data left after initial cleaning. Exiting.")
        return

    # Prepare X and y
    X = df.drop(columns=[target])
    y = df[target]

    if 'Number' in X.columns:
        X = X.drop(columns=['Number'])

    if X.empty:
        print("Error: No features left after dropping 'Number' and 'Pain rating'. Exiting.")
        return

    # --- 2.5. Train-Test Split ---
    # Splitting data BEFORE any preprocessing fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    if X_train.empty:
        print("Error: Training set is empty after split. Exiting.")
        return

    # --- 3. Preprocessing ---
    # Identify column types from the TRAINING data
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Special handling for 'Q28. JointP' if it exists and is categorical
    if 'Q28. JointP' in X_train.columns and 'Q28. JointP' in categorical_cols:
        # Apply mapping to both train and test sets to maintain consistency
        X_train['Q28. JointP'] = X_train['Q28. JointP'].map({'Y': 1, 'N': 0, np.nan: np.nan}) # Handle potential NaNs
        X_test['Q28. JointP'] = X_test['Q28. JointP'].map({'Y': 1, 'N': 0, np.nan: np.nan})

        # Update column lists if 'Q28. JointP' becomes numeric
        if pd.api.types.is_numeric_dtype(X_train['Q28. JointP']):
            if 'Q28. JointP' not in numerical_cols:
                numerical_cols.append('Q28. JointP')
            if 'Q28. JointP' in categorical_cols:
                categorical_cols.remove('Q28. JointP')


    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Ensure cols exist in X_train before adding to transformers
    final_categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    final_numerical_cols = [col for col in numerical_cols if col in X_train.columns]

    transformers_list = []
    if final_numerical_cols:
        transformers_list.append(('num', numerical_transformer, final_numerical_cols))
    if final_categorical_cols:
        transformers_list.append(('cat', categorical_transformer, final_categorical_cols))

    if not transformers_list:
        print("Error: No features to process after identifying types. Check column types in training data.")
        return

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')

    # FIT preprocessor on TRAINING data ONLY
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(X_train)

    # Transform training and test data
    X_train_processed_array = preprocessor.transform(X_train)
    X_test_processed_array = preprocessor.transform(X_test) # Used later for evaluation


    # Reconstruct feature names (based on fitting on X_train)
    all_processed_feature_names = []
    try:
        num_feat_names_out = []
        cat_feat_names_out = []

        for name, trans_pipeline, cols_in_transformer in preprocessor.transformers_:
            if trans_pipeline == 'drop':
                continue
            if name == 'num' and final_numerical_cols: # Ensure there were numerical cols
                # For numerical transformers, try get_feature_names_out from steps, else use original cols
                current_num_names = list(cols_in_transformer) # Default to original names
                if hasattr(trans_pipeline, 'named_steps'): # Pipeline of transformers
                    for step_name, step_trans in trans_pipeline.named_steps.items():
                        if hasattr(step_trans, 'get_feature_names_out'):
                            try:
                                current_num_names = list(step_trans.get_feature_names_out(cols_in_transformer))
                                break # Take names from first step that provides them
                            except TypeError:
                                current_num_names = list(step_trans.get_feature_names_out())
                                break
                elif hasattr(trans_pipeline, 'get_feature_names_out'): # Single transformer
                     try:
                        current_num_names = list(trans_pipeline.get_feature_names_out(cols_in_transformer))
                     except TypeError:
                        current_num_names = list(trans_pipeline.get_feature_names_out())
                num_feat_names_out.extend(current_num_names)

            elif name == 'cat' and final_categorical_cols: # Ensure there were categorical cols
                # For categorical (OHE), rely on its get_feature_names_out
                onehot_step = trans_pipeline.named_steps.get('onehot')
                if onehot_step and hasattr(onehot_step, 'get_feature_names_out'):
                    cat_feat_names_out.extend(list(onehot_step.get_feature_names_out(cols_in_transformer)))
                else: # Fallback if OHE names can't be retrieved
                    pass


        all_processed_feature_names.extend(num_feat_names_out)
        all_processed_feature_names.extend(cat_feat_names_out)

        # Handling remainder columns more robustly
        if preprocessor.remainder == 'passthrough':
            # Get indices of columns NOT processed by 'num' or 'cat'
            processed_feature_indices = []
            for _, _, cols_indices_or_names in preprocessor.transformers_:
                if isinstance(cols_indices_or_names[0], str): # Column names
                    for col_name in cols_indices_or_names:
                        if col_name in X_train.columns:
                             processed_feature_indices.append(X_train.columns.get_loc(col_name))
                else: # Column indices
                    processed_feature_indices.extend(cols_indices_or_names)

            # Get names of remainder columns based on their original positions in X_train
            remainder_col_indices = [i for i in range(len(X_train.columns)) if i not in processed_feature_indices]
            remainder_col_names = [X_train.columns[i] for i in remainder_col_indices]
            all_processed_feature_names.extend(remainder_col_names)


        seen = set()
        all_processed_feature_names = [x for x in all_processed_feature_names if not (x in seen or seen.add(x))]

        if not cat_feat_names_out and final_categorical_cols:
             print("Warning: Could not automatically determine OHE feature names. Feature name reconstruction might be incomplete.")


    except Exception as e:
        print(f"Error getting feature names after preprocessing: {e}. Using generic names.")
        all_processed_feature_names = [f"feature_{i}" for i in range(X_train_processed_array.shape[1])]

    if X_train_processed_array.shape[1] != len(all_processed_feature_names):
        print(f"CRITICAL Error: Mismatch in processed array columns ({X_train_processed_array.shape[1]}) "
              f"and reconstructed feature names ({len(all_processed_feature_names)}).")
        print("Reconstructed names attempt:", all_processed_feature_names)
        print("Available X_train columns:", X_train.columns.tolist())
        print("Transformers:", preprocessor.transformers_)
        print("Using generic feature names as a last resort for DataFrame construction.")
        all_processed_feature_names = [f"feature_{i}" for i in range(X_train_processed_array.shape[1])]


    X_train_processed_df = pd.DataFrame(
        X_train_processed_array,
        columns=all_processed_feature_names,
        index=X_train.index # Preserve original index from X_train
    )

    # --- 4. Model for Feature Importance (trained on TRAINING data) ---
    print("\nTraining feature importance model on training data...")
    model_fi = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
    model_fi.fit(X_train_processed_df, y_train)
    print(f"OOB Score of RandomForest for feature importance (all features, on training data): {model_fi.oob_score_:.4f}")

    importances = model_fi.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X_train_processed_df.columns, 'importance': importances})
    top_5_features_df = feature_importance_df.sort_values(by='importance', ascending=False).head(10) # Changed to top 10
    print("\n--- Top 10 Most Important Features (from training data after preprocessing) ---") # Changed to top 10
    print(top_5_features_df)
    top_5_processed_feature_names = top_5_features_df['feature'].tolist() # Still named top_5 for consistency

    # --- 5. Model for Prediction (trained on TRAINING data with top features) ---
    X_top5_for_training = X_train_processed_df[top_5_processed_feature_names]
    print("\nTraining prediction model on training data (top features)...")
    model_pred = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
    model_pred.fit(X_top5_for_training, y_train)
    print(f"OOB Score of RandomForest for prediction (top features, on training data): {model_pred.oob_score_:.4f}")

    # --- Evaluate on TRAINING Set ---
    evaluate_model_on_full_dataset(
        model=model_pred,
        preprocessor=preprocessor, # Already fitted on X_train
        original_X_for_eval=X_train.copy(),
        original_y_for_eval=y_train.copy(),
        top_5_processed_feature_names=top_5_processed_feature_names,
        all_processed_feature_names=all_processed_feature_names,
        dataset_name="Training Set",
        show_all_predictions=True # Typically False for training set to save space, or True if desired
    )


if __name__ == '__main__':
    solve()