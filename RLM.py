import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Read and prepare data (Same as before) ---
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
    else:
        print(f"Warning: Column '{col}' is not in df.columns when trying to convert to numeric.")

# --- Pair T0 and T2 data and calculate the change ---
if 'Number' not in df.columns:
    print("Error: No 'Number' column to pair patients.")
    exit()

# Variables of interest for calculating delta (INCLUDING ALL 5 BIOMARKERS + PAIN RATING)
vars_for_delta = ['Pain_Rating', 'Transferrin_Actual', 'CD4_Percent',
                  'Platelets', 'Neutrophils', 'Trans_Sat_Percent']
df_pivoted_list = []

for var in vars_for_delta:
    if var in df.columns:
        df_pivot_var = df.pivot_table(index='Number', columns='Timepoint', values=var)
        df_pivot_var.columns = [f'{var}_T0', f'{var}_T2']
        df_pivoted_list.append(df_pivot_var)
    else:
        print(f"Warning: Column '{var}' not found in data when pivoting.")

if not df_pivoted_list:
    print("Error: No valid variables to pivot and calculate delta.")
    exit()

df_paired = df_pivoted_list[0]
for i in range(1, len(df_pivoted_list)):
    df_paired = pd.merge(df_paired, df_pivoted_list[i], on='Number', how='inner')
df_paired.reset_index(inplace=True)

for var in vars_for_delta:
    col_t0 = f'{var}_T0'
    col_t2 = f'{var}_T2'
    if col_t0 in df_paired.columns and col_t2 in df_paired.columns:
        df_paired[f'{var}_Change'] = df_paired[col_t2] - df_paired[col_t0]
    else:
         print(f"Warning: Cannot calculate {var}_Change due to missing T0 or T2 column after pivoting.")

# PREPARE DATA FOR RLM (ALL 5 BIOMARKERS)
# df_paired already contains _Change for all 5 biomarkers for plotting and modeling
cols_for_rlm_model = ['Pain_Rating_Change', 'Transferrin_Actual_Change', 'CD4_Percent_Change', 'Platelets_Change', 'Neutrophils_Change', 'Trans_Sat_Percent_Change' ]
df_rlm_model_data = df_paired.dropna(subset=cols_for_rlm_model).copy()

if df_rlm_model_data.empty:
    print("Not enough data (for the specified _Change columns) after removing NaN to run RLM.")
    exit()

print(f"\nNumber of patients used for RLM (all 5 biomarkers) after processing: {len(df_rlm_model_data)}")
print("\nFirst 5 rows of data prepared for RLM (all 5 biomarkers):")
print(df_rlm_model_data[['Number'] + cols_for_rlm_model].head())


y_rlm = df_rlm_model_data['Pain_Rating_Change']
X_vars_rlm = df_rlm_model_data[['Transferrin_Actual_Change', 'CD4_Percent_Change', 'Platelets_Change', 'Neutrophils_Change', 'Trans_Sat_Percent_Change' ]]

# Model 1: No constant
print("\n--- RLM Results (NO CONSTANT) predicting Pain_Rating_Change ---")
try:
    rlm_model_no_const = sm.RLM(y_rlm, X_vars_rlm, M=sm.robust.norms.HuberT())
    rlm_results_no_const = rlm_model_no_const.fit()
    print(rlm_results_no_const.summary())
    # (Detailed interpretation can be added back if desired)
except Exception as e:
    print(f"Error running RLM without constant: {e}")

# Model 2: With constant
X_vars_rlm_with_const = sm.add_constant(X_vars_rlm)
print("\n\n--- RLM Results (WITH CONSTANT) predicting Pain_Rating_Change ---")
try:
    rlm_model_with_const = sm.RLM(y_rlm, X_vars_rlm_with_const, M=sm.robust.norms.HuberT())
    rlm_results_with_const = rlm_model_with_const.fit()
    print(rlm_results_with_const.summary())
except Exception as e:
    print(f"Error running RLM with constant: {e}")

biomarkers_to_plot_change = {
    'Transferrin': 'Transferrin_Actual_Change',
    'CD4%': 'CD4_Percent_Change',
    'Platelets': 'Platelets_Change',
    'Neutrophils': 'Neutrophils_Change',
    '% Transferrin Saturation': 'Trans_Sat_Percent_Change' # Translated label
}

for bm_label, bm_change_col in biomarkers_to_plot_change.items():
    if bm_change_col in df_paired.columns and 'Pain_Rating_Change' in df_paired.columns:
        # Create a temporary DataFrame with only the 2 columns to plot, remove NaNs for that pair
        plot_data = df_paired[['Pain_Rating_Change', bm_change_col]].dropna()
        if not plot_data.empty:
            plt.figure(figsize=(8, 6))
            sns.regplot(x=bm_change_col, y='Pain_Rating_Change', data=plot_data,
                        line_kws={"color": "red"}, scatter_kws={"alpha":0.5}, robust=True)
            plt.title(f'Pain Rating Change vs. {bm_label} Change (with RLM line)')
            plt.xlabel(f'{bm_label} Change (T2 - T0)')
            plt.ylabel('Pain Rating Change (T2 - T0)')
            plt.grid(True)
            plt.show()
        else:
            print(f"Not enough data to plot for {bm_label}_Change vs Pain_Rating_Change after removing NaN.")
    else:
        print(f"Warning: Column {bm_change_col} or Pain_Rating_Change not in df_paired for plotting.")

print("\n--- Analysis and visualization complete ---")