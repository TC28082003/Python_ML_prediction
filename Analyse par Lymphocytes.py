import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kstest # For statistical comparison

# --- 1. Read and clean basic data ---
file_path = 'Pain_python.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

print("--- General data information ---")
df.info()

# Rename columns for easier use
df.rename(columns={'Q28. JointP': 'Joint_Pain', 'Pain rating': 'Pain_Rating'}, inplace=True)

# Convert columns that might be numeric but are in object format
cols_to_numeric = ['Lymphocytes', 'Age', 'Pain_Rating', 'Neutrophils', 'WCC', 'CRP', 'HgB', 'Platelets',
                   'CD3Total', 'CD8-Suppr', 'CD4-Helper', 'CD19 Bcell', 'Ferritin', 'CK', 'Iron', '%Trans sat',
                   'Transf'] # Add Transferrin if not already present

for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Separate T0 and T2 data
df_t0 = df[df['Timepoint'] == 'T0'].copy()
df_t2 = df[df['Timepoint'] == 'T2'].copy()

print(f"\nNumber of patients at T0: {len(df_t0)}")
print(f"Number of patients at T2: {len(df_t2)}")

# --- 2. Descriptive statistics for 'Lymphocytes' column for T0 and T2 ---
lymph_col = 'Lymphocytes'
if lymph_col in df_t0.columns and lymph_col in df_t2.columns:
    print(f"\n--- Descriptive statistics for {lymph_col} at T0 ---")
    print(df_t0[lymph_col].describe())
    print(f"\n--- Descriptive statistics for {lymph_col} at T2 ---")
    print(df_t2[lymph_col].describe())

    # Remove NaN in Lymphocytes column for specific analyses
    df_t0_lymph_cleaned = df_t0.dropna(subset=[lymph_col])
    df_t2_lymph_cleaned = df_t2.dropna(subset=[lymph_col])

    # --- 3. Compare distribution of Lymphocyte counts between T0 and T2 ---
    plt.figure(figsize=(12, 6))
    sns.histplot(df_t0_lymph_cleaned[lymph_col], color="skyblue", label='T0', kde=True, bins=20, stat="density")
    sns.histplot(df_t2_lymph_cleaned[lymph_col], color="red", label='T2', kde=True, bins=20, stat="density", alpha=0.7)
    plt.title(f'Distribution of {lymph_col} at T0 and T2')
    plt.xlabel(f'{lymph_col} Count')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=pd.concat([df_t0_lymph_cleaned[[lymph_col]].assign(Timepoint='T0'),
                                df_t2_lymph_cleaned[[lymph_col]].assign(Timepoint='T2')]),
                x='Timepoint', y=lymph_col)
    plt.title(f'Boxplot of {lymph_col} at T0 vs T2')
    plt.show()

    # Statistical test for difference (e.g., Mann-Whitney U test)
    if not df_t0_lymph_cleaned.empty and not df_t2_lymph_cleaned.empty:
        u_stat, p_val = mannwhitneyu(df_t0_lymph_cleaned[lymph_col], df_t2_lymph_cleaned[lymph_col], alternative='two-sided')
        print(f"\nComparing {lymph_col} between T0 and T2 (Mann-Whitney U):")
        print(f"U-statistic: {u_stat}, P-value: {p_val}")
        if p_val < 0.05:
            print(f"There is a statistically significant difference in {lymph_col} between T0 and T2.")
        else:
            print(f"There is no statistically significant difference in {lymph_col} between T0 and T2.")

    # --- Verify findings of Garg et al. ---
    # 1. Decrease in Pain Rating
    pain_col = 'Pain_Rating'
    if pain_col in df_t0.columns and pain_col in df_t2.columns:
        df_t0_pain_cleaned = df_t0.dropna(subset=[pain_col])
        df_t2_pain_cleaned = df_t2.dropna(subset=[pain_col])

        if not df_t0_pain_cleaned.empty and not df_t2_pain_cleaned.empty:
            median_pain_t0 = df_t0_pain_cleaned[pain_col].median()
            median_pain_t2 = df_t2_pain_cleaned[pain_col].median()
            print(f"\nMedian {pain_col} at T0: {median_pain_t0}")
            print(f"Median {pain_col} at T2: {median_pain_t2}")

            u_stat_pain, p_val_pain = mannwhitneyu(df_t0_pain_cleaned[pain_col], df_t2_pain_cleaned[pain_col], alternative='greater') # T0 > T2
            print(f"Comparing {pain_col} between T0 and T2 (Mann-Whitney U): U-statistic: {u_stat_pain}, P-value: {p_val_pain}")
            if p_val_pain < 0.001: # According to the paper
                print("Pain level significantly decreased from T0 to T2 (p < 0.001).")
            elif p_val_pain < 0.05:
                 print("Pain level significantly decreased from T0 to T2 (p < 0.05).")
            else:
                print("No significant decrease in pain level from T0 to T2.")

            plt.figure(figsize=(8,6))
            sns.histplot(df_t0_pain_cleaned[pain_col], color="skyblue", label='T0 Pain', kde=True, stat="density")
            sns.histplot(df_t2_pain_cleaned[pain_col], color="red", label='T2 Pain', kde=True, stat="density", alpha=0.7)
            plt.title("Distribution of Pain Rating at T0 and T2")
            plt.legend()
            plt.show()


    # 2. Changes in other biomarkers (Transferrin, CD4%, Platelets, Neutrophils)
    biomarkers_to_check = {'Transferrin': 'Transf', 'CD4%': 'CD4%', 'Platelets': 'Platelets', 'Neutrophils': 'Neutrophils'}

    for bm_name, bm_col in biomarkers_to_check.items():
        if bm_col in df_t0.columns and bm_col in df_t2.columns:
            df_t0_bm_cleaned = df_t0.dropna(subset=[bm_col])
            df_t2_bm_cleaned = df_t2.dropna(subset=[bm_col])

            if not df_t0_bm_cleaned.empty and not df_t2_bm_cleaned.empty:
                median_bm_t0 = df_t0_bm_cleaned[bm_col].median()
                median_bm_t2 = df_t2_bm_cleaned[bm_col].median()
                print(f"\n--- {bm_name} ---")
                print(f"Median at T0: {median_bm_t0}")
                print(f"Median at T2: {median_bm_t2}")

                # Mann-Whitney U test for median change
                u_stat_bm, p_val_bm_median = mannwhitneyu(df_t0_bm_cleaned[bm_col], df_t2_bm_cleaned[bm_col], alternative='two-sided')
                print(f"Comparing median of {bm_name} between T0 and T2 (Mann-Whitney U): U-statistic: {u_stat_bm}, P-value: {p_val_bm_median}")

                # Kolmogorov-Smirnov test for distribution change
                ks_stat_bm, p_val_bm_dist = kstest(df_t0_bm_cleaned[bm_col], df_t2_bm_cleaned[bm_col])
                print(f"Comparing distribution of {bm_name} between T0 and T2 (K-S test): KS-statistic: {ks_stat_bm}, P-value: {p_val_bm_dist}")

                if p_val_bm_median < 0.05 and p_val_bm_dist < 0.05:
                     print(f"There is a significant change in both median AND distribution of {bm_name} between T0 and T2.")
                elif p_val_bm_median < 0.05:
                    print(f"There is a significant change in the median of {bm_name} between T0 and T2.")
                elif p_val_bm_dist < 0.05:
                    print(f"There is a significant change in the distribution of {bm_name} between T0 and T2.")
                else:
                    print(f"No significant change in {bm_name} between T0 and T2.")

                plt.figure(figsize=(10, 4))
                plt.subplot(1,2,1)
                sns.histplot(df_t0_bm_cleaned[bm_col], color="skyblue", label='T0', kde=True, stat="density")
                sns.histplot(df_t2_bm_cleaned[bm_col], color="red", label='T2', kde=True, stat="density", alpha=0.7)
                plt.title(f'Distribution of {bm_name}')
                plt.legend()

                plt.subplot(1,2,2)
                sns.boxplot(data=pd.concat([df_t0_bm_cleaned[[bm_col]].assign(Timepoint='T0'),
                                           df_t2_bm_cleaned[[bm_col]].assign(Timepoint='T2')]),
                            x='Timepoint', y=bm_col)
                plt.title(f'Boxplot of {bm_name}')
                plt.tight_layout()
                plt.show()
        else:
            print(f"Column {bm_col} not found in T0 or T2 data.")


    # --- Pair patients to analyze changes (if possible) ---
    # Assume 'Number' column is the unique identifier for patients
    if 'Number' in df.columns:
        # Create a pivot DataFrame where each patient is a row, and T0 and T2 biomarkers are columns
        df_pivot_lymph = df.pivot_table(index='Number', columns='Timepoint', values=lymph_col).reset_index()
        df_pivot_pain = df.pivot_table(index='Number', columns='Timepoint', values='Pain_Rating').reset_index()

        # Rename columns for clarity
        df_pivot_lymph.columns = ['Number', 'Lymph_T0', 'Lymph_T2']
        df_pivot_pain.columns = ['Number', 'Pain_T0', 'Pain_T2']

        # Merge these pivot DataFrames
        df_paired = pd.merge(df_pivot_lymph, df_pivot_pain, on='Number', how='inner')
        df_paired.dropna(inplace=True) # Remove patients who do not have both T0 and T2 data

        if not df_paired.empty:
            df_paired['Lymph_Change'] = df_paired['Lymph_T2'] - df_paired['Lymph_T0']
            df_paired['Pain_Change'] = df_paired['Pain_T2'] - df_paired['Pain_T0'] # Usually a negative value if pain decreases

            print("\n--- Analysis of changes in paired patients ---")
            print(df_paired[['Lymph_Change', 'Pain_Change']].describe())

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='Lymph_Change', y='Pain_Change', data=df_paired)
            plt.title(f'{lymph_col} Change vs. {pain_col} Change')
            plt.xlabel(f'{lymph_col} Change (T2 - T0)')
            plt.ylabel(f'{pain_col} Change (T2 - T0)')
            plt.grid(True)
            plt.show()

            # Correlation between changes
            correlation_changes = df_paired[['Lymph_Change', 'Pain_Change']].corr()
            print("\nCorrelation between Lymphocyte change and Pain Rating change:")
            print(correlation_changes)
        else:
            print("Cannot pair enough patients to analyze changes.")
    else:
        print("No 'Number' column to pair patients.")


else:
    print(f"Error: Column '{lymph_col}' not found in T0 or T2 data.")

print("\n--- Analysis complete ---")