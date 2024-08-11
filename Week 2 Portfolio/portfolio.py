import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.width', None)        # Disable column width limit
pd.options.display.float_format = '{:.7f}'.format  # Set float display format

file_path = 'water_potability.csv'
df = pd.read_csv(file_path)

# Potability already exists as the target variable, as it is categorical because it indicates whether
# or no the water is drinkable
# 0: Not potable
# 1: Potable

df = df.drop_duplicates()
df.fillna(0, inplace=True)  # Replace NaNs with 0 in order for data to work

def remove_outliers(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

df_no_outliers = remove_outliers(df)

df_no_outliers.to_csv('water_potability_cleaned.csv', index=False)

# Check outliers’ correction by comparing the shape before and after
print(f"Original shape: {df.shape}")
print(f"Shape after removing outliers: {df_no_outliers.shape}")

# 3. Check for missing values
missing_values = df.isna().sum()
empty_strings = (df == '').sum()
combined_missing = missing_values + empty_strings
print("Combined missing values and empty strings in each column:")
print(combined_missing)

print(df.head())