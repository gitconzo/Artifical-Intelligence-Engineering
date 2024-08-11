import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

file_path = 'water_potability_cleaned.csv'
df = pd.read_csv(file_path)

columns_to_normalise = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                        'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Copy to preserve og data
df_normalised = df.copy()

# Min-Max Normalisation
for column in columns_to_normalise:
    min_val = df_normalised[column].min()
    max_val = df_normalised[column].max()
    df_normalised[column] = (df_normalised[column] - min_val) / (max_val - min_val)

# Creating new DataFrame to form normalised csv
df_normalised.to_csv('water_potability_normalised.csv', index=False)

print(df_normalised.head())