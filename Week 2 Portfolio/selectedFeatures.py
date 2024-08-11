import pandas as pd

df_normalised = pd.read_csv('water_potability_normalised.csv')

selected_normalised_features = [
    'ph', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon'
]

# DataFrame with selected features plus the target column
df_selected_features = df_normalised[selected_normalised_features + ['Potability']]
df_selected_features.to_csv('water_potability_selected_features.csv', index=False)


df_cleansed = pd.read_csv('water_potability_cleaned.csv')

selected_cleansed_features = [
    'ph', 'Hardness', 'Chloramines', 'Sulfate',
]

df_selected_converted = df_cleansed[selected_cleansed_features + ['Potability']]
df_selected_converted.to_csv('water_potability_selected_converted.csv', index=False)