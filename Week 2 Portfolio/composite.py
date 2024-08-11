import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('water_potability_normalised.csv')

# Water Quality Index (WQI)
df['WQI'] = (df['ph'] + df['Chloramines'] + df['Sulfate'] + df['Conductivity']) / 4

# Organic Contaminant Index (OCI)
df['OCI'] = (df['Organic_carbon'] + df['Trihalomethanes']) / 2

# Total Solids Index (TSI)
df['TSI'] = (df['Solids'] + df['Turbidity']) / 2

# Chemical Contaminant Index (CCI)
df['CCI'] = (df['Chloramines'] + df['Sulfate'] + df['Organic_carbon']) / 3

# Clarity and Solids Index (CSI)
df['PLI'] = (df['Chloramines'] + df['Sulfate'] + df['Trihalomethanes']) / 3

df.to_csv('water_potability_features.csv', index=False)