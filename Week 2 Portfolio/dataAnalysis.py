import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Used to suppress display of warnings
import warnings

# OLs library
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# import missingno as mno
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

# import zscore for scaling the data
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Pre-processing methods
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.compose import TransformedTargetRegressor

# The regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor

# Cross-validation methods
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics

from sklearn.pipeline import Pipeline

# Feature-selection methods
from sklearn.feature_selection import SelectFromModel

# Bootstrap sampling
from sklearn.utils import resample

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.width', None)        # Disable column width limit
pd.options.display.float_format = '{:.7f}'.format  # Set float display format

file_path = 'water_potability.csv'
df = pd.read_csv(file_path).replace('', np.nan).astype(float)

target = 'Potability'
predictors = df.columns[df.columns != target]

# 2. histograms for each predictor
df[predictors].hist(figsize=(15, 10), bins=30)
plt.suptitle('Univariate Analysis: Histograms of Predictors', y=1.02)
plt.show()

# 3. Summary
summary_stats = df.describe()
print(summary_stats)

# 4. Scatter plot matrix
sns.pairplot(df, diag_kind='kde', hue=target)
plt.suptitle('Multivariate Analysis: Pairplot', y=1.02)
plt.show()

# 5. heatmap with pairwise
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Pairwise Correlations')
plt.show()