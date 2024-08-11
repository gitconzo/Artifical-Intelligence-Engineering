import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def decision_tree(csv_file, feature_cols, target_col, dataset_name):
    df = pd.read_csv(csv_file)

    # Split dataset into features and target
    X = df[feature_cols]
    y = df[target_col]

    # 70-30 split using train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create Decision Tree classifier
    clf = DecisionTreeClassifier()

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

datasets = {
    "water_potability_cleaned.csv": (['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                                      'Organic_carbon', 'Trihalomethanes', 'Turbidity'], 'Potability'),
    "water_potability_normalised.csv": (['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                                         'Organic_carbon', 'Trihalomethanes', 'Turbidity'], 'Potability'),
    "water_potability_features.csv": (['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                                       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'WQI', 'OCI',
                                       'TSI', 'CCI', 'PLI'], 'Potability'),
    "water_potability_selected_features.csv": (['ph', 'Chloramines', 'Sulfate', 'Conductivity',
                                                'Organic_carbon'], 'Potability'),
    "water_potability_selected_converted.csv": (['ph', 'Hardness', 'Chloramines', 'Sulfate'], 'Potability')
}

# Evaluate each feature set
results = {}
for file, (features, target) in datasets.items():
    accuracy = decision_tree(file, features, target, file)
    results[file] = accuracy

results_df = pd.DataFrame(list(results.items()), columns=['Feature Set', 'Accuracy'])
print(results_df)