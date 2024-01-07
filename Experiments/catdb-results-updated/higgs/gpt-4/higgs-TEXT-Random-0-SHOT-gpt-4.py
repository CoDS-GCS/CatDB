# python-import
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/higgs/higgs_train.csv")
test = pd.read_csv("data/higgs/higgs_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: 'total_energy'
# Usefulness: The total energy of the system can be a useful feature as it relates to the conservation of energy in particle physics.
# The total energy is calculated as the sum of the magnitudes of the energy of all the jets and the missing energy.
train['total_energy'] = train['jet1pt'] + train['jet2pt'] + train['jet3pt'] + train['jet4pt'] + train['missing_energy_magnitude']
test['total_energy'] = test['jet1pt'] + test['jet2pt'] + test['jet3pt'] + test['jet4pt'] + test['missing_energy_magnitude']
# end-added-column

# python-added-column
# Feature name and description: 'total_tag'
# Usefulness: The total tag of the system can be a useful feature as it relates to the number of b-tagged jets in the event.
# The total tag is calculated as the sum of the b-tags of all the jets.
train['total_tag'] = train['jet1b-tag'] + train['jet2b-tag'] + train['jet3b-tag'] + train['jet4b-tag']
test['total_tag'] = test['jet1b-tag'] + test['jet2b-tag'] + test['jet3b-tag'] + test['jet4b-tag']
# end-added-column

# python-dropping-columns
# Explanation why the column 'lepton_eta' is dropped
# 'lepton_eta' is a measure of the pseudorapidity of the lepton, which is a measure of the angle of the lepton's
# momentum vector with respect to the beam axis. This is a highly technical feature that may not be useful for our
# binary classification problem, and may lead to overfitting.
train.drop(columns=['lepton_eta'], inplace=True)
test.drop(columns=['lepton_eta'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# RandomForestClassifier is chosen due to its robustness to overfitting and ability to handle a large number of features.
# It also has the advantage of providing feature importances, which can be useful for further feature engineering.
X = train.drop('class', axis=1)
y = train['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
X_test = test.drop('class', axis=1)
y_test = test['class']

X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

X_test_scaled = scaler.transform(X_test)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation