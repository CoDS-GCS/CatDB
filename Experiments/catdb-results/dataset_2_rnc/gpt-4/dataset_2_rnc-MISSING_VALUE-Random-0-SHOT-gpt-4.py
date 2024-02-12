# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the training and test datasets
train_data = pd.read_csv("data/dataset_2_rnc/dataset_2_rnc_train.csv")
test_data = pd.read_csv("data/dataset_2_rnc/dataset_2_rnc_test.csv")

# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
for col in train_data.columns:
    if len(train_data[col].unique()) == len(train_data):
        train_data.drop(col,inplace=True,axis=1)

# Feature name and description: c_8_c_11_ratio
train_data['c_8_c_11_ratio'] = train_data['c_8'] / train_data['c_11']
test_data['c_8_c_11_ratio'] = test_data['c_8'] / test_data['c_11']

# Drop column c_14
train_data.drop(columns=['c_14'], inplace=True)
test_data.drop(columns=['c_14'], inplace=True)

# Use a LabelEncoder technique
le = LabelEncoder()
train_data['c_21'] = le.fit_transform(train_data['c_21'])
test_data['c_21'] = le.transform(test_data['c_21'])

# Convert all categorical variables into numerical ones
le_dict = {}
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        le_dict[col] = le

for col in test_data.columns:
    if test_data[col].dtype == 'object':
        if col in le_dict:
            le = le_dict[col]
            test_data[col] = le.transform(test_data[col])

# Ensure that the test data has the same columns as the training data
test_data = test_data[train_data.columns]

# Use a RandomForestClassifier technique
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}")