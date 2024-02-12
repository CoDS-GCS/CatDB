# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the training and test datasets
train_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_test.csv')

# Fill missing values in the dataset
train_data.fillna(train_data.select_dtypes(include=['float64','int64']).median(), inplace=True)
test_data.fillna(test_data.select_dtypes(include=['float64','int64']).median(), inplace=True)

# Feature: c_8_c_11_ratio
train_data['c_8_c_11_ratio'] = train_data['c_8'] / train_data['c_11']
test_data['c_8_c_11_ratio'] = test_data['c_8'] / test_data['c_11']

# Drop columns with high frequency of missing values
train_data.drop(columns=['c_14', 'c_10', 'c_19'], inplace=True)
test_data.drop(columns=['c_14', 'c_10', 'c_19'], inplace=True)

# Use a LabelEncoder technique
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = le.fit_transform(train_data[column].astype(str))

for column in test_data.columns:
    if test_data[column].dtype == 'object':
        if column in le.classes_:
            test_data[column] = le.transform(test_data[column].astype(str))
        else:
            test_data[column] = le.fit_transform(test_data[column].astype(str))

# Use a RandomForestClassifier technique
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(train_data.drop('c_21', axis=1), train_data['c_21'])

# Report evaluation based on only test dataset
predictions = clf.predict(test_data.drop('c_21', axis=1))
Accuracy = accuracy_score(test_data['c_21'], predictions)
F1_score = f1_score(test_data['c_21'], predictions, average='weighted')

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")