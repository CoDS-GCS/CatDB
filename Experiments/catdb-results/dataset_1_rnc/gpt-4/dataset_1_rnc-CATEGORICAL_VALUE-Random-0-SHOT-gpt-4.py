# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_train.csv")
test_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_test.csv")

# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.nunique() != 1]

# Feature name and description: c_5_c_14_ratio
# Usefulness: This feature represents the ratio of c_5 to c_14, which might be useful for the classification task.
train_data['c_5_c_14_ratio'] = train_data['c_5'] / train_data['c_14']
test_data['c_5_c_14_ratio'] = test_data['c_5'] / test_data['c_14']

# Explanation why the column c_27 is dropped: c_27 is a categorical column with only one distinct value, which does not provide any useful information for the classification task.
if 'c_27' in train_data.columns:
    train_data.drop(columns=['c_27'], inplace=True)
if 'c_27' in test_data.columns:
    test_data.drop(columns=['c_27'], inplace=True)

# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that can handle both categorical and numerical features. It also has built-in feature importance estimation, which can be useful for feature selection.
X_train = train_data.drop(columns=['c_24'])
y_train = train_data['c_24']
X_test = test_data.drop(columns=['c_24'])
y_test = test_data['c_24']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}")