# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")
test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")
# ```end

# ```python
# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)
# ```end

# ```python
# Drop columns with high missing value frequency
# Explanation: Columns with high missing value frequency can lead to overfitting and poor generalization to new data
columns_to_drop = ['c_65', 'c_54', 'c_36', 'c_68', 'c_13', 'c_70']
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)
# ```end-dropping-columns

# ```python
# Feature engineering
# Feature: c_53_c_76_interaction
# Usefulness: Interaction features can capture the combined effect of two or more features, which can be useful for the prediction task.
train_data['c_53_c_76_interaction'] = train_data['c_53'] * train_data['c_76']
test_data['c_53_c_76_interaction'] = test_data['c_53'] * test_data['c_76']
# ```end

# ```python
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets. It can handle binary classification tasks well.
X_train = train_data.drop(columns=['c_1'])
y_train = train_data['c_1']
X_test = test_data.drop(columns=['c_1'])
y_test = test_data['c_1']

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end