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
train_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_train.csv")
test_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_test.csv")
# ```end

# ```python
# Handle missing values
# Since the missing value frequency is low for all columns, we can use mean imputation
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# c_27 is a static column with all values being 0.0, so we drop it
train_data.drop(columns=['c_27'], inplace=True)
test_data.drop(columns=['c_27'], inplace=True)
# ```end-dropping-columns

# ```python
# Feature engineering
# We can create a new feature that is a combination of 'c_5' and 'c_14' as they have a wide range of values and might have a combined effect on 'c_24'
# Usefulness: This adds useful real world knowledge to classify 'c_24' as it might capture complex relationships between 'c_5' and 'c_14' that the model might not be able to learn.
train_data['c_5_c_14'] = train_data['c_5'] * train_data['c_14']
test_data['c_5_c_14'] = test_data['c_5'] * test_data['c_14']
# ```end

# ```python
# Prepare data for training
X_train = train_data.drop(columns=['c_24'])
y_train = train_data['c_24']
X_test = test_data.drop(columns=['c_24'])
y_test = test_data['c_24']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets. It can capture complex patterns and interactions between features.
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