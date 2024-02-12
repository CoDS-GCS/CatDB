# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_train.csv')
test_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
for col in train_data.columns:
    if len(train_data[col].unique()) < 2:
        train_data.drop(col,inplace=True,axis=1)
# ```end

# ```python
# Feature: c_1_c_9_interaction
# Usefulness: This feature captures the interaction between 'c_1' and 'c_9' which might be useful for the prediction of 'c_9'.
train_data['c_1_c_9_interaction'] = train_data['c_1'] * train_data['c_9']
test_data['c_1_c_9_interaction'] = test_data['c_1'] * test_data['c_9']
# ```end

# ```python-dropping-columns
# Explanation why the column c_2 is dropped: The column 'c_2' might be highly correlated with other columns and thus, it might not add much information for the prediction of 'c_9'.
train_data.drop(columns=['c_2'], inplace=True)
test_data.drop(columns=['c_2'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that can handle both categorical and numerical features. It also has the ability to handle large datasets and it provides feature importance which can be useful for feature selection.
X_train = train_data.drop('c_9', axis=1)
y_train = train_data['c_9']
X_test = test_data.drop('c_9', axis=1)
y_test = test_data['c_9']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end