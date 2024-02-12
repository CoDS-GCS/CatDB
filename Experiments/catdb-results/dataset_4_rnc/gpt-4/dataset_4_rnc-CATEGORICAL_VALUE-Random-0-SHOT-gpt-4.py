# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_4_rnc/dataset_4_rnc_train.csv")
test_data = pd.read_csv("data/dataset_4_rnc/dataset_4_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
for col in train_data.columns:
    if len(train_data[col].unique()) < 2:
        train_data.drop(col,inplace=True,axis=1)
# ```end

# ```python
# Feature name and description: c_4_c_26_interaction
# Usefulness: This feature represents the interaction between 'c_4' and 'c_26'. This could be useful if the effect of 'c_4' on the target variable 'c_61' depends on the value of 'c_26'.
train_data['c_4_c_26_interaction'] = train_data['c_4'] * train_data['c_26']
test_data['c_4_c_26_interaction'] = test_data['c_4'] * test_data['c_26']
# ```end

# ```python-dropping-columns
# Explanation why the column c_1 is dropped: Assuming 'c_1' is highly correlated with other features and does not provide additional information for the prediction of 'c_61'.
train_data.drop(columns=['c_1'], inplace=True)
test_data.drop(columns=['c_1'], inplace=True)
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on both linear and non-linear problems. It also handles feature interactions well, which is useful given the new feature we created.
X_train = train_data.drop('c_61', axis=1)
y_train = train_data['c_61']
X_test = test_data.drop('c_61', axis=1)
y_test = test_data['c_61']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
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