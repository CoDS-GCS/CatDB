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
# In this case, we don't have any columns to remove based on the provided schema and data profiling info
# ```end

# ```python
# c_1 is a categorical column with only 3 distinct values. We can convert it to dummy variables.
# Usefulness: This adds useful real world knowledge to classify 'c_9' as it increases the dimensionality of the data, allowing the model to capture more complex patterns.
train_data = pd.get_dummies(train_data, columns=['c_1'])
test_data = pd.get_dummies(test_data, columns=['c_1'])
# ```end

# ```python
# Explanation why the column XX is dropped
# In this case, we don't have any columns to drop based on the provided schema and data profiling info
# ```end

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets. It can handle both categorical and numerical features, and it also has built-in feature importance estimation.
X_train = train_data.drop(columns=['c_9'])
y_train = train_data['c_9']
X_test = test_data.drop(columns=['c_9'])
y_test = test_data['c_9']

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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