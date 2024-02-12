# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_4_rnc/dataset_4_rnc_train.csv')
test_data = pd.read_csv('data/dataset_4_rnc/dataset_4_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
low_ratio_cols = [col for col in train_data.columns if train_data[col].nunique() < 2]
static_cols = [col for col in train_data.columns if train_data[col].std() < 0.1]
unique_cols = [col for col in train_data.columns if train_data[col].nunique() == train_data.shape[0]]

# Drop these columns from the dataframe
train_data.drop(columns=low_ratio_cols + static_cols + unique_cols, inplace=True)
test_data.drop(columns=low_ratio_cols + static_cols + unique_cols, inplace=True)
# ```end

# ```python
# Feature engineering
# Add a new feature that is the mean of all the features
# Usefulness: This feature can capture the overall trend of the data and can be useful for the classifier.
train_data['mean'] = train_data.mean(axis=1)
test_data['mean'] = test_data.mean(axis=1)
# ```end

# ```python
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a robust and versatile classifier that can handle both binary and multiclass tasks. It can also handle a mix of categorical and numerical features, and it does not require feature scaling.
X_train = train_data.drop(columns=['c_61'])
y_train = train_data['c_61']
X_test = test_data.drop(columns=['c_61'])
y_test = test_data['c_61']

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