# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/Sonar/Sonar_train.csv")
test_data = pd.read_csv("data/Sonar/Sonar_test.csv")
# ```end

# ```python
# Drop columns with low distinct-count
# Explanation: Columns with low distinct-count may not provide useful information for the classifier
low_distinct_cols = ['V53', 'V60', 'V54', 'V52', 'V56', 'V51', 'V55', 'V58', 'V57', 'V59', 'V1']
train_data.drop(columns=low_distinct_cols, inplace=True)
test_data.drop(columns=low_distinct_cols, inplace=True)
# ```end-dropping-columns

# ```python
# Add new feature: mean of all features
# Usefulness: This new feature may capture the overall level of the sonar return, which could be useful for the classifier
train_data['mean_features'] = train_data.mean(axis=1)
test_data['mean_features'] = test_data.mean(axis=1)
# ```end

# ```python
# Prepare data for training
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']
# ```end

# ```python
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a robust and versatile classifier that can handle both binary and multiclass tasks.
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