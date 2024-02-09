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
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
# ```end

# ```python
# Separate the features and the target variable for training data
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']

# Separate the features and the target variable for test data
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Predict the target variable for test data
y_pred = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end