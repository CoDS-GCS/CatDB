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
for column in train_data.columns:
    if len(train_data[column].unique()) < 2:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
# ```end

# ```python
# Split the data into features and target variable
X_train = train_data.drop(columns=['c_61'])
y_train = train_data['c_61']
X_test = test_data.drop(columns=['c_61'])
y_test = test_data['c_61']
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Make predictions on the test set
y_pred = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
# ```end

# ```python
# Print the accuracy result
print(f"Accuracy:{Accuracy}")   
# ```end

# ```python
# Print the f1 score result
print(f"F1_score:{F1_score}") 
# ```end