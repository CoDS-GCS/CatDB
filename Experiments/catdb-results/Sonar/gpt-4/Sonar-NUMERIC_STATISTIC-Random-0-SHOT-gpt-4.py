# ```python
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/Sonar/Sonar_train.csv")
test_data = pd.read_csv("data/Sonar/Sonar_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if len(train_data[column].unique()) == 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Convert the Class column to numeric
le = LabelEncoder()
train_data['Class'] = le.fit_transform(train_data['Class'])
test_data['Class'] = le.transform(test_data['Class'])
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a meta estimator that fits a number of decision tree classifiers 
# on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(train_data.drop('Class', axis=1), train_data['Class'])
# ```end

# ```python
# Report evaluation based on only test dataset
predictions = clf.predict(test_data.drop('Class', axis=1))

# Calculate the model accuracy
Accuracy = accuracy_score(test_data['Class'], predictions)

# Calculate the model f1 score
F1_score = f1_score(test_data['Class'], predictions)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end