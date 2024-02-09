# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/Sonar/Sonar_train.csv')
test_data = pd.read_csv('data/Sonar/Sonar_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if train_data[column].nunique() <= 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Split the data into features and target variable
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has the feature_importances_ attribute which can help in understanding the most important features for the task.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end