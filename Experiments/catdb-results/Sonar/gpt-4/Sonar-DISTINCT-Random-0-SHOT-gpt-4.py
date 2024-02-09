# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('data/Sonar/Sonar_train.csv')
test_data = pd.read_csv('data/Sonar/Sonar_test.csv')

# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.nunique() / train_data.shape[0] > 0.1]

# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a robust and versatile classifier that works well on a wide range of datasets
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Check if 'Class' column exists in the dataframe before dropping
if 'Class' in train_data.columns:
    X_train = train_data.drop(columns=['Class'])
    y_train = train_data['Class']
    clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
# Check if 'Class' column exists in the dataframe before dropping
if 'Class' in test_data.columns:
    X_test = test_data.drop(columns=['Class'])
    y_test = test_data['Class']
    if 'Class' in train_data.columns:
        y_pred = clf.predict(X_test)

        # Calculate the model accuracy
        Accuracy = accuracy_score(y_test, y_pred)

        # Calculate the model f1 score
        F1_score = f1_score(y_test, y_pred)

        # Print the accuracy result
        print(f"Accuracy:{Accuracy}")

        # Print the f1 score result
        print(f"F1_score:{F1_score}")