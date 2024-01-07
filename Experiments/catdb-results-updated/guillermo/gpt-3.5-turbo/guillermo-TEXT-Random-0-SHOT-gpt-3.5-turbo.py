# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 

# python-load-dataset
# load train and test datasets (csv file formats) here 
train_data = pd.read_csv('data/guillermo/guillermo_train.csv')
test_data = pd.read_csv('data/guillermo/guillermo_test.csv')
# 

# python-added-column
# Feature: V2606 + V208
# Usefulness: The combination of V2606 and V208 can provide additional information about the relationship between two variables, which can be useful for binary classification.
train_data['new_feature'] = train_data['V2606'] + train_data['V208']
test_data['new_feature'] = test_data['V2606'] + test_data['V208']
# 

# python-dropping-columns
# Dropping columns V2606 and V208 as they are no longer needed after creating the new feature
train_data.drop(columns=['V2606', 'V208'], inplace=True)
test_data.drop(columns=['V2606', 'V208'], inplace=True)
# 

# python-training-technique
# Use Logistic Regression as the binary classification technique
# Explanation: Logistic Regression is a commonly used algorithm for binary classification tasks and can handle both numerical and categorical features.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# 

# python-evaluation
# Report evaluation based on only test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")

# end-evaluation# 