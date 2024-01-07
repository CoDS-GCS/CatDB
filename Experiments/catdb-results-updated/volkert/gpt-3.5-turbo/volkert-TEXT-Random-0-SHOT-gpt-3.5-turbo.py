# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/volkert/volkert_train.csv')
test_data = pd.read_csv('data/volkert/volkert_test.csv')
# end-load-dataset

# python-added-column
# Add new columns
train_data['V107_V110'] = train_data['V107'] * train_data['V110']
train_data['V158_V150'] = train_data['V158'] * train_data['V150']
train_data['V79_V66'] = train_data['V79'] * train_data['V66']
train_data['V130_V54'] = train_data['V130'] * train_data['V54']
train_data['V163_V49'] = train_data['V163'] * train_data['V49']

test_data['V107_V110'] = test_data['V107'] * test_data['V110']
test_data['V158_V150'] = test_data['V158'] * test_data['V150']
test_data['V79_V66'] = test_data['V79'] * test_data['V66']
test_data['V130_V54'] = test_data['V130'] * test_data['V54']
test_data['V163_V49'] = test_data['V163'] * test_data['V49']
# end-added-column

# python-dropping-columns
# Drop redundant columns
train_data.drop(columns=['V107', 'V110', 'V158', 'V150', 'V79', 'V66', 'V130', 'V54', 'V163', 'V49'], inplace=True)
test_data.drop(columns=['V107', 'V110', 'V158', 'V150', 'V79', 'V66', 'V130', 'V54', 'V163', 'V49'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the multiclass classification technique
trn_X = train_data.drop(columns=['class'])
trn_y = train_data['class']

# Split the train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(trn_X, trn_y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train the classifier
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate on the test dataset
test_X = test_data.drop(columns=['class'])
test_y = test_data['class']
test_X_scaled = scaler.transform(test_X)
pred_y = clf.predict(test_X_scaled)
accuracy = accuracy_score(test_y, pred_y)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation
# 