# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/jannis/jannis_train.csv')
test_data = pd.read_csv('data/jannis/jannis_test.csv')
# end-load-dataset

# python-added-column
# Add new columns
train_data['V19_V9'] = train_data['V19'] * train_data['V9']
train_data['V39_V47'] = train_data['V39'] + train_data['V47']
train_data['V33_V41'] = train_data['V33'] - train_data['V41']
train_data['V6_V1'] = train_data['V6'] / train_data['V1']
train_data['V8_V16'] = train_data['V8'] * train_data['V16']

test_data['V19_V9'] = test_data['V19'] * test_data['V9']
test_data['V39_V47'] = test_data['V39'] + test_data['V47']
test_data['V33_V41'] = test_data['V33'] - test_data['V41']
test_data['V6_V1'] = test_data['V6'] / test_data['V1']
test_data['V8_V16'] = test_data['V8'] * test_data['V16']
# end-added-column

# python-dropping-columns
# Drop columns
train_data.drop(columns=['V17', 'V21', 'V50', 'V10', 'V11', 'V24', 'V42', 'V36', 'V30', 'V29', 'V3', 'V35', 'V4', 'V25', 'V22', 'V43', 'V53', 'V45', 'V48', 'V46', 'V13', 'V20', 'V27', 'V52', 'V54', 'V12', 'V14', 'V51', 'V34', 'V32', 'V18', 'V49', 'V31', 'V5', 'V40', 'V23', 'V38', 'V26', 'V28', 'V7', 'V2', 'V15'], inplace=True)
test_data.drop(columns=['V17', 'V21', 'V50', 'V10', 'V11', 'V24', 'V42', 'V36', 'V30', 'V29', 'V3', 'V35', 'V4', 'V25', 'V22', 'V43', 'V53', 'V45', 'V48', 'V46', 'V13', 'V20', 'V27', 'V52', 'V54', 'V12', 'V14', 'V51', 'V34', 'V32', 'V18', 'V49', 'V31', 'V5', 'V40', 'V23', 'V38', 'V26', 'V28', 'V7', 'V2', 'V15'], inplace=True)
# end-dropping-columns

# python-training-technique
# Train the classifier
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate the classifier
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 