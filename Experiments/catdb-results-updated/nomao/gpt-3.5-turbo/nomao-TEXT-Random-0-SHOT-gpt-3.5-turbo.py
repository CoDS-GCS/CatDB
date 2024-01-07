# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets
train_data = pd.read_csv('data/nomao/nomao_train.csv')
test_data = pd.read_csv('data/nomao/nomao_test.csv')
# end-load-dataset

# python-added-column
# Add new column V36_V2
# Usefulness: This column represents the product of V36 and V2, capturing the interaction between these two attributes.
train_data['V36_V2'] = train_data['V36'] * train_data['V2']
test_data['V36_V2'] = test_data['V36'] * test_data['V2']

# Add new column V10_V30
# Usefulness: This column represents the product of V10 and V30, capturing the interaction between these two attributes.
train_data['V10_V30'] = train_data['V10'] * train_data['V30']
test_data['V10_V30'] = test_data['V10'] * test_data['V30']

# Add new column V44_V35
# Usefulness: This column represents the product of V44 and V35, capturing the interaction between these two attributes.
train_data['V44_V35'] = train_data['V44'] * train_data['V35']
test_data['V44_V35'] = test_data['V44'] * test_data['V35']

# Add new column V34_V99
# Usefulness: This column represents the product of V34 and V99, capturing the interaction between these two attributes.
train_data['V34_V99'] = train_data['V34'] * train_data['V99']
test_data['V34_V99'] = test_data['V34'] * test_data['V99']
# end-added-column

# python-dropping-columns
# Drop V36, V2, V10, V30, V44, V35, V34, V99
# Explanation: These columns are dropped as they are used to create new columns and may be redundant.
train_data.drop(columns=['V36', 'V2', 'V10', 'V30', 'V44', 'V35', 'V34', 'V99'], inplace=True)
test_data.drop(columns=['V36', 'V2', 'V10', 'V30', 'V44', 'V35', 'V34', 'V99'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the binary classification technique
# Explanation: Logistic Regression is a commonly used algorithm for binary classification problems.
# It is selected here for its simplicity and interpretability.
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
# end-evaluation
# 