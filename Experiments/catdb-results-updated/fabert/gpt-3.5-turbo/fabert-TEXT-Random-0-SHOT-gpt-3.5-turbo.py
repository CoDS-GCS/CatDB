# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv("data/fabert/fabert_train.csv")
test_data = pd.read_csv("data/fabert/fabert_test.csv")
# end-load-dataset

# python-added-column
# Create a new column 'V744_V457' that represents the product of 'V744' and 'V457'
train_data['V744_V457'] = train_data['V744'] * train_data['V457']
test_data['V744_V457'] = test_data['V744'] * test_data['V457']

# Create a new column 'V165_V290' that represents the product of 'V165' and 'V290'
train_data['V165_V290'] = train_data['V165'] * train_data['V290']
test_data['V165_V290'] = test_data['V165'] * test_data['V290']

# Create a new column 'V204_V526' that represents the product of 'V204' and 'V526'
train_data['V204_V526'] = train_data['V204'] * train_data['V526']
test_data['V204_V526'] = test_data['V204'] * test_data['V526']

# Create a new column 'V707_V745' that represents the product of 'V707' and 'V745'
train_data['V707_V745'] = train_data['V707'] * train_data['V745']
test_data['V707_V745'] = test_data['V707'] * test_data['V745']
# end-added-column

# python-dropping-columns
# Drop columns that may be redundant and hurt the predictive performance
# Explanation: 'V321', 'V156', 'V774' are dropped as they are not relevant for the classification task
train_data.drop(columns=['V321', 'V156', 'V774'], inplace=True)
test_data.drop(columns=['V321', 'V156', 'V774'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the multiclass classification technique
# Explanation: Logistic Regression is a commonly used algorithm for multiclass classification tasks
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate the model on the test dataset
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation