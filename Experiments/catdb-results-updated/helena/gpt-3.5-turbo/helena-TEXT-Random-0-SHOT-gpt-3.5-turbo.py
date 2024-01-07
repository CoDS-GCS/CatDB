# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/helena/helena_train.csv')
test_data = pd.read_csv('data/helena/helena_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'V6_V21' as a product of 'V6' and 'V21'
train_data['V6_V21'] = train_data['V6'] * train_data['V21']
test_data['V6_V21'] = test_data['V6'] * test_data['V21']

# Add new column 'V18_V1' as a product of 'V18' and 'V1'
train_data['V18_V1'] = train_data['V18'] * train_data['V1']
test_data['V18_V1'] = test_data['V18'] * test_data['V1']

# Add new column 'V27_V9' as a product of 'V27' and 'V9'
train_data['V27_V9'] = train_data['V27'] * train_data['V9']
test_data['V27_V9'] = test_data['V27'] * test_data['V9']
# end-added-column

# python-dropping-columns
# Drop 'V2' column as it may be redundant and hurt the predictive performance
train_data.drop(columns=['V2'], inplace=True)
test_data.drop(columns=['V2'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the multiclass classification technique
clf = LogisticRegression()
X = train_data.drop(columns=['class'])
y = train_data['class']
clf.fit(X, y)
# end-training-technique

# python-evaluation
# Evaluate on test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation