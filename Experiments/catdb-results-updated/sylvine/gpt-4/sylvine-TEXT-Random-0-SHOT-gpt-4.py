# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv('data/sylvine/sylvine_train.csv')
test = pd.read_csv('data/sylvine/sylvine_test.csv')
# end-load-dataset

# python-added-column
# Feature name: 'V5_V17_sum'
# Usefulness: This feature is the sum of V5 and V17. This could be useful if there is a relationship between these two features that helps predict the class.
train['V5_V17_sum'] = train['V5'] + train['V17']
test['V5_V17_sum'] = test['V5'] + test['V17']
# end-added-column

# python-added-column
# Feature name: 'V3_V13_diff'
# Usefulness: This is the difference between V3 and V13. This could be useful if the difference between these two features has an impact on the class.
train['V3_V13_diff'] = train['V3'] - train['V13']
test['V3_V13_diff'] = test['V3'] - test['V13']
# end-added-column

# python-dropping-columns
# Explanation why the column V2 is dropped: V2 might be highly correlated with other features, causing multicollinearity, which can hurt the model's performance.
train.drop(columns=['V2'], inplace=True)
test.drop(columns=['V2'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: RandomForestClassifier is a powerful and versatile machine learning model that performs well on many types of data. It also handles multicollinearity well, which is beneficial given the nature of the dataset.

# Separate features and target
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation