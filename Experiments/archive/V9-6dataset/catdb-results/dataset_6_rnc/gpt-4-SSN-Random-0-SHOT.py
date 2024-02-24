# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_6_rnc/dataset_6_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_6_rnc/dataset_6_rnc_test.csv')

# Perform data cleaning and preprocessing
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data.drop(columns=['c_11']))
test_data_scaled = scaler.transform(test_data.drop(columns=['c_11']))

# Select the appropriate features and target variables for the question
X_train = train_data_scaled
y_train = train_data['c_11']

X_test = test_data_scaled
y_test = test_data['c_11']

# Choose the suitable machine learning algorithm or technique (classifier)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")