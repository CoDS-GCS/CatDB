# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_5_rnc/dataset_5_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_5_rnc/dataset_5_rnc_test.csv')

# Perform data cleaning and preprocessing
# In this case, we assume that the data is already clean and does not contain any missing or incorrect values.
# If this is not the case, appropriate data cleaning steps should be added here.

# Perform feature processing
# In this case, all features are numerical, so no encoding is necessary.
# If there were categorical features, they would need to be encoded here.

# Select the appropriate features and target variables for the question
# In this case, we are predicting 'c_9', so it is our target variable. All other columns are our features.
X_train = train_data.drop('c_9', axis=1)
y_train = train_data['c_9']
X_test = test_data.drop('c_9', axis=1)
y_test = test_data['c_9']

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
# In this case, we are using Logistic Regression because it is a simple and effective algorithm for binary classification problems.
# It is also relatively fast and does not require a lot of computational resources, which makes it suitable for a multi-threaded environment with various CPU configurations.
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")