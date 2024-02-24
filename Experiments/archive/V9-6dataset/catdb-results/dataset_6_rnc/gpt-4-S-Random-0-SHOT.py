# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_6_rnc/dataset_6_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_6_rnc/dataset_6_rnc_test.csv')

# Perform data cleaning and preprocessing
# In this case, we assume that the data is already clean and does not require any additional preprocessing

# Perform feature processing
# In this case, we assume that all features are numerical and do not require any encoding

# Select the appropriate features and target variables for the question
features = ['c_6', 'c_7', 'c_1', 'c_5', 'c_9', 'c_3', 'c_8', 'c_2', 'c_4', 'c_10']
target = 'c_11'

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
# We choose Logistic Regression as it is a simple and effective algorithm for binary classification problems
# It is also less prone to overfitting compared to more complex models
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred, average='macro')

# Print the accuracy result
print(f"Accuracy: {Accuracy}")

# Print the f1 score result
print(f"F1_score: {F1_score}")