# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_2_rnc/dataset_2_rnc_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Drop columns with high frequency of missing values
train_data.drop(columns=['c_14', 'c_10', 'c_19'], inplace=True)
test_data.drop(columns=['c_14', 'c_10', 'c_19'], inplace=True)

# Encode categorical values
le = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
# ```end

# ```python
# Select the appropriate features and target variables for the question
# In this case, we are predicting 'c_21', so we will use all other columns as features
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']

X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# We will use Logistic Regression as it is a good baseline model for binary classification problems
# It is also efficient and does not require high computational resources, making it suitable for a multi-threaded environment with various CPU configurations
clf = LogisticRegression()
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy: {Accuracy}")

# Print the f1 score result
print(f"F1_score: {F1_score}")
# ```end