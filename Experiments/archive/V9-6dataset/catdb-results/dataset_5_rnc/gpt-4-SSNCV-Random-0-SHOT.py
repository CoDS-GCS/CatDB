# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_5_rnc/dataset_5_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_5_rnc/dataset_5_rnc_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Fill missing values with the median of the column
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)
# ```end

# ```python
# Perform feature processing
# Encode categorical values by dummyEncode
le = LabelEncoder()
train_data['c_1'] = le.fit_transform(train_data['c_1'])
test_data['c_1'] = le.transform(test_data['c_1'])
# ```end

# ```python
# Select the appropriate features and target variables for the question
features = ['c_1', 'c_7', 'c_8', 'c_3', 'c_6', 'c_5', 'c_2', 'c_4']
target = 'c_9'
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
# ```end

# ```python
# Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier
# Explanation why the column c_1 is dropped: c_1 is a categorical column with only 3 distinct values, which may not provide much information for the classifier.
# df.drop(columns=['c_1'], inplace=True)
# ```end-dropping-columns

# ```python
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# Explanation why the solution is selected: Logistic Regression is a simple and fast algorithm for binary classification problems. It is also easy to interpret and understand.
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   

# Print the f1 score result
print(f"F1_score:{F1_score}") 
# ```end