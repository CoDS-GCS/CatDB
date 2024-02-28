# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_6_rnc/dataset_6_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_6_rnc/dataset_6_rnc_test.csv')

# Perform data cleaning and preprocessing
# In this case, we assume that the data is already clean and does not contain any missing or incorrect values.
# If this is not the case, appropriate data cleaning steps should be added here.

# Perform feature processing
# In this case, all features are numerical, so we don't need to encode categorical values.
# However, we will standardize the features to have zero mean and unit variance.

# Select the appropriate features and target variables for the question
# In this case, we assume that all features are relevant and the target variable is 'c_11'.
X_train = train_data.drop(columns=['c_11'])
y_train = train_data['c_11']
X_test = test_data.drop(columns=['c_11'])
y_test = test_data['c_11']

# Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier
# In this case, we assume that all features are relevant and do not drop any columns.
# If this is not the case, appropriate feature selection steps should be added here.

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
# In this case, we use Logistic Regression as it is a simple and effective algorithm for binary classification problems.
# It is also suitable for multi-threaded environments and can be easily parallelized across multiple CPUs.
clf = LogisticRegression(random_state=0, solver='saga', multi_class='ovr', n_jobs=-1)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")