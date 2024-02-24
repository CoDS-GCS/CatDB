# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_4_rnc/dataset_4_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_4_rnc/dataset_4_rnc_test.csv')

# Perform data cleaning and preprocessing
# As per the given schema, there are no categorical variables. So, we don't need to perform any encoding.
# Also, there is no mention of missing values. So, we assume that there are no missing values.
# However, we need to scale the features as they have different ranges.

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data.drop(columns=['c_61']))
test_data_scaled = scaler.transform(test_data.drop(columns=['c_61']))

# Select the appropriate features and target variables for the question
# As per the given schema, 'c_61' is the target variable and all other columns are features.

X_train = train_data_scaled
y_train = train_data['c_61']

X_test = test_data_scaled
y_test = test_data['c_61']

# Choose the suitable machine learning algorithm or technique (classifier)
# As the target variable is binary, we can use Logistic Regression for this problem.
# Logistic Regression is a simple yet powerful algorithm that works well for binary classification problems.

clf = LogisticRegression()
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
# We use accuracy and f1 score as evaluation metrics.

y_pred = clf.predict(X_test)

Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")