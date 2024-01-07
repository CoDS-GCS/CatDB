# python-import
# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv("data/credit-g/credit-g_train.csv")
test_data = pd.read_csv("data/credit-g/credit-g_test.csv")
# end-load-dataset

# python-added-column
# Feature name: 'employment_duration'
# Usefulness: This feature can be useful because the duration of employment can affect a person's ability to repay a loan.
# We assume that the longer a person is employed, the more stable their income is, which can affect their creditworthiness.
train_data['employment_duration'] = train_data['employment'].apply(lambda x: len(x))
test_data['employment_duration'] = test_data['employment'].apply(lambda x: len(x))
# end-added-column

# python-added-column
# Feature name: 'num_dependents_credit_amount_ratio'
# Usefulness: This feature can be useful because the ratio of credit amount to number of dependents can affect a person's ability to repay a loan.
# We assume that the higher the ratio, the more difficult it may be for a person to repay the loan.
train_data['num_dependents_credit_amount_ratio'] = train_data['credit_amount'] / train_data['num_dependents']
test_data['num_dependents_credit_amount_ratio'] = test_data['credit_amount'] / test_data['num_dependents']
# end-added-column

# python-dropping-columns
# Explanation why the column 'other_payment_plans' is dropped
# The 'other_payment_plans' column is dropped because it may not directly contribute to the binary classification task.
# Other payment plans may not directly affect a person's ability to repay a loan.
train_data.drop(columns=['other_payment_plans'], inplace=True)
test_data.drop(columns=['other_payment_plans'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: Random Forest classifier is selected because it is an ensemble learning method that can handle both categorical and numerical data.
# It is also robust to outliers and can handle unbalanced data, which makes it a good choice for this task.

# First, we need to encode the categorical features
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])

# Split the train data into training and validation sets
        
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
test_predictions = rf.predict(test_data.drop('class', axis=1))
test_accuracy = accuracy_score(test_data['class'], test_predictions)
print(f"Accuracy: {test_accuracy*100:.2f}")
# end-evaluation