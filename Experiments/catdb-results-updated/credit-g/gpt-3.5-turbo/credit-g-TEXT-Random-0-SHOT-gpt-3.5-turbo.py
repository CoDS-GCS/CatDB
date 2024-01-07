# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv("data/credit-g/credit-g_train.csv")
test_data = pd.read_csv("data/credit-g/credit-g_test.csv")
# end-load-dataset

# python-added-column
# Add new column 'employment_savings' combining 'employment' and 'savings_status'
train_data['employment_savings'] = train_data['employment'] + '_' + train_data['savings_status']
test_data['employment_savings'] = test_data['employment'] + '_' + test_data['savings_status']

# Add new column 'job_purpose' combining 'job' and 'purpose'
train_data['job_purpose'] = train_data['job'] + '_' + train_data['purpose']
test_data['job_purpose'] = test_data['job'] + '_' + test_data['purpose']

# Add new column 'credit_duration_ratio' as the ratio of 'credit_amount' to 'duration'
train_data['credit_duration_ratio'] = train_data['credit_amount'] / train_data['duration']
test_data['credit_duration_ratio'] = test_data['credit_amount'] / test_data['duration']
# end-added-column

# python-dropping-columns
# Drop columns that may be redundant or not useful for classification
train_data.drop(columns=['employment', 'savings_status', 'job', 'purpose'], inplace=True)
test_data.drop(columns=['employment', 'savings_status', 'job', 'purpose'], inplace=True)
# end-dropping-columns

# python-training-technique
# Encode the target variable 'class' to numeric values
le = LabelEncoder()
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.fit_transform(test_data[column])

# train_data['class'] = le.fit_transform(train_data['class'])

# Split the dataset into training and validation sets    
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

# Train a random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Calculate accuracy score
test_predictions = clf.predict(test_data.drop('class', axis=1))
test_accuracy = accuracy_score(test_data['class'], test_predictions)
print(f"Accuracy: {test_accuracy*100:.2f}")
# end-evaluation