# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv('data/numerai28.6/numerai28.6_train.csv')
test = pd.read_csv('data/numerai28.6/numerai28.6_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: 'mean_attributes'
# Usefulness: This feature calculates the mean of all attributes for each row. This might be useful because it can capture the overall trend of the attributes for each instance.
train['mean_attributes'] = train.mean(axis=1)
test['mean_attributes'] = test.mean(axis=1)
# end-added-column

# python-added-column
# Feature name and description: 'sum_attributes'
# Usefulness: This feature calculates the sum of all attributes for each row. This might be useful because it can capture the overall magnitude of the attributes for each instance.
train['sum_attributes'] = train.sum(axis=1)
test['sum_attributes'] = test.sum(axis=1)
# end-added-column

# python-dropping-columns
# Dropping attribute_0 because it has the least correlation with the target variable
correlation = train.corr()
least_corr = correlation['attribute_21'].idxmin()
train.drop(columns=[least_corr], inplace=True)
test.drop(columns=[least_corr], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# I chose Random Forest Classifier because it works well with a large number of features and it's less likely to overfit.
X_train = train.drop('attribute_21', axis=1)
y_train = train['attribute_21']
X_test = test.drop('attribute_21', axis=1)
y_test = test['attribute_21']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation