# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# end-import

# python-load-dataset 
# load train and test datasets (csv file formats) here 
train = pd.read_csv('data/dionis/dionis_train.csv')
test = pd.read_csv('data/dionis/dionis_test.csv')
# end-load-dataset 

# python-added-column 
# Feature name: sum_features
# Usefulness: This feature represents the sum of all features for a given sample. It can provide an overall sense of the scale of the feature values for that sample.
train['sum_features'] = train.drop('class', axis=1).sum(axis=1)
test['sum_features'] = test.drop('class', axis=1).sum(axis=1)
# end-added-column

# python-added-column 
# Feature name: mean_features
# Usefulness: This feature represents the mean of all features for a given sample. It can provide an overall sense of the average of the feature values for that sample.
train['mean_features'] = train.drop(['class', 'sum_features'], axis=1).mean(axis=1)
test['mean_features'] = test.drop(['class', 'sum_features'], axis=1).mean(axis=1)
# end-added-column

# python-dropping-columns
# Explanation why the column V1 is dropped
# Column V1 is dropped because it has a high correlation with V2, thus it's redundant and may lead to overfitting.
train.drop(columns=['V1'], inplace=True)
test.drop(columns=['V1'], inplace=True)
# end-dropping-columns

# python-training-technique 
# Use a RandomForestClassifier
# Explanation why the solution is selected: RandomForestClassifier is a powerful ensemble method that can handle a large number of features and it is less prone to overfitting.
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Create a pipeline with a standard scaler and a random forest classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
