# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv('data/adult/adult_train.csv')
test = pd.read_csv('data/adult/adult_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: is_married
# Usefulness: Marital status can have an impact on the income of an individual. 
# Married individuals may have combined income and therefore may have a higher income.
train['is_married'] = train['marital-status'].apply(lambda x: 1 if 'Married' in x else 0)
test['is_married'] = test['marital-status'].apply(lambda x: 1 if 'Married' in x else 0)
# end-added-column

# python-dropping-columns
# Explanation: The column 'fnlwgt' is dropped because it represents final weight, which is irrelevant to the income of an individual.
train.drop(columns=['fnlwgt'], inplace=True)
test.drop(columns=['fnlwgt'], inplace=True)
# end-dropping-columns

# python-other
# Explanation: Label encoding is required to convert categorical data into numerical data for the model to process.
le = LabelEncoder()
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
# end-other

# python-training-technique
# Use a binary classification technique
# Explanation: Random Forest Classifier is used because it is a versatile algorithm that can handle both categorical and numerical data. 
# It also performs well on large datasets and can handle many features without overfitting.
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

pipeline = Pipeline(steps=[('s',StandardScaler()), ('m',RandomForestClassifier())])
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation