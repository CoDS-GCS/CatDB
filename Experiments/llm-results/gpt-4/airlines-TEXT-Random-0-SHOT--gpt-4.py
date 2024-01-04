# python-import
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
# end-import


# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('/home/saeed/Documents/Github/CatDB/Experiments/data/airlines/airlines_train.csv')
test_data = pd.read_csv('/home/saeed/Documents/Github/CatDB/Experiments/data/airlines/airlines_test.csv')
# end-load-dataset


# python-added-column
# Flight Length in Hour
# Usefulness: Flight length may affect the departure delay. If the flight is longer, it might more likely to be delayed.

train_data['LengthInHour'] = train_data['Length'] / 60
test_data['LengthInHour'] = test_data['Length'] / 60
# end-added-column

# python-added-column
# Split DayOfWeek to Weekday and Weekend
# Usefulness: Flights during weekdays might have different delay patterns compared to weekends.

train_data['IsWeekday'] = train_data['DayOfWeek'].apply(lambda x: 1 if x < 5 else 0)
test_data['IsWeekday'] = test_data['DayOfWeek'].apply(lambda x: 1 if x < 5 else 0)
# end-added-column

# python-dropping-columns
# Drop 'Flight' and 'Length' columns as they have been transformed into new features.
train_data.drop(columns=['Flight', 'Length'], inplace=True)
test_data.drop(columns=['Flight', 'Length'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use XGBoost Classifier as the downstream classifier.
# XGBoost is selected due to its versatility and it can model complex patterns in the data. 

# Encoding categorical attributes
for col in ['AirportTo', 'AirportFrom', 'Airline']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Split train_data into features and target
X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]

# Split test_data into features and target
X_test = test_data.drop("Delay", axis=1)
y_test = test_data["Delay"]

# Create a pipeline for XGBoost Classifier
pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Fit the model
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'The accuracy score on the test dataset is: {acc}')
# end-evaluation
