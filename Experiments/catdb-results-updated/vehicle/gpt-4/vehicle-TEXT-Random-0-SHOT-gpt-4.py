# python-import
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_df = pd.read_csv("data/vehicle/vehicle_train.csv")
test_df = pd.read_csv("data/vehicle/vehicle_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: 'RADIUS_TO_GYRATION_RATIO'
# Usefulness: This feature indicates the ratio of the radius to the gyration, which might be a useful measure for the shape of the vehicle.
train_df['RADIUS_TO_GYRATION_RATIO'] = train_df['RADIUS_RATIO'] / train_df['SCALED_RADIUS_OF_GYRATION']
test_df['RADIUS_TO_GYRATION_RATIO'] = test_df['RADIUS_RATIO'] / test_df['SCALED_RADIUS_OF_GYRATION']
# end-added-column

# python-added-column
# Feature name and description: 'SCATTER_TO_ELONGATEDNESS_RATIO'
# Usefulness: This feature indicates the ratio of scatter to elongatedness, which might be a useful measure for the compactness of the vehicle.
train_df['SCATTER_TO_ELONGATEDNESS_RATIO'] = train_df['SCATTER_RATIO'] / train_df['ELONGATEDNESS']
test_df['SCATTER_TO_ELONGATEDNESS_RATIO'] = test_df['SCATTER_RATIO'] / test_df['ELONGATEDNESS']
# end-added-column

# python-dropping-columns
# Explanation why the column 'COMPACTNESS' is dropped: 
# Compactness is a measure of how closely the shape of an object approaches that of a mathematically compact shape. 
# But in our case, it might be highly correlated with other shape features like 'ELONGATEDNESS', 'SCATTER_RATIO', etc. 
# Hence, to avoid multicollinearity, we can drop this feature.
train_df.drop(columns=['COMPACTNESS'], inplace=True)
test_df.drop(columns=['COMPACTNESS'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation why the solution is selected: RandomForestClassifier is selected because it is a robust and versatile classifier 
# that works well on multiclass classification problems. It also has the ability to handle a large number of features and 
# it provides importance of features, which can be useful for further feature selection.
X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation