# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/vehicle/vehicle_train.csv')
test_data = pd.read_csv('data/vehicle/vehicle_test.csv')
# end-load-dataset

# python-added-column
# Add new column: Ratio of elongatedness to circularity
# Usefulness: Elongatedness and circularity are important attributes for classifying 'Class'. 
# This new column captures the ratio between elongatedness and circularity, providing additional information.
train_data['ELONGATEDNESS_CIRCULARITY_RATIO'] = train_data['ELONGATEDNESS'] / train_data['CIRCULARITY']
test_data['ELONGATEDNESS_CIRCULARITY_RATIO'] = test_data['ELONGATEDNESS'] / test_data['CIRCULARITY']

# Add new column: Ratio of radius ratio to scatter ratio
# Usefulness: The ratio of radius ratio to scatter ratio can provide insights into the shape characteristics of the vehicles.
train_data['RADIUS_SCATTER_RATIO'] = train_data['RADIUS_RATIO'] / train_data['SCATTER_RATIO']
test_data['RADIUS_SCATTER_RATIO'] = test_data['RADIUS_RATIO'] / test_data['SCATTER_RATIO']

# Add new column: Ratio of compactness to aspect ratio
# Usefulness: The ratio of compactness to aspect ratio can indicate the overall shape compactness of the vehicles.
train_data['COMPACTNESS_ASPECT_RATIO'] = train_data['COMPACTNESS'] / train_data['PR.AXIS_ASPECT_RATIO']
test_data['COMPACTNESS_ASPECT_RATIO'] = test_data['COMPACTNESS'] / test_data['PR.AXIS_ASPECT_RATIO']
# end-added-column

# python-dropping-columns
# Drop column 'MAX.LENGTH_ASPECT_RATIO' as it may be redundant and hurt the predictive performance
train_data.drop(columns=['MAX.LENGTH_ASPECT_RATIO'], inplace=True)
test_data.drop(columns=['MAX.LENGTH_ASPECT_RATIO'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the multiclass classification technique
# Explanation: Logistic Regression is a commonly used algorithm for multiclass classification problems.
# It is suitable for this dataset as it can handle both numerical and categorical features.
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate on the test dataset
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']

# Standardize the test features
X_test_scaled = scaler.transform(X_test)

# Predict the class labels
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation
# 