# python-import
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv('data/covertype/covertype_train.csv')
test = pd.read_csv('data/covertype/covertype_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: Distance_to_Hydrology
# Usefulness: This feature combines horizontal and vertical distance to hydrology, 
# which could provide a more comprehensive measure of distance to water bodies.
train['Distance_to_Hydrology'] = (train['Horizontal_Distance_To_Hydrology']**2 + train['Vertical_Distance_To_Hydrology']**2)**0.5
test['Distance_to_Hydrology'] = (test['Horizontal_Distance_To_Hydrology']**2 + test['Vertical_Distance_To_Hydrology']**2)**0.5
# end-added-column

# python-added-column
# Feature name and description: Mean_Distance_to_Features
# Usefulness: This feature averages the distances to hydrology, fire points, and roadways, 
# which could provide a general measure of remoteness.
features = ['Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways']
train['Mean_Distance_to_Features'] = train[features].mean(axis=1)
test['Mean_Distance_to_Features'] = test[features].mean(axis=1)
# end-added-column

# python-dropping-columns
# Explanation why the column XX is dropped
# The columns 'Horizontal_Distance_To_Hydrology' and 'Vertical_Distance_To_Hydrology' are dropped 
# because they are now represented in the new 'Distance_to_Hydrology' feature.
train.drop(columns=['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'], inplace=True)
test.drop(columns=['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation: RandomForestClassifier is a robust and widely-used method for multiclass classification. 
# It can handle a large number of features and is less prone to overfitting.
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation