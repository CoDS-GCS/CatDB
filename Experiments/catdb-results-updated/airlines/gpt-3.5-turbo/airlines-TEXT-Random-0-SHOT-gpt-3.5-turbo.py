# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/airlines/airlines_train.csv')
test_data = pd.read_csv('data/airlines/airlines_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'AirportTo_AirportFrom' to capture the combination of 'AirportTo' and 'AirportFrom'
train_data['AirportTo_AirportFrom'] = train_data['AirportTo'] + '_' + train_data['AirportFrom']
test_data['AirportTo_AirportFrom'] = test_data['AirportTo'] + '_' + test_data['AirportFrom']

# Add new column 'Flight_Length' to capture the product of 'Flight' and 'Length'
train_data['Flight_Length'] = train_data['Flight'] * train_data['Length']
test_data['Flight_Length'] = test_data['Flight'] * test_data['Length']

# Add new column 'Time_DayOfWeek' to capture the sum of 'Time' and 'DayOfWeek'
train_data['Time_DayOfWeek'] = train_data['Time'] + train_data['DayOfWeek']
test_data['Time_DayOfWeek'] = test_data['Time'] + test_data['DayOfWeek']
# end-added-column

# python-dropping-columns
# Drop 'AirportTo' and 'AirportFrom' columns as they are captured in 'AirportTo_AirportFrom'
train_data.drop(columns=['AirportTo', 'AirportFrom'], inplace=True)
test_data.drop(columns=['AirportTo', 'AirportFrom'], inplace=True)
# end-dropping-columns

for col in ['Airline', 'AirportTo_AirportFrom']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])

# python-training-technique
# Use Random Forest Classifier for binary classification
clf = RandomForestClassifier()

# Split the data into features and target variable
X_train = train_data.drop(columns=['Delay'])
y_train = train_data['Delay']

# Train the classifier
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate on test dataset
X_test = test_data.drop(columns=['Delay'])
y_test = test_data['Delay']
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 