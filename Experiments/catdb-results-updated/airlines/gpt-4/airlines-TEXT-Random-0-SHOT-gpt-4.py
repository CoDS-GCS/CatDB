# python-import
# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/airlines/airlines_train.csv")
test = pd.read_csv("data/airlines/airlines_test.csv")
# end-load-dataset

# python-added-column
# Feature name: Route
# Usefulness: The combination of 'AirportFrom' and 'AirportTo' can provide useful information about specific routes that might have more delays.
train['Route'] = train['AirportFrom'] + "_" + train['AirportTo']
test['Route'] = test['AirportFrom'] + "_" + test['AirportTo']
# end-added-column

# python-added-column
# Feature name: Flight_Time
# Usefulness: The product of 'Flight' and 'Time' can provide useful information about the total flight time which might affect the delay.
train['Flight_Time'] = train['Flight'] * train['Time']
test['Flight_Time'] = test['Flight'] * test['Time']
# end-added-column

# python-dropping-columns
# Explanation why the column 'Flight' and 'Time' are dropped: These columns are now represented in the 'Flight_Time' feature and are no longer needed separately.
train.drop(columns=['Flight', 'Time'], inplace=True)
test.drop(columns=['Flight', 'Time'], inplace=True)
# end-dropping-columns

# python-other
# Explanation why this line of code is required: The categorical variables need to be encoded for the model to process them.
for col in ['AirportTo', 'AirportFrom', 'Airline', 'Route']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])
# end-other

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and widely used model for classification problems. It also handles a mix of categorical and numerical features well.
X_train = train.drop('Delay', axis=1)
y_train = train['Delay']
X_test = test.drop('Delay', axis=1)
y_test = test['Delay']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation