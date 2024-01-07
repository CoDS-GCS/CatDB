# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/robert/robert_train.csv')
test_data = pd.read_csv('data/robert/robert_test.csv')
# end-load-dataset

# python-added-column
# (Feature name and description)
# Usefulness: This feature calculates the sum of V6111 and V1150 and adds useful real-world knowledge by capturing the combined effect of these two attributes on the target class.
train_data['V6111_V1150_sum'] = train_data['V6111'] + train_data['V1150']
test_data['V6111_V1150_sum'] = test_data['V6111'] + test_data['V1150']
# end-added-column

# python-added-column
# (Feature name and description)
# Usefulness: This feature calculates the product of V6111 and V1150 and adds useful real-world knowledge by capturing the interaction between these two attributes on the target class.
train_data['V6111_V1150_product'] = train_data['V6111'] * train_data['V1150']
test_data['V6111_V1150_product'] = test_data['V6111'] * test_data['V1150']
# end-added-column

# python-dropping-columns
# Explanation why the column V1150 is dropped
train_data.drop(columns=['V1150'], inplace=True)
test_data.drop(columns=['V1150'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a Random Forest classifier for multiclass classification
# Explanation: Random Forest is a powerful ensemble method that can handle multiclass classification problems effectively.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = RandomForestClassifier()
classifier.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
X_test_scaled = scaler.transform(X_test)
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation