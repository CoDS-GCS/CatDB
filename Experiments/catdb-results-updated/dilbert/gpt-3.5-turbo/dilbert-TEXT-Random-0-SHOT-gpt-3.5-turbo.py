# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/dilbert/dilbert_train.csv')
test_data = pd.read_csv('data/dilbert/dilbert_test.csv')
# end-load-dataset

# Added Columns
# python-added-column
# Feature: V732 + V1272
# Usefulness: This feature represents the sum of V732 and V1272, which captures the combined effect of these two variables on the target class.
train_data['V732_V1272_sum'] = train_data['V732'] + train_data['V1272']
test_data['V732_V1272_sum'] = test_data['V732'] + test_data['V1272']

# Feature: V575 * V1712
# Usefulness: This feature represents the product of V575 and V1712, which captures the interaction between these two variables on the target class.
train_data['V575_V1712_product'] = train_data['V575'] * train_data['V1712']
test_data['V575_V1712_product'] = test_data['V575'] * test_data['V1712']

# Feature: V352 - V21
# Usefulness: This feature represents the difference between V352 and V21, which captures the contrast between these two variables on the target class.
train_data['V352_V21_difference'] = train_data['V352'] - train_data['V21']
test_data['V352_V21_difference'] = test_data['V352'] - test_data['V21']

# Feature: V357 / V1962
# Usefulness: This feature represents the ratio of V357 to V1962, which captures the relative importance of these two variables on the target class.
train_data['V357_V1962_ratio'] = train_data['V357'] / train_data['V1962']
test_data['V357_V1962_ratio'] = test_data['V357'] / test_data['V1962']

# Drop redundant columns
train_data.drop(columns=['V732', 'V1272', 'V575', 'V1712', 'V352', 'V21', 'V357', 'V1962'], inplace=True)
test_data.drop(columns=['V732', 'V1272', 'V575', 'V1712', 'V352', 'V21', 'V357', 'V1962'], inplace=True)
# end-added-column

# python-training-technique
# Use a multiclass classification technique
# Explanation: Logistic Regression is selected as it is a commonly used algorithm for multiclass classification problems and performs well in many scenarios.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split the dataset into train and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_final, y_train_final)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

# Standardize the test features
X_test_scaled = scaler.transform(X_test)

# Predict the class labels for the test set
y_pred = model.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation