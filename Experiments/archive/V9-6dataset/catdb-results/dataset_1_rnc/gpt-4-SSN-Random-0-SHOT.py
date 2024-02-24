# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_test.csv')

# Perform data cleaning and preprocessing
scaler = StandardScaler()
imputer = SimpleImputer()

train_data_scaled = scaler.fit_transform(imputer.fit_transform(train_data.drop(columns=['c_24'])))
test_data_scaled = scaler.transform(imputer.transform(test_data.drop(columns=['c_24'])))

# Select the appropriate features and target variables for the question
X_train = train_data_scaled
y_train = train_data['c_24']

X_test = test_data_scaled
y_test = test_data['c_24']

# Perform drops columns
X_train = pd.DataFrame(X_train).drop(columns=[26])  # 'c_27' is the 27th column, so its index will be 26
X_test = pd.DataFrame(X_test).drop(columns=[26])

# Choose the suitable machine learning algorithm or technique (classifier)
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy: {Accuracy}")

# Print the f1 score result
print(f"F1_score: {F1_score}")