# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")
test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")
# ```end

# ```python
# Handle missing values
# As the dataset contains missing values, we will use SimpleImputer to fill missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')

# Fit on the training data
imputer.fit(train_data)

# Transform both training and test data
train_data = pd.DataFrame(imputer.transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
# ```end

# ```python
# Drop columns with low variance as they do not contribute much to the model
# Here we assume that columns with a standard deviation less than 0.1 are considered low variance
low_variance_cols = [col for col in train_data.columns if train_data[col].std() < 0.1]

# Drop these columns from the training and test datasets
train_data.drop(columns=low_variance_cols, inplace=True)
test_data.drop(columns=low_variance_cols, inplace=True)
# ```end

# ```python
# Split the data into features and target variable
X_train = train_data.drop(columns=['c_1'])
y_train = train_data['c_1']

X_test = test_data.drop(columns=['c_1'])
y_test = test_data['c_1']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets
# It can handle binary features, categorical features, and numerical features, and provides a pretty good indicator of the importance it assigns to your features
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Fit the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate F1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy and F1 score
print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")
# ```end