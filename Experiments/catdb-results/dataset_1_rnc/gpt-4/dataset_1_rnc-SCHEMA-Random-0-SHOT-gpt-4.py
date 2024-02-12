# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_test.csv')

# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]

# Feature name and description: c_5_c_14_ratio
# Usefulness: This feature represents the ratio of c_5 to c_14. This could be useful if there is a meaningful relationship between these two features that could help predict 'c_24'.
train_data['c_5_c_14_ratio'] = train_data['c_5'] / train_data['c_14']
test_data['c_5_c_14_ratio'] = test_data['c_5'] / test_data['c_14']

# Explanation why the column c_2 is dropped: Assuming that c_2 is not relevant to the prediction of 'c_24' based on domain knowledge.
train_data.drop(columns=['c_2'], inplace=True)
test_data.drop(columns=['c_2'], inplace=True)

# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a versatile and widely used machine learning algorithm that can handle both categorical and numerical features. It also has the advantage of being able to handle missing values and outliers.
X_train = train_data.drop('c_24', axis=1)
y_train = train_data['c_24']

# Ensure that the test data has the same columns as the training data
X_test = test_data[X_train.columns]
y_test = test_data['c_24']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")