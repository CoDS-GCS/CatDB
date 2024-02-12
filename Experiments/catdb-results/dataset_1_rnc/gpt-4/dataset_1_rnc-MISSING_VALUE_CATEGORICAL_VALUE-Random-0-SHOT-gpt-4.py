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
train_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_train.csv")
test_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# c_27 is a static column with only one distinct value, so it can be dropped
train_data.drop(columns=['c_27'], inplace=True)
test_data.drop(columns=['c_27'], inplace=True)
# ```end-dropping-columns

# ```python
# Impute missing values
# For simplicity, we will use mean imputation for numerical columns and most frequent imputation for categorical columns
num_cols = ['c_5', 'c_14', 'c_8', 'c_15', 'c_17', 'c_10', 'c_7', 'c_18', 'c_11', 'c_21', 'c_6', 'c_12', 'c_9', 'c_13', 'c_4', 'c_19', 'c_16', 'c_20', 'c_22']
cat_cols = ['c_2', 'c_1', 'c_28', 'c_24']

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])
test_data[num_cols] = num_imputer.transform(test_data[num_cols])

train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
test_data[cat_cols] = cat_imputer.transform(test_data[cat_cols])
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets
# It can handle both numerical and categorical data, and it also has built-in feature importance estimation
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Separate features and target
X_train = train_data.drop(columns=['c_24'])
y_train = train_data['c_24']

X_test = test_data.drop(columns=['c_24'])
y_test = test_data['c_24']

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
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
# ```end