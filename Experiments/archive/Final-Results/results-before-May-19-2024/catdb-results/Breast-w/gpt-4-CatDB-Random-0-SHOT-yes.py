# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Breast-w/Breast-w_train.csv')
test_data = pd.read_csv('../../../data/Breast-w/Breast-w_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Impute missing values in 'Bare_Nuclei' column with median
imputer = SimpleImputer(strategy='median')
train_data['Bare_Nuclei'] = imputer.fit_transform(train_data['Bare_Nuclei'].values.reshape(-1,1))
test_data['Bare_Nuclei'] = imputer.transform(test_data['Bare_Nuclei'].values.reshape(-1,1))
# ```end

# ```python
# Perform feature processing
# Scale numerical columns
scaler = MinMaxScaler()
numerical_cols = ['Marginal_Adhesion', 'Normal_Nucleoli', 'Bare_Nuclei', 'Clump_Thickness', 'Single_Epi_Cell_Size', 'Cell_Shape_Uniformity', 'Bland_Chromatin', 'Mitoses', 'Cell_Size_Uniformity']
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
# ```end

# ```python
# Select the appropriate features and target variables for the question
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has the advantage of working well with default parameters.
clf = RandomForestClassifier(max_leaf_nodes=500)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end