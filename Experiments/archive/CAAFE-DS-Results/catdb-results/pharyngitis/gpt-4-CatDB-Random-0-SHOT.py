# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/pharyngitis/pharyngitis_train.csv')
test_data = pd.read_csv('../../../data/pharyngitis/pharyngitis_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Impute missing values for numerical columns
num_imputer = SimpleImputer(strategy='mean')
train_data[['swollenadp', 'temperature']] = num_imputer.fit_transform(train_data[['swollenadp', 'temperature']])
test_data[['swollenadp', 'temperature']] = num_imputer.transform(test_data[['swollenadp', 'temperature']])

# Impute missing values for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = ['exudate', 'tonsillarswelling', 'tender', 'cough', 'pain', 'scarlet', 'abdopain', 'sudden', 'conjunctivitis', 'rhinorrhea', 'nauseavomit', 'diarrhea', 'petechiae', 'erythema', 'headache']
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
test_data[cat_cols] = cat_imputer.transform(test_data[cat_cols])
# ```end

# ```python
# Perform feature processing
# Scale numerical columns
scaler = MinMaxScaler()
train_data[['exudate', 'number', 'tonsillarswelling', 'tender', 'cough', 'pain', 'scarlet', 'abdopain', 'sudden', 'conjunctivitis', 'rhinorrhea', 'nauseavomit', 'diarrhea', 'swollenadp', 'petechiae', 'erythema', 'headache', 'age_y', 'temperature']] = scaler.fit_transform(train_data[['exudate', 'number', 'tonsillarswelling', 'tender', 'cough', 'pain', 'scarlet', 'abdopain', 'sudden', 'conjunctivitis', 'rhinorrhea', 'nauseavomit', 'diarrhea', 'swollenadp', 'petechiae', 'erythema', 'headache', 'age_y', 'temperature']])
test_data[['exudate', 'number', 'tonsillarswelling', 'tender', 'cough', 'pain', 'scarlet', 'abdopain', 'sudden', 'conjunctivitis', 'rhinorrhea', 'nauseavomit', 'diarrhea', 'swollenadp', 'petechiae', 'erythema', 'headache', 'age_y', 'temperature']] = scaler.transform(test_data[['exudate', 'number', 'tonsillarswelling', 'tender', 'cough', 'pain', 'scarlet', 'abdopain', 'sudden', 'conjunctivitis', 'rhinorrhea', 'nauseavomit', 'diarrhea', 'swollenadp', 'petechiae', 'erythema', 'headache', 'age_y', 'temperature']])

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first')
train_data = pd.get_dummies(train_data, columns=cat_cols, drop_first=True)
test_data = pd.get_dummies(test_data, columns=cat_cols, drop_first=True)
# ```end

# ```python
# Select the appropriate features and target variables
X_train = train_data.drop('radt', axis=1)
y_train = train_data['radt']
X_test = test_data.drop('radt', axis=1)
y_test = test_data['radt']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen due to its ability to handle both numerical and categorical data, and its robustness to overfitting
clf = RandomForestClassifier(n_estimators=100, random_state=42)
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