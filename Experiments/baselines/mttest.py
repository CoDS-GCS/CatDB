# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the training and test datasets
train_data = pd.read_csv('../../../data/PC1/PC1_train.csv')
test_data = pd.read_csv('../../../data/PC1/PC1_test.csv')

# Perform data cleaning and preprocessing
# Fill missing values with median for numerical columns and most frequent value for categorical columns
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Define numerical columns based on the schema
num_cols = ['lOComment','lOCode','lOBlank','locCodeAndComment','L','iv(G)','ev(g)','branchCount','uniq_Op','uniq_Opnd','V','I','T','E','total_Op','v(g)','D','loc','N','total_Opnd','B']

# Separate features and target before applying transformations
X_train = train_data.drop(columns=['defects'])
y_train = train_data['defects']
X_test = test_data.drop(columns=['defects'])
y_test = test_data['defects']

# Apply imputers
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Perform feature processing
# Scale numerical columns and encode categorical columns
scaler = StandardScaler()

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, num_cols)])

# Preprocess the datasets
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it can handle both numerical and categorical features, and it is robust to outliers.
clf = RandomForestClassifier(max_leaf_nodes=500)

# Fit the model
clf.fit(X_train, y_train)

# Report evaluation based on train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy: {Train_Accuracy}")
print(f"Train_F1_score: {Train_F1_score}")
print(f"Test_Accuracy: {Test_Accuracy}")
print(f"Test_F1_score: {Test_F1_score}")