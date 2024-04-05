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
# Load the datasets
train_data = pd.read_csv('../../../data/albert/albert_train.csv')
test_data = pd.read_csv('../../../data/albert/albert_test.csv')
# ```end

# ```python
# Define the columns for preprocessing
num_cols = ['V17','V37','V44','V16','V55','V5','V34','V58','V39','V25','V29','V57']
cat_cols = ['V47','V70','V1','V6','V9','V42','V8','V64','V67','V38','V13','V41','V75','V52','V11','V19','V71','V43','V51','V33','V72','V53','V32','V35','V3','V12','V69','V40','V10','V59','V4','V7','V50']
scale_cols = ['V74','V47','V61','V70','V1','V17','V37','V36','V77','V76','V6','V60','V44','V78','V16','V9','V42','V55','V8','V31','V5','V64','V18','V20','V63','V49','V67','V34','V15','V56','V38','V23','V58','V73','V66','V54','V39','V13','V41','V27','V75','V2','V52','V46','V11','V25','V19','V71','V43','V29','V45','V21','V51','V33','V72','V26','V65','V53','V24','V28','V32','V35','V48','V14','V62','V22','V3','V12','V68','V30','V69','V57','V40','V10','V59','V4','V7','V50']
encode_cols = ['V74','V47','V70','V1','V36','V76','V6','V60','V9','V42','V8','V64','V18','V63','V49','V67','V15','V38','V73','V54','V13','V41','V27','V75','V52','V11','V19','V71','V43','V45','V21','V51','V33','V72','V65','V53','V32','V35','V48','V14','V22','V3','V12','V68','V30','V69','V40','V10','V59','V4','V7','V50']
# ```end

# ```python
# Define the preprocessing steps
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
# ```end

# ```python
# Split the features and target variable
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Predict the train and test data
train_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on train and test dataset
Train_Accuracy = accuracy_score(y_train, train_preds)
Test_Accuracy = accuracy_score(y_test, test_preds)
Train_F1_score = f1_score(y_train, train_preds)
Test_F1_score = f1_score(y_test, test_preds)

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end