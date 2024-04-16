# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error

# Load the training and test datasets
train_data = pd.read_csv('../../../data/Dionis/Dionis_train.csv')
test_data = pd.read_csv('../../../data/Dionis/Dionis_test.csv')

# Drop the unnecessary columns
drop_cols = ['V27','V37','V54','V35','V33','V14']
train_data.drop(columns=drop_cols, inplace=True)
test_data.drop(columns=drop_cols, inplace=True)

# Define the columns to be scaled and encoded
scale_cols = ['V15','V21','V13','V26','V6','V30','V10','V32','V11','V38','V8','V40','V28','V34','V2','V22','V9','V59','V17','V48','V50','V19','V57','V58','V45','V60','V24','V53','V1','V55','V41','V31','V44','V56','V20','V39','V23','V52','V29','V5','V46','V42','V16','V43','V18','V7','V3','V4','V36','V12','V51','V49','V25','V47']
encode_cols = ['V21','V26','V6','V10','V45','V60','V53','V3','V49','V25']

# Remove the overlap between 'scale_cols' and 'encode_cols'
scale_cols = list(set(scale_cols) - set(encode_cols))

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), scale_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), encode_cols)])

# Define the model
model = RandomForestClassifier(max_leaf_nodes=500)

# Combine preprocessing and modeling steps into a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split the data into features and target variable
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the training and test data
train_preds = pipeline.predict(X_train)
test_preds = pipeline.predict(X_test)

# Calculate the model R-Squared and Root Mean Squared Error
Train_R_Squared = r2_score(y_train, train_preds)
Test_R_Squared = r2_score(y_test, test_preds)
Train_RMSE = np.sqrt(mean_squared_error(y_train, train_preds))
Test_RMSE = np.sqrt(mean_squared_error(y_test, test_preds))

# Print the train and test results
print(f"Train_R_Squared:{Train_R_Squared}")   
print(f"Train_RMSE:{Train_RMSE}") 
print(f"Test_R_Squared:{Test_R_Squared}")   
print(f"Test_RMSE:{Test_RMSE}") 
# ```