# Import all required packages
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the datasets
train_data = pd.read_csv('../../../data/drug-directory/drug-directory_train.csv')
test_data = pd.read_csv('../../../data/drug-directory/drug-directory_test.csv')

# Drop the unnecessary columns
drop_cols = ['ENDMARKETINGDATE','PROPRIETARYNAMESUFFIX','NDC_EXCLUDE_FLAG','DEASCHEDULE']
train_data.drop(columns=drop_cols, inplace=True)
test_data.drop(columns=drop_cols, inplace=True)

# Define preprocessing for numerical columns
num_cols = ['STARTMARKETINGDATE','LISTING_RECORD_CERTIFIED_THROUGH']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

# Define preprocessing for categorical columns
cat_cols = ['LISTING_RECORD_CERTIFIED_THROUGH','MARKETINGCATEGORYNAME','ROUTENAME','DOSAGEFORMNAME','ACTIVE_NUMERATOR_STRENGTH']
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Fit the model
pipeline.fit(train_data.drop('PRODUCTTYPENAME', axis=1), train_data['PRODUCTTYPENAME'])

# Predict the training set
train_preds = pipeline.predict(train_data.drop('PRODUCTTYPENAME', axis=1))
train_proba = pipeline.predict_proba(train_data.drop('PRODUCTTYPENAME', axis=1))

# Predict the test set
test_preds = pipeline.predict(test_data.drop('PRODUCTTYPENAME', axis=1))
test_proba = pipeline.predict_proba(test_data.drop('PRODUCTTYPENAME', axis=1))

# Calculate the model accuracy and log loss
Train_Accuracy = accuracy_score(train_data['PRODUCTTYPENAME'], train_preds)
Test_Accuracy = accuracy_score(test_data['PRODUCTTYPENAME'], test_preds)
Train_Log_loss = log_loss(train_data['PRODUCTTYPENAME'], train_proba)
Test_Log_loss = log_loss(test_data['PRODUCTTYPENAME'], test_proba)

# Print the results
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")