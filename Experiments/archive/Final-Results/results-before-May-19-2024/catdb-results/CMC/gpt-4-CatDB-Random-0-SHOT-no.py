# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/CMC/CMC_train.csv')
test_data = pd.read_csv('../../../data/CMC/CMC_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# As per the given schema, there are no missing values and all the columns are of correct data type. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# Select numerical columns for scaling
num_cols = ['Husbands_occupation', 'Number_of_children_ever_born', 'Wifes_age', 'Husbands_education', 'Standard-of-living_index', 'Wifes_education']

# Select categorical columns for one-hot encoding
cat_cols = ['Husbands_occupation', 'Husbands_education', 'Standard-of-living_index', 'Wifes_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)])

# Define pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(max_leaf_nodes=500))])
# ```end

# ```python
# Select the appropriate features and target variables
X_train = train_data.drop('Contraceptive_method_used', axis=1)
y_train = train_data['Contraceptive_method_used']
X_test = test_data.drop('Contraceptive_method_used', axis=1)
y_test = test_data['Contraceptive_method_used']
# ```end

# ```python
# Train the model
pipeline.fit(X_train, y_train)
# ```end

# ```python
# Predict on train and test data
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Calculate probabilities for log loss
y_train_prob = pipeline.predict_proba(X_train)
y_test_prob = pipeline.predict_proba(X_test)
# ```end

# ```python
# Calculate accuracy and log loss
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_Log_loss = log_loss(y_train, y_train_prob)
Test_Log_loss = log_loss(y_test, y_test_prob)

# Print the results
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end