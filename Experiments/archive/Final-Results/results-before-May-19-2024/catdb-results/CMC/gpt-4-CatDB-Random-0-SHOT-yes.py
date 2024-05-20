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
# As per the dataset description, there are no missing values. So, we don't need to handle missing values.
# But we need to handle categorical variables and scale numerical variables.

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Wifes_age', 'Number_of_children_ever_born']),
        ('cat', OneHotEncoder(), ['Husbands_occupation', 'Husbands_education', 'Standard-of-living_index', 'Wifes_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure'])])
# ```end

# ```python
# Define the classifier
# We are using RandomForestClassifier as it is a good choice for multiclass classification problems.
# We are passing max_leaf_nodes=500 as parameter to avoid overfitting.
classifier = RandomForestClassifier(max_leaf_nodes=500, random_state=0)
# ```end

# ```python
# Combine preprocessor and classifier into a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])
# ```end

# ```python
# Train the model
X_train = train_data.drop('Contraceptive_method_used', axis=1)
y_train = train_data['Contraceptive_method_used']
pipeline.fit(X_train, y_train)
# ```end

# ```python
# Evaluate the model
X_test = test_data.drop('Contraceptive_method_used', axis=1)
y_test = test_data['Contraceptive_method_used']

# Predict the classes and probability for test data
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Calculate accuracy and log loss
Train_Accuracy = accuracy_score(y_train, pipeline.predict(X_train))
Test_Accuracy = accuracy_score(y_test, y_pred)
Train_Log_loss = log_loss(y_train, pipeline.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, y_pred_proba)

# Print the results
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end