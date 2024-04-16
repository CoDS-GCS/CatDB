# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Heart-Statlog/Heart-Statlog_train.csv')
test_data = pd.read_csv('../../../data/Heart-Statlog/Heart-Statlog_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# As per the given schema and data profiling info, there is no missing data or outliers. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['oldpeak', 'maximum_heart_rate_achieved', 'resting_blood_pressure', 'serum_cholestoral', 'age', 'chest']),
        ('cat', OneHotEncoder(), ['slope', 'number_of_major_vessels', 'resting_electrocardiographic_results', 'thal', 'sex', 'exercise_induced_angina', 'fasting_blood_sugar'])
    ])
# ```end

# ```python
# Select the appropriate features and target variables for the question
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
# ```end

# ```python
# Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection)
# As per the given schema and data profiling info, all the columns seem to be important and there is no redundant column. So, no column is dropped.
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has the feature of handling overfitting.
# If the algorithm is RandomForestClassifier then pass max_leaf_nodes=500 as parameter.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(max_leaf_nodes=500))])
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Predict the target variable for train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on train and test dataset
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_F1_score = f1_score(y_train, y_train_pred, average='weighted')
Test_F1_score = f1_score(y_test, y_test_pred, average='weighted')

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end