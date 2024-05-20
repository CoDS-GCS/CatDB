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
train_data = pd.read_csv('../../../data/Higgs/Higgs_train.csv')
test_data = pd.read_csv('../../../data/Higgs/Higgs_test.csv')
# ```end

# ```python
# Define preprocessing steps
numeric_features = ['m_bb', 'jet 2 eta', 'jet 4 pt', 'jet 4 phi', 'jet 3 b-tag', 'jet 3 phi', 'jet 1 b-tag', 'jet 3 eta', 'missing energy magnitude', 'jet 3 pt', 'lepton eta', 'm_wbb', 'jet 1 phi', 'jet 4 eta', 'lepton pT', 'jet 1 pt', 'm_jj', 'm_wwbb', 'missing energy phi', 'm_jlv', 'jet 2 b-tag', 'm_jjj', 'jet 4 b-tag', 'm_lv', 'lepton phi', 'jet 2 phi', 'jet 2 pt', 'jet 1 eta']
categorical_features = ['jet 2 eta', 'jet 4 pt', 'jet 4 phi', 'jet 3 b-tag', 'jet 3 phi', 'jet 1 b-tag', 'jet 3 eta', 'jet 3 pt', 'lepton eta', 'jet 1 phi', 'jet 4 eta', 'lepton pT', 'jet 1 pt', 'jet 2 b-tag', 'jet 4 b-tag', 'lepton phi', 'jet 2 phi', 'jet 2 pt', 'jet 1 eta']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# ```end

# ```python
# Define the classifier
# RandomForestClassifier is chosen because it is a versatile and widely used algorithm that can handle both numerical and categorical data, and it also provides a good indicator of the importance it assigns to the features.
clf = RandomForestClassifier(max_leaf_nodes=500)

# Append classifier to preprocessing pipeline
# Now we have a full prediction pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', clf)])
# ```end

# ```python
# Split the data into features and target variable
X_train = train_data.drop('Target', axis=1)
y_train = train_data['Target']

X_test = test_data.drop('Target', axis=1)
y_test = test_data['Target']
# ```end

# ```python
# Train the model
model.fit(X_train, y_train)
# ```end

# ```python
# Make predictions on the train and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# ```end

# ```python
# Report evaluation based on train and test dataset
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end