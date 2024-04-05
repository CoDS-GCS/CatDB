# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/jungle_chess_2pcs_raw_endgame_complete/jungle_chess_2pcs_raw_endgame_complete_train.csv')
test_data = pd.read_csv('../../../data/jungle_chess_2pcs_raw_endgame_complete/jungle_chess_2pcs_raw_endgame_complete_test.csv')
# ```end

# ```python
# Define the columns to be scaled and encoded
scale_cols = ['black_piece0_file', 'white_piece0_rank', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_file']
encode_cols = ['black_piece0_file', 'white_piece0_rank', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_file']
# ```end

# ```python
# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), scale_cols),
        ('cat', OneHotEncoder(), encode_cols)])
# ```end

# ```python
# Define the classifier
classifier = RandomForestClassifier(n_jobs=-1, random_state=42)
# ```end

# ```python
# Combine preprocessing and classifier into a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])
# ```end

# ```python
# Fit the pipeline on the training data
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
pipeline.fit(X_train, y_train)
# ```end

# ```python
# Predict on the training data and calculate accuracy and log loss
y_train_pred = pipeline.predict(X_train)
y_train_prob = pipeline.predict_proba(X_train)
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_prob)
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
# ```end

# ```python
# Predict on the test data and calculate accuracy and log loss
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_prob)
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end