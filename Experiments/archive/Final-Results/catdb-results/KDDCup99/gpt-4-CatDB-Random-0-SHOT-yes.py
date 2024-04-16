# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/KDDCup99/KDDCup99_train.csv')
test_data = pd.read_csv('../../../data/KDDCup99/KDDCup99_test.csv')
# ```end

# ```python
# Drop the 'lnum_outbound_cmds' column as it is not useful for the analysis
train_data.drop(columns=['lnum_outbound_cmds'], inplace=True)
test_data.drop(columns=['lnum_outbound_cmds'], inplace=True)
# ```end

# ```python
# Define the columns to be scaled and encoded
scale_cols = ['lnum_shells','num_failed_logins','dst_host_count','src_bytes','dst_bytes','lnum_root','duration','urgent','count','dst_host_srv_count','lnum_access_files','lnum_compromised','srv_count','lnum_file_creations','hot','wrong_fragment','lsu_attempted','dst_host_same_srv_rate','srv_rerror_rate','dst_host_srv_serror_rate','serror_rate','dst_host_diff_srv_rate','dst_host_srv_diff_host_rate','rerror_rate','dst_host_serror_rate','srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_srv_rerror_rate','diff_srv_rate','dst_host_rerror_rate','same_srv_rate','srv_serror_rate']
encode_cols = ['service','flag','protocol_type','lroot_shell','is_host_login','is_guest_login','land','logged_in']
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
clf = RandomForestClassifier(max_leaf_nodes=500)
# ```end

# ```python
# Combine preprocessing and classifier into a pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', clf)])
# ```end

# ```python
# Define the target variable
y_train = train_data['label']
y_test = test_data['label']

# Drop the target variable from the training and test datasets
X_train = train_data.drop('label', axis=1)
X_test = test_data.drop('label', axis=1)
# ```end

# ```python
# Train the model using the pipeline
with parallel_backend('threading', n_jobs=-1):
    pipe.fit(X_train, y_train)
# ```end

# ```python
# Predict the target variable for the training and test datasets
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

# Calculate the probabilities for the training and test datasets
y_train_prob = pipe.predict_proba(X_train)
y_test_prob = pipe.predict_proba(X_test)
# ```end

# ```python
# Calculate the accuracy for the training and test datasets
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

# Calculate the log loss for the training and test datasets
Train_Log_loss = log_loss(y_train, y_train_prob)
Test_Log_loss = log_loss(y_test, y_test_prob)
# ```end

# ```python
# Print the accuracy and log loss for the training and test datasets
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end