# Import all required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

# Load the training and test datasets
train_data = pd.read_csv("../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../data/Jungle-Chess/Jungle-Chess_test.csv")

# Perform feature processing
# Select the appropriate features and target variables for the question
features = ['black_piece0_file', 'white_piece0_rank', 'white_piece0_strength', 'black_piece0_rank', 'black_piece0_strength', 'white_piece0_file']
target = ['class']

# Define the preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, features)])

# Define the model
model = RandomForestClassifier(max_leaf_nodes=500)

# Combine preprocessing and modeling steps
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Train the model
clf.fit(train_data[features], np.ravel(train_data[target]))

# Predict the target for the training data
train_predictions = clf.predict(train_data[features])

# Predict the target for the test data
test_predictions = clf.predict(test_data[features])

# Report evaluation based on train and test dataset
Train_Accuracy = accuracy_score(train_data[target], train_predictions)
Test_Accuracy = accuracy_score(test_data[target], test_predictions)

# Binarize the output
lb = LabelBinarizer()
train_data_binarized = lb.fit_transform(train_data[target])
train_predictions_binarized = lb.transform(train_predictions)
test_data_binarized = lb.transform(test_data[target])
test_predictions_binarized = lb.transform(test_predictions)

Train_Log_loss = log_loss(train_data_binarized, train_predictions_binarized)
Test_Log_loss = log_loss(test_data_binarized, test_predictions_binarized)

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")