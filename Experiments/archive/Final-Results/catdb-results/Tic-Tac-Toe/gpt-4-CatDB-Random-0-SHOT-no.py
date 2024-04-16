# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv')
test_data = pd.read_csv('../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# As per the given schema, there are no missing values and all the columns are of integer type. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# Select the appropriate features and target variables for the question
# As per the given schema, all the columns except 'Class' are features and 'Class' is the target variable.

features = ['top-middle-square', 'middle-right-square', 'middle-left-square', 'bottom-right-square', 'bottom-left-square', 'middle-middle-square', 'bottom-middle-square', 'top-left-square', 'top-right-square']
target = ['Class']

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]
# ```end

# ```python
# Perform feature scaling and encoding
# As per the given schema, all the features are of integer type and need to be scaled. Also, they need to be one-hot encoded.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(), features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(max_leaf_nodes=500))])
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train.values.ravel())
# ```end

# ```python
# Predict the target for training data
train_preds = clf.predict(X_train)

# Calculate the model accuracy and f1 score for training data
Train_Accuracy = accuracy_score(y_train, train_preds)
Train_F1_score = f1_score(y_train, train_preds)

# Print the train accuracy and f1 score results
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
# ```end

# ```python
# Predict the target for test data
test_preds = clf.predict(X_test)

# Calculate the model accuracy and f1 score for test data
Test_Accuracy = accuracy_score(y_test, test_preds)
Test_F1_score = f1_score(y_test, test_preds)

# Print the test accuracy and f1 score results
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}")
# ```end