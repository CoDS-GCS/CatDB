# python-import
# Import all required packages
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/kc1/kc1_train.csv")
test = pd.read_csv("data/kc1/kc1_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: total_Op_per_loc
# Usefulness: This feature gives us an idea about the density of operations per line of code. It might be useful to detect defects as more operations per line could potentially lead to more complex and error-prone code.
train['total_Op_per_loc'] = train['total_Op'] / train['loc']
test['total_Op_per_loc'] = test['total_Op'] / test['loc']
# end-added-column

# python-added-column
# Feature name and description: total_Opnd_per_loc
# Usefulness: This feature gives us an insight about the density of operands per line of code. It might be useful to detect defects as more operands per line could lead to more complex and error-prone code.
train['total_Opnd_per_loc'] = train['total_Opnd'] / train['loc']
test['total_Opnd_per_loc'] = test['total_Opnd'] / test['loc']
# end-added-column

# python-dropping-columns
# Explanation why the column 'loc' is dropped: The 'loc' (lines of code) column is dropped because we have already used it to create new features that capture more information. Keeping 'loc' might lead to redundancy and multicollinearity issues.
train.drop(columns=['loc'], inplace=True)
test.drop(columns=['loc'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: Random Forest Classifier is chosen because it works well with a mix of numerical and categorical features. It also handles overfitting by averaging the result of different decision trees.

# Split the training data into features and target variable
X_train = train.drop('defects', axis=1)
y_train = train['defects']

# Define preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Append classifier to preprocessing pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Fit the model
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
X_test = test.drop('defects', axis=1)
y_test = test['defects']

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation