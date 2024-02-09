# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/horsecolic/horsecolic_train.csv")
test_data = pd.read_csv("data/horsecolic/horsecolic_test.csv")
# ```end

# ```python
# Drop the 'LesionSubtype' column as it has only one distinct value and hence does not contribute to the model
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)
# ```end-dropping-columns

# ```python
# Add a new feature 'AbdominalIssue' which is a combination of 'AbdominalDistension' and 'Abdomen'
# Usefulness: This feature combines the information about abdominal distension and the condition of the abdomen to give a more comprehensive view of the abdominal health of the horse. This can be useful in predicting whether a surgical lesion is present or not.
train_data['AbdominalIssue'] = train_data['AbdominalDistension'] + train_data['Abdomen']
test_data['AbdominalIssue'] = test_data['AbdominalDistension'] + test_data['Abdomen']
# ```end

# ```python
# Define the pipeline
# The pipeline first fills missing values with the median of the column, then scales the features to have zero mean and unit variance, and finally applies a Random Forest Classifier.

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
# ```end

# ```python
# Separate the features and the target variable
X_train = train_data.drop(columns=['SurgicalLesion'])
y_train = train_data['SurgicalLesion']

X_test = test_data.drop(columns=['SurgicalLesion'])
y_test = test_data['SurgicalLesion']
# ```end

# ```python
# Train the model using the pipeline
pipeline.fit(X_train, y_train)
# ```end

# ```python
# Predict the target variable for the test set
y_pred = pipeline.predict(X_test)
# ```end

# ```python
# Calculate the accuracy and f1 score of the model
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

# Print the accuracy and f1 score
print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")
# ```end