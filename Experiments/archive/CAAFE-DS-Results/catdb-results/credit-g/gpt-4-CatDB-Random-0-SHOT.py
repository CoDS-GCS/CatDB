# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/credit-g/credit-g_train.csv')
test_data = pd.read_csv('../../../data/credit-g/credit-g_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# As per the given schema and data profiling info, there is no missing data. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['num_dependents', 'credit_amount', 'residence_since', 'age', 'installment_commitment', 'duration', 'existing_credits']),
        ('cat', OneHotEncoder(), ['num_dependents', 'residence_since', 'installment_commitment', 'existing_credits', 'credit_history', 'job', 'housing', 'own_telephone', 'personal_status', 'property_magnitude', 'foreign_worker', 'other_parties', 'other_payment_plans', 'savings_status', 'checking_status', 'employment'])
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
# Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier
# As per the given schema and data profiling info, there is no redundant column. So, no column is dropped.
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile and widely used algorithm that can handle both numerical and categorical data, and it also has methods for balancing error in class populations.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred, average='weighted')
Test_F1_score = f1_score(y_test, y_test_pred, average='weighted')

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end