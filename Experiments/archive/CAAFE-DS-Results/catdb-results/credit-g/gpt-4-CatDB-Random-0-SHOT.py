# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/credit-g/credit-g_train.csv')
test_data = pd.read_csv('../../../data/credit-g/credit-g_test.csv')

# Perform feature processing
# Define the columns to be scaled and encoded
scale_cols = ['residence_since', 'job', 'other_parties', 'duration', 'employment', 'housing', 'checking_status', 'savings_status', 'property_magnitude', 'credit_amount', 'credit_history', 'age', 'other_payment_plans', 'existing_credits', 'personal_status', 'purpose', 'installment_commitment', 'num_dependents']
encode_cols = ['residence_since', 'job', 'other_parties', 'employment', 'housing', 'checking_status', 'savings_status', 'property_magnitude', 'credit_history', 'other_payment_plans', 'existing_credits', 'personal_status', 'installment_commitment', 'num_dependents', 'own_telephone', 'foreign_worker']

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), scale_cols),
        ('cat', OneHotEncoder(), encode_cols)])

# Select the appropriate features and target variables
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

# Choose the suitable machine learning algorithm or technique (classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline that preprocesses the data and then fits the classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Report evaluation based on train and test dataset
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")