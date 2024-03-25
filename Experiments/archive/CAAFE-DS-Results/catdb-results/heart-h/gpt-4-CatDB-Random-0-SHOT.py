# Import all required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Load the datasets
train_data = pd.read_csv('../../../data/heart-h/heart-h_train.csv')
test_data = pd.read_csv('../../../data/heart-h/heart-h_test.csv')

# Drop the 'ca' column
train_data.drop(columns=['ca'], inplace=True)
test_data.drop(columns=['ca'], inplace=True)

# Define preprocessing steps
num_cols = ['thalach', 'age', 'trestbps', 'chol', 'oldpeak']
cat_cols = ['chest_pain', 'thal', 'exang', 'fbs', 'sex', 'slope', 'restecg']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Define the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Combine preprocessing and classifier into a pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', classifier)])

# Train the model
X_train = train_data.drop('num', axis=1)
y_train = train_data['num']
model.fit(X_train, y_train)

# Evaluate the model
X_test = test_data.drop('num', axis=1)
y_test = test_data['num']

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred, average='weighted')
Test_F1_score = f1_score(y_test, y_test_pred, average='weighted')

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")