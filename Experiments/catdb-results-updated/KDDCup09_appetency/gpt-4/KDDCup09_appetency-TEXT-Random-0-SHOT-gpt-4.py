# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/KDDCup09_appetency/KDDCup09_appetency_train.csv')
test_data = pd.read_csv('data/KDDCup09_appetency/KDDCup09_appetency_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: Var136_Var168
# Usefulness: This feature might capture some interaction between Var136 and Var168 that could be useful for predicting 'APPETENCY'.
train_data['Var136_Var168'] = train_data['Var136'] * train_data['Var168']
test_data['Var136_Var168'] = test_data['Var136'] * test_data['Var168']
# end-added-column

# python-dropping-columns
# Drop columns with string data type as they might not be useful for our Random Forest model
# and also to avoid overfitting.
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data.drop(columns=[col], inplace=True)
        test_data.drop(columns=[col], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# I chose Random Forest because it's a powerful ensemble method which can handle both numerical and categorical features well.
# It's also less likely to overfit compared to other models.

# Separate features and target
X_train = train_data.drop('APPETENCY', axis=1)
y_train = train_data['APPETENCY']
X_test = test_data.drop('APPETENCY', axis=1)
y_test = test_data['APPETENCY']

# Preprocessing for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X_train.select_dtypes(include=['int64', 'float64']).columns)])

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
preds_test = my_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds_test)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation