# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/BNG_heart-statlog/BNG_heart-statlog_train.csv')
test_data = pd.read_csv('../../../data/BNG_heart-statlog/BNG_heart-statlog_test.csv')

# Perform feature processing
# Define the columns to be scaled and encoded
scale_cols = ['thal', 'resting_electrocardiographic_results', 'number_of_major_vessels', 'slope', 'age', 'resting_blood_pressure', 'maximum_heart_rate_achieved', 'oldpeak', 'serum_cholestoral', 'chest']
encode_cols = ['thal', 'resting_electrocardiographic_results', 'number_of_major_vessels', 'slope', 'fasting_blood_sugar', 'exercise_induced_angina', 'sex']

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

# Fit and transform the training data
X_train = preprocessor.fit_transform(X_train)

# Transform the test data
X_test = preprocessor.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Fit the model
clf.fit(X_train, y_train)

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