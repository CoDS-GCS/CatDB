# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

categorical_features = ["residence_since", "savings_status", "job", "purpose", "property_magnitude",
                         "personal_status", "num_dependents", "existing_credits", "employment",
                         "other_payment_plans", "housing", "duration", "checking_status",
                         "installment_commitment", "credit_history", "other_parties",
                         "foreign_worker", "own_telephone"]
numerical_features = ["credit_amount", "age"]

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")

def feature_engineering(data):
    data['total_credit_cost'] = data['credit_amount'] * data['duration']
    return data

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

numerical_features.append('total_credit_cost')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selector(dtype_include=['int64', 'float64'])),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
    ])

model = RandomForestClassifier(max_leaf_nodes=500, random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

def process_data(data):
    X = data.drop('class', axis=1)
    y = data['class']
    return X, y

n_jobs = -1  # Use all available cores
results = Parallel(n_jobs=n_jobs)(delayed(process_data)(df) for df in [train_data, test_data])
X_train, y_train = results[0]
X_test, y_test = results[1]

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)

y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

Train_AUC = roc_auc_score(y_train, y_train_pred)
Test_AUC = roc_auc_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end