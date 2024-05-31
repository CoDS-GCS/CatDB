# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

categorical_features = ["residence_since", "savings_status", "job", "purpose", "property_magnitude",
                         "personal_status", "num_dependents", "existing_credits", "employment",
                         "other_payment_plans", "housing", "duration", "checking_status",
                         "installment_commitment", "credit_history", "other_parties",
                         "foreign_worker", "own_telephone", "class"]
numerical_features = ["credit_amount", "age"]

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selector(dtype_include=['int64', 'float64'])),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
    ])


def process_data(data):
    return preprocessor.fit_transform(data)

n_jobs = -1  # Use all available cores
results = Parallel(n_jobs=n_jobs)(delayed(process_data)(df) for df in [train_data, test_data])
train_data_processed, test_data_processed = results
# ```end