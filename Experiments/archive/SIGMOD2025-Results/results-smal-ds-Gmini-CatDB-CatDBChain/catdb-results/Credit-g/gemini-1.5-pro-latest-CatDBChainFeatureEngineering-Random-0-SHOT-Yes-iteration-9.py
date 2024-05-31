# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

categorical_features = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude',
                        'personal_status', 'num_dependents', 'existing_credits', 'employment',
                        'other_payment_plans', 'housing', 'checking_status',
                        'installment_commitment', 'credit_history', 'other_parties',
                        'foreign_worker', 'own_telephone']

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

def feature_engineering(data):
    # Example: Combining 'credit_amount' and 'duration' to get total credit cost
    data['total_credit_cost'] = data['credit_amount'] * data['duration']
    # Example: Creating age group from 'age'
    data['age_group'] = pd.cut(data['age'], bins=[18, 30, 50, 100], labels=['Young', 'Adult', 'Senior'])
    return data

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

categorical_features.append('age_group')

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)
# ```end