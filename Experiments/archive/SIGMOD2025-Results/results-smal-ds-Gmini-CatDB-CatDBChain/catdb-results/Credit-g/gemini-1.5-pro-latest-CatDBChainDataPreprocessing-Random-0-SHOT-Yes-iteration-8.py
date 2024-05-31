# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")

categorical_features = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude',
                        'personal_status', 'num_dependents', 'existing_credits', 'employment',
                        'other_payment_plans', 'housing', 'duration', 'checking_status',
                        'installment_commitment', 'credit_history', 'other_parties',
                        'foreign_worker', 'own_telephone', 'class']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end