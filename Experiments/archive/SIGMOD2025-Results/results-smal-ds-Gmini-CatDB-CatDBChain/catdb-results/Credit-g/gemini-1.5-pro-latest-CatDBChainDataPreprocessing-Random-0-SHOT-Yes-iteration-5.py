# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Added LogisticRegression

categorical_features = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude',
                        'personal_status', 'num_dependents', 'existing_credits', 'employment',
                        'other_payment_plans', 'housing', 'duration', 'checking_status',
                        'installment_commitment', 'credit_history', 'other_parties',
                        'foreign_worker', 'own_telephone']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()) # Added LogisticRegression to the pipeline
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
# ```end