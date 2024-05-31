# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

categorical_features = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                        'middle-left-square', 'bottom-right-square', 'top-right-square',
                        'middle-right-square', 'middle-middle-square', 'top-left-square']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(n_jobs=-1))  # Use all available cores
])
# ```end