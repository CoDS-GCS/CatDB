# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

categorical_cols = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                   'middle-left-square', 'bottom-right-square', 'top-right-square',
                   'middle-right-square', 'middle-middle-square', 'top-left-square', 'Class']

train_data = pd.read_csv('../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv')
test_data = pd.read_csv('../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
# ```end
# ```python
train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end
# ```python
# ```end