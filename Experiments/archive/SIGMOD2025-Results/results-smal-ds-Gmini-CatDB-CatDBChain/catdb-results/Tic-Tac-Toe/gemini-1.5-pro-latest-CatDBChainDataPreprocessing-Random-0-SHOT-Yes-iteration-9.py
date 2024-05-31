# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

categorical_features = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                        'middle-left-square', 'bottom-right-square', 'top-right-square',
                        'middle-right-square', 'middle-middle-square', 'top-left-square', 'Class']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('encoder', encoder, categorical_features)],
    remainder='passthrough'  # Pass through other columns
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
# ```end