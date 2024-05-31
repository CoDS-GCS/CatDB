# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

categorical_features = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                       'middle-left-square', 'bottom-right-square', 'top-right-square',
                       'middle-right-square', 'middle-middle-square', 'top-left-square']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(n_jobs=-1))
])

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

pipeline.fit(X_train, y_train)
# ```end