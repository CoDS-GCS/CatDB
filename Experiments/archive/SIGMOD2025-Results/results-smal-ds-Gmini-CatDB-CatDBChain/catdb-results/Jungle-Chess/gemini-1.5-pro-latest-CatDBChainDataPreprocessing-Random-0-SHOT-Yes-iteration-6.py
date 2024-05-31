# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 

categorical_features = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 
                        'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(n_jobs=-1))
])

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline.fit(X_train, y_train)
# ```end