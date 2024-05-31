# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

train_data = pd.read_csv('../../../data/Balance-Scale/Balance-Scale_train.csv')
test_data = pd.read_csv('../../../data/Balance-Scale/Balance-Scale_test.csv')

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

X_train_transformed = pipeline.fit_transform(X_train)

X_test_transformed = pipeline.transform(X_test)
# ```end