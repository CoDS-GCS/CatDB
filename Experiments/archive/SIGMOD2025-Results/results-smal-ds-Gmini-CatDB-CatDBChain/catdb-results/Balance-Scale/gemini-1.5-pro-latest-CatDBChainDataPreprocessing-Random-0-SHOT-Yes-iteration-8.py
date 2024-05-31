# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance', 'class']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end