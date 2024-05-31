# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

X_train = train_data.drop('defects', axis=1)
y_train = train_data['defects']
X_test = test_data.drop('defects', axis=1)
y_test = test_data['defects']

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Add your desired classifier here
])

pipeline.fit(X_train, y_train)
# ```end