# ```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['loc', 'N', 'V', 'D', 'E', 'T', 'B', 'I', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
X_train = train_data.drop('defects', axis=1)
y_train = train_data['defects']
X_test = test_data.drop('defects', axis=1)
y_test = test_data['defects']
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
# ```end