# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression

categorical_features = ['Wifes_education', 'Husbands_occupation', 'Standard_of_living_index', 'Husbands_education', 
                       'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']
numerical_features = ['Wifes_age', 'Number_of_children_ever_born']

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', selector(dtype_include=['int64'])),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=['object', 'bool']))
    ])

model = LogisticRegression(n_jobs=-1)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])
# ```end