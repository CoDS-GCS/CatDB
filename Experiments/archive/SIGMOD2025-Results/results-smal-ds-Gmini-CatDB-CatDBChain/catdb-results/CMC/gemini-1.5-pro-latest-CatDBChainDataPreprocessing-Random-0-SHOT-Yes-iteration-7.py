# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

categorical_features = ['Wifes_education', 'Number_of_children_ever_born', 'Husbands_occupation', 'Wifes_age', 'Standard_of_living_index', 'Husbands_education', 
                       'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure','Contraceptive_method_used']
numerical_features = []

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', selector(dtype_include=['int64'])),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=['object', 'bool']))
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(LogisticRegression(), n_jobs=-1))
])
# ```end