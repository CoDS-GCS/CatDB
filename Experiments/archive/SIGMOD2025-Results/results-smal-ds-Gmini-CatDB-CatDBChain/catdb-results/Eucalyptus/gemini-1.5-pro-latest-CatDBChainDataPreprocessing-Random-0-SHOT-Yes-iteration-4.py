# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


categorical_cols = ["Stem_Fm", "Vig", "Brnch_Fm", "Ins_res", "Crown_Fm", "Altitude", "Rep", "Rainfall", "Map_Ref", "Locality", "Frosts", "Utility", "Sp", "Latitude", "Year", "Abbrev"]
numerical_cols = ["DBH", "Surv", "Ht", "PMCno"]

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, categorical_cols)
    ])

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', MultiOutputClassifier(LogisticRegression(), n_jobs=-1))
    ]
)