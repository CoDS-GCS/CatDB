# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

categorical_features = ['Wifes_education', 'Number_of_children_ever_born', 'Husbands_occupation', 'Wifes_age', 'Standard_of_living_index', 'Husbands_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']
numerical_features = [col for col in  ['Wifes_education', 'Number_of_children_ever_born', 'Husbands_occupation', 'Wifes_age', 'Standard_of_living_index', 'Husbands_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure','Contraceptive_method_used'] if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        # You can add more transformers for numerical features if needed
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(LogisticRegression(n_jobs=-1)))  # Use all available cores
])
# ```end