# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression

target_variable = 'Contraceptive_method_used'

categorical_features = ['Wifes_education', 'Husbands_occupation', 'Standard_of_living_index', 'Husbands_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']
numerical_features = ['Wifes_age', 'Number_of_children_ever_born'] 

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        # Add numerical transformers if needed, e.g., StandardScaler
    ],
    remainder='passthrough'  # Pass through any remaining columns
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(n_jobs=-1))  # Use all available cores
])
# ```end