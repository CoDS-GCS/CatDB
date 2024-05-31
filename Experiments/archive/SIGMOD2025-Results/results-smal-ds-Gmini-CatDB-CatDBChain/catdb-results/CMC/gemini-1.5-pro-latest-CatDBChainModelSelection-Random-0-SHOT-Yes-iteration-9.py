# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

target_variable = 'Contraceptive_method_used'

categorical_features = ['Wifes_education', 'Husbands_occupation', 'Standard_of_living_index', 'Husbands_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure', 'Number_of_children_ever_born', 'Wifes_age']
numerical_features = [] 

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        # Add numerical transformers if needed, e.g., StandardScaler
    ],
    remainder='passthrough'  # Pass through any remaining columns
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1, max_leaf_nodes=500))  # Use all available cores
])