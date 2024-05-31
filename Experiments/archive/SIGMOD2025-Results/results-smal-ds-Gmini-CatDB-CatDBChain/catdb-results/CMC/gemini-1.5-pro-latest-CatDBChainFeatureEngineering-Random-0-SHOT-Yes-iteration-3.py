# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_features = ['Wifes_education', 'Number_of_children_ever_born', 'Husbands_occupation', 'Wifes_age',
                        'Standard-of-living_index', 'Husbands_education', 
                        'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']

encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")

train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end