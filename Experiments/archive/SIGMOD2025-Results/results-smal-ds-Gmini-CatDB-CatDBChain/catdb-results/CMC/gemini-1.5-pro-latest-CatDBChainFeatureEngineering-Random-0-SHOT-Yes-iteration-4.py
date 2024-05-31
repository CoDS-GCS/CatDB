# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

target_variable = 'Contraceptive_method_used'

categorical_features = ['Wifes_education', 'Husbands_occupation', 'Standard-of-living_index', 
                        'Husbands_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']
numerical_features = ['Wifes_age', 'Number_of_children_ever_born']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features),
        # No specific transformation for numerical features for now, but you can add scaling if needed
        # ('num', StandardScaler(), numerical_features) 
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")

X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test = test_data.drop(columns=[target_variable])
y_test = test_data[target_variable]

X_train_processed = pipeline.fit_transform(X_train)

X_test_processed = pipeline.transform(X_test)
# ```end