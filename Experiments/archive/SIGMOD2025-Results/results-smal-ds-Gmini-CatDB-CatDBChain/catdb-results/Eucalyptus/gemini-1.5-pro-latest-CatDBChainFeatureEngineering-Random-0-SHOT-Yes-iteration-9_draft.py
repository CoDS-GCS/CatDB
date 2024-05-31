# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 
                        'Rainfall', 'Map_Ref', 'Locality', 'Frosts', 'Sp', 
                        'Latitude', 'Year', 'Abbrev']
numerical_features = ['DBH', 'Surv', 'Ht']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = make_column_transformer(
    (categorical_transformer, categorical_features),
    (numerical_transformer, numerical_features),
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_train.csv")
test_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_test.csv")

X_train = train_data.drop('Utility', axis=1)
y_train = train_data['Utility']
X_test = test_data.drop('Utility', axis=1)
y_test = test_data['Utility']

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
# ```end