# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall',
                       'Map_Ref', 'Locality', 'Frosts', 'Sp', 'Latitude', 'Year', 'Abbrev']  # Removed 'Utility'

numerical_features = ['DBH', 'Ht', 'Surv']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('imputer', SimpleImputer(strategy='mean'))  # Impute missing values after one-hot encoding
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor)
    ]
)

train_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_train.csv")
test_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_test.csv")

X_train = train_data.drop('Utility', axis=1)
y_train = train_data['Utility']
X_test = test_data.drop('Utility', axis=1)
y_test = test_data['Utility']

train_data_processed = pipeline.fit_transform(X_train, y_train)

test_data_processed = pipeline.transform(X_test)
# ```end