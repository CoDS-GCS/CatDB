# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

categorical_cols = ['Normal_Nucleoli', 'Bland_Chromatin', 'Clump_Thickness', 'Cell_Shape_Uniformity',
                   'Bare_Nuclei', 'Cell_Size_Uniformity', 'Marginal_Adhesion', 'Mitoses', 'Single_Epi_Cell_Size']

train_data = pd.read_csv('../../../data/Breast-w/Breast-w_train.csv')
test_data = pd.read_csv('../../../data/Breast-w/Breast-w_test.csv')

train_data['Combined_Feature'] = train_data['Normal_Nucleoli'] * train_data['Clump_Thickness']
test_data['Combined_Feature'] = test_data['Normal_Nucleoli'] * test_data['Clump_Thickness']

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_cols),
        ('num', StandardScaler(), ['Combined_Feature'])  # Apply scaling to the new feature
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(X_train)
test_data_processed = pipeline.transform(X_test)
# ```end