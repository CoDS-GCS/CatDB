# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['loc', 'N', 'V', 'L', 'D', 'I', 'E', 'B', 'T', 'lOCode', 
                       'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

def feature_engineering(df):
    df['N+V'] = df['N'] + df['V']
    df['D/I'] = df['D'] / df['I']
    return df

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

X_train = train_data.drop('defects', axis=1)
y_train = train_data['defects']
X_test = test_data.drop('defects', axis=1)
y_test = test_data['defects']

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=0.95))  # Optional: Add PCA for dimensionality reduction
        ]), numerical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Add your desired classifier here
])

pipeline.fit(X_train, y_train)
# ```end