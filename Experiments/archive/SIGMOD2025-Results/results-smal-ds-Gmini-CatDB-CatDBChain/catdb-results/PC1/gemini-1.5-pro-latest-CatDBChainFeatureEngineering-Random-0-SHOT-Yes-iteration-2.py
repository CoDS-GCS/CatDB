# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['loc', 'N', 'V', 'D', 'E', 'T', 'B', 'I', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

target_variable = 'defects'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Apply standard scaling to numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=10)),  # Select top 10 features using ANOVA F-value
    ('pca', PCA(n_components=0.95))  # Apply PCA to retain 95% of variance
])

X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test = test_data.drop(columns=[target_variable])
y_test = test_data[target_variable]

pipeline.fit(X_train, y_train)

X_test_transformed = pipeline.transform(X_test)
# ```end