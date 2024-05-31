# ```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['loc', 'N', 'V', 'D', 'E', 'T', 'B', 'I', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

train_data['N_density'] = train_data['N'] / train_data['loc']  # Operators and operands density
train_data['V_density'] = train_data['V'] / train_data['loc']  # Halstead Volume density
train_data['D_density'] = train_data['D'] / train_data['loc']  # Halstead Difficulty density

test_data['N_density'] = test_data['N'] / test_data['loc']
test_data['V_density'] = test_data['V'] / test_data['loc']
test_data['D_density'] = test_data['D'] / test_data['loc']

numerical_features.extend(['N_density', 'V_density', 'D_density'])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing values
            ('scaler', StandardScaler()),  # Scale numerical features
            ('pca', PCA(n_components=0.95))  # Apply PCA for dimensionality reduction
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])