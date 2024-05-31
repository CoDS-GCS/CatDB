# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['I', 'B', 'uniq_Opnd', 'E', 'N', 'loc', 'total_Opnd', 'total_Op', 'V', 'T', 'branchCount', 'D', 'lOCode']

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

def add_interaction_terms(X):
    for i in range(len(numerical_features)):
        for j in range(i+1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            X[f'{feature1}_x_{feature2}'] = X[feature1] * X[feature2]
    return X

train_data = add_interaction_terms(train_data.copy())
test_data = add_interaction_terms(test_data.copy())

numerical_features += [col for col in train_data.columns if col not in categorical_features + ['defects']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),  # Standardize numerical features
            ('pca', PCA(n_components=0.95)),  # Apply PCA for dimensionality reduction
            ('feature_selection', SelectKBest(f_classif, k=10))  # Select top k features
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1, max_leaf_nodes=500, random_state=42))
])

pipeline.fit(train_data.drop('defects', axis=1), train_data['defects'])

train_predictions = pipeline.predict(train_data.drop('defects', axis=1))
test_predictions = pipeline.predict(test_data.drop('defects', axis=1))

Train_Accuracy = accuracy_score(train_data['defects'], train_predictions)
Test_Accuracy = accuracy_score(test_data['defects'], test_predictions)
Train_F1_score = f1_score(train_data['defects'], train_predictions)
Test_F1_score = f1_score(test_data['defects'], test_predictions)
Train_AUC = roc_auc_score(train_data['defects'], train_predictions)
Test_AUC = roc_auc_score(test_data['defects'], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end