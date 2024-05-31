# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")

categorical_features = ['Normal_Nucleoli', 'Bland_Chromatin', 'Clump_Thickness',
                        'Cell_Shape_Uniformity', 'Bare_Nuclei', 'Cell_Size_Uniformity',
                        'Marginal_Adhesion', 'Mitoses', 'Single_Epi_Cell_Size']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, random_state=42))
])
# ```end
# ```python
from sklearn.utils import parallel_backend

with parallel_backend('threading'):
    pipeline.fit(train_data.drop(columns=['Class']), train_data['Class'])
    train_predictions = pipeline.predict(train_data.drop(columns=['Class']))
    test_predictions = pipeline.predict(test_data.drop(columns=['Class']))
# ```end
# ```python
Train_Accuracy = accuracy_score(train_data['Class'], train_predictions)
Test_Accuracy = accuracy_score(test_data['Class'], test_predictions)
Train_F1_score = f1_score(train_data['Class'], train_predictions)
Test_F1_score = f1_score(test_data['Class'], test_predictions)
Train_AUC = roc_auc_score(train_data['Class'], train_predictions)
Test_AUC = roc_auc_score(test_data['Class'], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end