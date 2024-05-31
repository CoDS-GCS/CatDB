# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))  # n_jobs=-1 to use all cores
])

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)
Train_AUC = roc_auc_score(y_train, y_train_pred)
Test_AUC = roc_auc_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end