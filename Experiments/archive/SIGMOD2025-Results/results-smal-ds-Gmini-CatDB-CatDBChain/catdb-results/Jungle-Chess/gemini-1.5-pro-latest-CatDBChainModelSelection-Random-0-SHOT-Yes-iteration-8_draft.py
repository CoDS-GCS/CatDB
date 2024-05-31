# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

categorical_features = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength',
                        'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500))
])

with parallel_backend('threading'):
    pipeline.fit(train_data.drop('class', axis=1), train_data['class'])

    train_predictions = pipeline.predict(train_data.drop('class', axis=1))
    train_proba = pipeline.predict_proba(train_data.drop('class', axis=1))
    test_predictions = pipeline.predict(test_data.drop('class', axis=1))
    test_proba = pipeline.predict_proba(test_data.drop('class', axis=1))

Train_Accuracy = accuracy_score(train_data['class'], train_predictions)
Test_Accuracy = accuracy_score(test_data['class'], test_predictions)
Train_Log_loss = log_loss(train_data['class'], train_proba)
Test_Log_loss = log_loss(test_data['class'], test_proba)
Train_AUC_OVO = roc_auc_score(train_data['class'], train_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data['class'], train_proba, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_data['class'], test_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data['class'], test_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end