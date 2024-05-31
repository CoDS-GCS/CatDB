# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")



categorical_cols = ['Normal_Nucleoli', 'Bland_Chromatin', 'Clump_Thickness', 'Cell_Shape_Uniformity',
                   'Bare_Nuclei', 'Cell_Size_Uniformity', 'Marginal_Adhesion', 'Mitoses', 'Single_Epi_Cell_Size']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = encoder.transform(train_data[categorical_cols])
test_encoded = encoder.transform(test_data[categorical_cols])

train_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

train_data = pd.concat([train_data, train_df], axis=1)
test_data = pd.concat([test_data, test_df], axis=1)


target_col = 'Class'

trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)

trn.fit(train_data.drop(columns=[target_col]), train_data[target_col])

train_predictions = trn.predict(train_data.drop(columns=[target_col]))
test_predictions = trn.predict(test_data.drop(columns=[target_col]))

Train_Accuracy = accuracy_score(train_data[target_col], train_predictions)
Test_Accuracy = accuracy_score(test_data[target_col], test_predictions)

Train_F1_score = f1_score(train_data[target_col], train_predictions)
Test_F1_score = f1_score(test_data[target_col], test_predictions)

Train_AUC = roc_auc_score(train_data[target_col], train_predictions)
Test_AUC = roc_auc_score(test_data[target_col], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")

print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end