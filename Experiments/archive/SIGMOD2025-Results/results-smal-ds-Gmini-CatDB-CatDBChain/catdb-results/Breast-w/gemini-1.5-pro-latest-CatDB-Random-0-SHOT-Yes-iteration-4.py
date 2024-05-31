# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
combined_data = pd.concat([train_data, test_data], axis=0)
for column in ["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", "Single_Epi_Cell_Size"]:
    encoder.fit(combined_data[[column]])
    train_encoded = encoder.transform(train_data[[column]]).toarray()
    test_encoded = encoder.transform(test_data[[column]]).toarray()
    train_encoded_df = pd.DataFrame(train_encoded, columns=[f"{column}_{i}" for i in range(train_encoded.shape[1])])
    test_encoded_df = pd.DataFrame(test_encoded, columns=[f"{column}_{i}" for i in range(test_encoded.shape[1])])
    train_data = pd.concat([train_data, train_encoded_df], axis=1)
    test_data = pd.concat([test_data, test_encoded_df], axis=1)

train_data.drop(columns=["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", "Single_Epi_Cell_Size"], inplace=True)
test_data.drop(columns=["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", "Single_Epi_Cell_Size"], inplace=True)

trn = RandomForestClassifier(max_leaf_nodes=500)

trn.fit(train_data.drop(columns=['Class']), train_data['Class'])

train_predictions = trn.predict(train_data.drop(columns=['Class']))
test_predictions = trn.predict(test_data.drop(columns=['Class']))

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