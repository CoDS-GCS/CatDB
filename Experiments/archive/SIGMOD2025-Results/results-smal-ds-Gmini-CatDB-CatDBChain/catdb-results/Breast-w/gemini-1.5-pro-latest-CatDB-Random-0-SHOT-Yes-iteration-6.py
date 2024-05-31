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

combined_data = pd.concat([train_data, test_data])
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(combined_data[categorical_cols])

train_encoded = encoder.transform(train_data[categorical_cols])
test_encoded = encoder.transform(test_data[categorical_cols])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

train_data = train_data.drop(categorical_cols, axis=1).reset_index(drop=True)
train_data = pd.concat([train_data, train_encoded_df], axis=1)

test_data = test_data.drop(categorical_cols, axis=1).reset_index(drop=True)
test_data = pd.concat([test_data, test_encoded_df], axis=1)

train_data['Ratio_Nucleoli_Chromatin'] = train_data['Normal_Nucleoli_4.0'] / (train_data['Bland_Chromatin_3.0'] + 1e-5)  # Adding a small constant to avoid division by zero
test_data['Ratio_Nucleoli_Chromatin'] = test_data['Normal_Nucleoli_4.0'] / (test_data['Bland_Chromatin_3.0'] + 1e-5)


trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42, n_jobs=-1) 

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

trn.fit(X_train, y_train)

train_preds = trn.predict(X_train)
test_preds = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_preds)
Test_Accuracy = accuracy_score(y_test, test_preds)

Train_F1_score = f1_score(y_train, train_preds)
Test_F1_score = f1_score(y_test, test_preds)

Train_AUC = roc_auc_score(y_train, train_preds)
Test_AUC = roc_auc_score(y_test, test_preds)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end