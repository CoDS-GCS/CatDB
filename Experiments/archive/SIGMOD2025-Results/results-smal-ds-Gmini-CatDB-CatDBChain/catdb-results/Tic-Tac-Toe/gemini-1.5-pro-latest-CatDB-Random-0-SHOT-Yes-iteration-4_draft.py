# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
categorical_cols = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square', 'middle-left-square', 
                   'bottom-right-square', 'top-right-square', 'middle-right-square', 'middle-middle-square', 'top-left-square']

encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

encoded_features_train = encoder.transform(train_data[categorical_cols]).toarray()
feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df_train = pd.DataFrame(encoded_features_train, columns=feature_names)
train_data = train_data.reset_index(drop=True).join(encoded_df_train)

encoded_features_test = encoder.transform(test_data[categorical_cols]).toarray()
encoded_df_test = pd.DataFrame(encoded_features_test, columns=feature_names)
test_data = test_data.reset_index(drop=True).join(encoded_df_test)


target_variable = 'Class'

trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
trn.fit(train_data.drop(columns=[target_variable] + categorical_cols), train_data[target_variable])


train_predictions = trn.predict(train_data.drop(columns=[target_variable] + categorical_cols))
test_predictions = trn.predict(test_data.drop(columns=[target_variable] + categorical_cols))

Train_Accuracy = accuracy_score(train_data[target_variable], train_predictions)
Test_Accuracy = accuracy_score(test_data[target_variable], test_predictions)
Train_F1_score = f1_score(train_data[target_variable], train_predictions)
Test_F1_score = f1_score(test_data[target_variable], test_predictions)
Train_AUC = roc_auc_score(train_data[target_variable], train_predictions)
Test_AUC = roc_auc_score(test_data[target_variable], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end