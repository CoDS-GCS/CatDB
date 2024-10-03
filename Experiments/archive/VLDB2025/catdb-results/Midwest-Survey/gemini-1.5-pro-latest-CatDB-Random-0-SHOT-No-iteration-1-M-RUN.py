# ```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

combined_data = pd.concat([train_data, test_data], ignore_index=True)

imputer = SimpleImputer(strategy='median')
combined_data['ZIP Code'] = imputer.fit_transform(combined_data[['ZIP Code']])

scaler = StandardScaler()
combined_data['ZIP Code'] = scaler.fit_transform(combined_data[['ZIP Code']])

for col in ["Gender", "Household Income", "In your own words, what would you call the part of the country you live in now?", "Age", "Education", "Location (Census Region)"]:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])

combined_data['RespondentID'] = scaler.fit_transform(combined_data[['RespondentID']])

combined_data = pd.get_dummies(combined_data, columns=["Gender", "North Dakota in MW?", "Wisconsin in MW?", "Household Income", "West Virginia in MW?", "In your own words, what would you call the part of the country you live in now?", "Kansas in MW?", "Missouri in MW?", "Colorado in MW?", "Indiana in MW?", "Oklahoma in MW?", "Pennsylvania in MW?", "Age", "Illinois in MW?", "Iowa in MW?", "Education", "Ohio in MW?", "South Dakota in MW?", "Minnesota in MW?", "Arkansas in MW?", "Montana in MW?", "Wyoming in MW?", "Kentucky in MW?", "Personally identification as a Midwesterner?", "Michigan in MW?", "Nebraska in MW?"])

train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]


trn = RandomForestClassifier(n_estimators=100, random_state=42)
trn.fit(train_data.drop(columns=['Location (Census Region)']), train_data['Location (Census Region)'])

Train_Accuracy = accuracy_score(train_data['Location (Census Region)'], trn.predict(train_data.drop(columns=['Location (Census Region)'])))
Test_Accuracy = accuracy_score(test_data['Location (Census Region)'], trn.predict(test_data.drop(columns=['Location (Census Region)'])))
Train_Log_loss = log_loss(train_data['Location (Census Region)'], trn.predict_proba(train_data.drop(columns=['Location (Census Region)'])))
Test_Log_loss = log_loss(test_data['Location (Census Region)'], trn.predict_proba(test_data.drop(columns=['Location (Census Region)'])))
Train_AUC_OVO = roc_auc_score(train_data['Location (Census Region)'], trn.predict_proba(train_data.drop(columns=['Location (Census Region)'])), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data['Location (Census Region)'], trn.predict_proba(train_data.drop(columns=['Location (Census Region)'])), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_data['Location (Census Region)'], trn.predict_proba(test_data.drop(columns=['Location (Census Region)'])), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data['Location (Census Region)'], trn.predict_proba(test_data.drop(columns=['Location (Census Region)'])), multi_class='ovr')
print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end