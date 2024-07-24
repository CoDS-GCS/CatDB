# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the training and test datasets
train_data = pd.read_csv('../../../data/eucalyptus/eucalyptus_train.csv')
test_data = pd.read_csv('../../../data/eucalyptus/eucalyptus_test.csv')

# Perform data cleaning and preprocessing
# Impute missing values for numerical columns
num_cols = ['PMCno','Surv','Stem_Fm','Vig','Brnch_Fm','Crown_Fm','Ins_res','DBH','Ht']
imputer = SimpleImputer(strategy='mean')
train_data[num_cols] = imputer.fit_transform(train_data[num_cols])
test_data[num_cols] = imputer.transform(test_data[num_cols])

# Perform feature processing
# Scale numerical columns
num_cols = ['Sp','Abbrev','Rainfall','Locality','Altitude','Latitude','Frosts','Rep','Year','Map_Ref','PMCno','Surv','Stem_Fm','Vig','Brnch_Fm','Crown_Fm','Ins_res','DBH','Ht']
scaler = MinMaxScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])

# One-hot encode categorical columns
cat_cols = ['Frosts','Rep','Year']
encoder = OneHotEncoder()
train_data = pd.concat([train_data, pd.DataFrame(encoder.fit_transform(train_data[cat_cols]).toarray(), columns=encoder.get_feature_names_out(cat_cols))], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(encoder.transform(test_data[cat_cols]).toarray(), columns=encoder.get_feature_names_out(cat_cols))], axis=1)

# Drop original categorical columns
train_data.drop(columns=cat_cols, inplace=True)
test_data.drop(columns=cat_cols, inplace=True)

# Select the appropriate features and target variables
X_train = train_data.drop(columns=['Utility'])
y_train = train_data['Utility']
X_test = test_data.drop(columns=['Utility'])
y_test = test_data['Utility']

# Convert all column names to string
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen due to its ability to handle both numerical and categorical features, and its robustness to overfitting
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, clf.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, clf.predict_proba(X_test))

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")