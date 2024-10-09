# ```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

combined_data = pd.concat([train_data, test_data], ignore_index=True)

numerical_cols = ["Complaint Response Time", "Avg Age of Cases Pending", "Escalated Complaint Response Time", "Initial Complaints"]
categorical_cols = ["PCM Index", "ERM Index", "CRM Index", "CSM Index", "Service Provider", "Escalated Complaints", "Year", "Month"]

for col in numerical_cols:
    Q1 = combined_data[col].quantile(0.25)
    Q3 = combined_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    combined_data = combined_data[(combined_data[col] >= lower_bound) & (combined_data[col] <= upper_bound)]


combined_data['Combined Complaint Ratio'] = combined_data['Escalated Complaints'] / combined_data['Initial Complaints']

train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])


trn = pipeline.fit(train_data.drop('CSRI', axis=1), train_data['CSRI'])

Train_R_Squared = r2_score(train_data['CSRI'], trn.predict(train_data.drop('CSRI', axis=1)))
Train_RMSE = np.sqrt(mean_squared_error(train_data['CSRI'], trn.predict(train_data.drop('CSRI', axis=1))))
Test_R_Squared = r2_score(test_data['CSRI'], trn.predict(test_data.drop('CSRI', axis=1)))
Test_RMSE = np.sqrt(mean_squared_error(test_data['CSRI'], trn.predict(test_data.drop('CSRI', axis=1))))
print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")

# ```end