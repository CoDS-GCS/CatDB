# ```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

categorical_features = ["PCM Index", "ERM Index", "CRM Index", "CSM Index", "Service Provider", "Escalated Complaints", "Year", "Month"]
numerical_features = ["Complaint Response Time", "Avg Age of Cases Pending", "Escalated Complaint Response Time", "Initial Complaints"]

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestRegressor(n_jobs=-1)  # n_jobs=-1 to use all processors

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

target_variable = "CSRI"

pipeline.fit(train_data.drop(columns=[target_variable]), train_data[target_variable])

train_predictions = pipeline.predict(train_data.drop(columns=[target_variable]))
test_predictions = pipeline.predict(test_data.drop(columns=[target_variable]))

Train_R_Squared = r2_score(train_data[target_variable], train_predictions)
Train_RMSE = mean_squared_error(train_data[target_variable], train_predictions, squared=False)
Test_R_Squared = r2_score(test_data[target_variable], test_predictions)
Test_RMSE = mean_squared_error(test_data[target_variable], test_predictions, squared=False)

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end