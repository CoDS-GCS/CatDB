# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

train_data = pd.read_csv("../../../data/gen_dataset_53-out-0.02-np-0-nc-0-mv-0_rnc/gen_dataset_53-out-0.02-np-0-nc-0-mv-0_rnc_train.csv")
test_data = pd.read_csv("../../../data/gen_dataset_53-out-0.02-np-0-nc-0-mv-0_rnc/gen_dataset_53-out-0.02-np-0-nc-0-mv-0_rnc_test.csv")

categorical_cols = ['c_5', 'c_3', 'c_1', 'c_6', 'c_10', 'c_12', 'c_15', 'c_4', 'c_8', 'c_2']


ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))  # Fit on combined data
train_cat_features = ohe.transform(train_data[categorical_cols])
test_cat_features = ohe.transform(test_data[categorical_cols])

train_cat_df = pd.DataFrame(train_cat_features, columns=ohe.get_feature_names_out(categorical_cols))
test_cat_df = pd.DataFrame(test_cat_features, columns=ohe.get_feature_names_out(categorical_cols))

train_data = pd.concat([train_data.drop(categorical_cols, axis=1).reset_index(drop=True), train_cat_df], axis=1)
test_data = pd.concat([test_data.drop(categorical_cols, axis=1).reset_index(drop=True), test_cat_df], axis=1)


model = RandomForestRegressor(n_jobs=-1, random_state=42)  # Using all cores for training

X_train = train_data.drop(columns=['c_17'])
y_train = train_data['c_17']
X_test = test_data.drop(columns=['c_17'])
y_test = test_data['c_17']



trn = model.fit(X_train, y_train)

Train_R_Squared = r2_score(y_train, model.predict(X_train))
Test_R_Squared = r2_score(y_test, model.predict(X_test))

Train_RMSE = mean_squared_error(y_train, model.predict(X_train), squared=False)
Test_RMSE = mean_squared_error(y_test, model.predict(X_test), squared=False)

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end