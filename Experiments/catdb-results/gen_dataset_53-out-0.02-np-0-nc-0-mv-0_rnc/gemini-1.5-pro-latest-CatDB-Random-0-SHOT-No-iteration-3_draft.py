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
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

def encode_data(df):
    encoded_features = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=[f"col_{i}" for i in range(encoded_features.shape[1])])
    df = df.drop(columns=categorical_cols, axis=1)
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df

train_data = encode_data(train_data.copy())
test_data = encode_data(test_data.copy())


def feature_engineer(df):
    # (Feature name: c_9_times_c_7)
    # Usefulness: Represents the interaction between c_9 and c_7, which might be relevant for predicting 'c_17'.
    df['c_9_times_c_7'] = df['c_9'] * df['c_7']
    return df

train_data = feature_engineer(train_data)
test_data = feature_engineer(test_data)

X_train = train_data.drop(columns=['c_17'])
y_train = train_data['c_17']
X_test = test_data.drop(columns=['c_17'])
y_test = test_data['c_17']


model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

Train_R_Squared = r2_score(y_train, model.predict(X_train))
Train_RMSE = mean_squared_error(y_train, model.predict(X_train), squared=False)
Test_R_Squared = r2_score(y_test, model.predict(X_test))
Test_RMSE = mean_squared_error(y_test, model.predict(X_test), squared=False)

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end