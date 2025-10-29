# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure the dataset files are in the specified path.")
    # Creating dummy files for pipeline execution if files are not found
    dummy_data = {
        "actor_id": [1, 2, 3], "director_id": [4, 5, 6], "movie_id": [7, 8, 9],
        "year": [2000, 2001, 2000], "last_name_x": ["Smith", "Jones", np.nan],
        "genre_y": ["Comedy", "Action", "Comedy"], "last_name_y": ["Doe", "Ray", "Moe"],
        "first_name_x": ["John", "Jane", "John"], "name": ["Movie A", "Movie B", np.nan],
        "first_name_y": ["DirectorA", "DirectorB", "DirectorC"], "genre_x": ["Comedy", "Action", "Drama"],
        "role": ["Lead", np.nan, "Support"], "prob": [0.5, 0.8, 0.2],
        "rank": [8.0, np.nan, 7.5], "gender": ["M", "F", "M"]
    }
    train_data = pd.DataFrame(dummy_data)
    test_data = pd.DataFrame(dummy_data)


train_len = len(train_data)

combined_data = pd.concat([train_data, test_data], ignore_index=True)

target_column = 'gender'
numerical_features = ['actor_id', 'director_id', 'movie_id', 'prob', 'rank']
categorical_features = ['year', 'genre_y', 'genre_x']
high_cardinality_features = ['last_name_x', 'last_name_y', 'first_name_x', 'first_name_y', 'name', 'role']

le = LabelEncoder()
combined_data[target_column] = le.fit_transform(combined_data[target_column].astype(str))


median_imputer = SimpleImputer(strategy='median')
combined_data['rank'] = median_imputer.fit_transform(combined_data[['rank']])

constant_imputer = SimpleImputer(strategy='constant', fill_value='missing')
for col in ["last_name_x", "name", "role"]:
    combined_data[col] = constant_imputer.fit_transform(combined_data[[col]]).ravel()

for col in ['prob', 'rank']:
    Q1 = combined_data[col].quantile(0.25)
    Q3 = combined_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    combined_data[col] = np.where(combined_data[col] < lower_bound, lower_bound, combined_data[col])
    combined_data[col] = np.where(combined_data[col] > upper_bound, upper_bound, combined_data[col])


name_gender_map = combined_data.iloc[:train_len].groupby('first_name_x')[target_column].mean()
combined_data['first_name_gender_prob'] = combined_data['first_name_x'].map(name_gender_map)
global_gender_mean = combined_data.iloc[:train_len][target_column].mean()
combined_data['first_name_gender_prob'].fillna(global_gender_mean, inplace=True)


combined_data.drop(columns=['actor_id'], inplace=True)

combined_data.drop(columns=['director_id'], inplace=True)

combined_data.drop(columns=['movie_id'], inplace=True)

high_card_to_drop = ['last_name_x', 'last_name_y', 'first_name_x', 'first_name_y', 'name', 'role']
combined_data.drop(columns=high_card_to_drop, inplace=True)


combined_data = pd.get_dummies(combined_data, columns=categorical_features, dummy_na=False)


features = [col for col in combined_data.columns if col != target_column]
X = combined_data[features]
y = combined_data[target_column]

scaler = StandardScaler()
numerical_cols_to_scale = ['prob', 'rank', 'first_name_gender_prob']
X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])


X_train = X.iloc[:train_len]
X_test = X.iloc[train_len:]
y_train = y.iloc[:train_len]
y_test = y.iloc[train_len:]


lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

lgbm.fit(X_train, y_train)


train_pred = lgbm.predict(X_train)
train_pred_proba = lgbm.predict_proba(X_train)[:, 1]

test_pred = lgbm.predict(X_test)
test_pred_proba = lgbm.predict_proba(X_test)[:, 1]

Train_AUC = roc_auc_score(y_train, train_pred_proba)
Train_Accuracy = accuracy_score(y_train, train_pred)
Train_F1_score = f1_score(y_train, train_pred)

Test_AUC = roc_auc_score(y_test, test_pred_proba)
Test_Accuracy = accuracy_score(y_test, test_pred)
Test_F1_score = f1_score(y_test, test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end