# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['is_train'] = 1
test_data['is_train'] = 0
df = pd.concat([train_data, test_data], ignore_index=True)


numerical_cols = ['actor_id', 'director_id', 'movie_id', 'prob', 'rank']
categorical_cols = ['year', 'genre_y', 'genre_x']
string_cols_to_impute = ['last_name_x', 'name', 'role']
high_cardinality_cols = ['last_name_x', 'last_name_y', 'first_name_x', 'first_name_y', 'name', 'role']
target_col = 'gender'


df.dropna(subset=[target_col], inplace=True)

median_imputer = SimpleImputer(strategy='median')
df[['rank']] = median_imputer.fit_transform(df[['rank']])

constant_imputer = SimpleImputer(strategy='constant', fill_value='missing')
df[string_cols_to_impute] = constant_imputer.fit_transform(df[string_cols_to_impute])

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

for col in ['prob', 'rank']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers from the dataframe
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]



actor_movie_counts = df.groupby('actor_id')['movie_id'].count().to_dict()
df['actor_movie_count'] = df['actor_id'].map(actor_movie_counts)

director_movie_counts = df.groupby('director_id')['movie_id'].count().to_dict()
df['director_movie_count'] = df['director_id'].map(director_movie_counts)

movie_cast_sizes = df.groupby('movie_id')['actor_id'].nunique().to_dict()
df['movie_cast_size'] = df['movie_id'].map(movie_cast_sizes)

actor_avg_ranks = df.groupby('actor_id')['rank'].mean().to_dict()
df['actor_avg_rank'] = df['actor_id'].map(actor_avg_ranks)

director_avg_ranks = df.groupby('director_id')['rank'].mean().to_dict()
df['director_avg_rank'] = df['director_id'].map(director_avg_ranks)

for col in ['actor_movie_count', 'director_movie_count', 'movie_cast_size', 'actor_avg_rank', 'director_avg_rank']:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col] = df[col].astype('category')

engineered_features = ['actor_movie_count', 'director_movie_count', 'movie_cast_size', 'actor_avg_rank', 'director_avg_rank']
numerical_features_to_scale = ['prob', 'rank'] + engineered_features

scaler = StandardScaler()

scaler.fit(df.loc[df['is_train'] == 1, numerical_features_to_scale])

df[numerical_features_to_scale] = scaler.transform(df[numerical_features_to_scale])

df.drop(columns=['actor_id'], inplace=True)

df.drop(columns=['director_id'], inplace=True)

df.drop(columns=['movie_id'], inplace=True)

df.drop(columns=high_cardinality_cols, inplace=True)

train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
test_df = df[df['is_train'] == 0].drop(columns=['is_train'])

X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

lgb_clf = lgb.LGBMClassifier(objective='binary',
                             random_state=42,
                             n_jobs=-1, # Use all available CPU cores
                             scale_pos_weight=scale_pos_weight)

lgb_clf.fit(X_train, y_train, categorical_feature=[col for col in categorical_cols if col in X_train.columns])

train_pred_proba = lgb_clf.predict_proba(X_train)[:, 1]
test_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]
train_pred = lgb_clf.predict(X_train)
test_pred = lgb_clf.predict(X_test)

Train_AUC = roc_auc_score(y_train, train_pred_proba)
Test_AUC = roc_auc_score(y_test, test_pred_proba)

Train_Accuracy = accuracy_score(y_train, train_pred)
Test_Accuracy = accuracy_score(y_test, test_pred)

Train_F1_score = f1_score(y_train, train_pred)
Test_F1_score = f1_score(y_test, test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}")
# ```end