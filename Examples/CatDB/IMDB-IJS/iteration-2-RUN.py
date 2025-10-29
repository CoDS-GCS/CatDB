# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

try:
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure the dataset files are in the specified path.")
    # Creating dummy files for pipeline execution if files are not found
    dummy_data = {
        "actor_id": [1, 2, 3, 4, 5], "director_id": [10, 20, 10, 30, 20], "movie_id": [100, 200, 100, 300, 400],
        "year": [2000, 2001, 2000, 2002, 2001], "last_name_x": ['Smith', 'Jones', np.nan, 'Williams', 'Brown'],
        "genre_y": ['Comedy', 'Action', 'Comedy', 'Drama', 'Action'], "last_name_y": ['Lee', 'Scott', 'Lee', 'King', 'Scott'],
        "first_name_x": ['John', 'Jane', 'Peter', 'Mary', 'David'], "name": ['MovieA', 'MovieB', np.nan, 'MovieD', 'MovieE'],
        "first_name_y": ['Spike', 'Ridley', 'Spike', 'Stephen', 'Ridley'], "genre_x": ['Comedy', 'Action', 'Comedy', 'Drama', 'Action'],
        "role": ['Hero', 'Villain', 'Support', np.nan, 'Hero'], "prob": [0.5, 0.8, 0.3, 0.9, 0.7],
        "rank": [8.0, 7.5, np.nan, 9.0, 6.5], "gender": ['M', 'F', 'M', 'F', 'M']
    }
    train_data = pd.DataFrame(dummy_data)
    test_data = pd.DataFrame(dummy_data)
    test_data['gender'] = ['F', 'M', 'F', 'M', 'F'] # slightly different target for test

combined_data = pd.concat([train_data, test_data], ignore_index=True)

rank_median = train_data['rank'].median()
train_data['rank'].fillna(rank_median, inplace=True)
test_data['rank'].fillna(rank_median, inplace=True)

string_cols_to_impute = ["last_name_x", "name", "role"]
for col in string_cols_to_impute:
    train_data[col].fillna('Unknown', inplace=True)
    test_data[col].fillna('Unknown', inplace=True)

numerical_features_for_outliers = ["actor_id", "director_id", "movie_id", "prob", "rank"]
for col in numerical_features_for_outliers:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
    test_data = test_data[(test_data[col] >= lower_bound) & (test_data[col] <= upper_bound)]


def create_features(df):
    # (Feature: actor_movie_count) - The number of movies a specific actor has appeared in.
    # Usefulness: This feature can indicate an actor's experience or popularity, which might correlate with other attributes.
    df['actor_movie_count'] = df.groupby('actor_id')['movie_id'].transform('count')

    # (Feature: director_movie_count) - The number of movies a specific director has directed.
    # Usefulness: This measures a director's prolificacy, which could be a useful signal.
    df['director_movie_count'] = df.groupby('director_id')['movie_id'].transform('count')

    # (Feature: director_actor_collaboration) - A count of collaborations between a specific director and actor.
    # Usefulness: Frequent collaborations might indicate a specific type of film or relationship, which could be predictive.
    df['director_actor_collaboration'] = df.groupby(['director_id', 'actor_id'])['movie_id'].transform('count')
    
    # (Feature: avg_rank_per_actor) - The average rank of movies for a given actor.
    # Usefulness: This feature represents the general quality or success of movies an actor is associated with.
    df['avg_rank_per_actor'] = df.groupby('actor_id')['rank'].transform('mean')

    return df

train_data = create_features(train_data)
test_data = create_features(test_data)

TARGET = 'gender'

le = LabelEncoder()
le.fit(combined_data[TARGET].astype(str))
train_data[TARGET] = le.transform(train_data[TARGET].astype(str))
test_data[TARGET] = le.transform(test_data[TARGET].astype(str))

y_train = train_data[TARGET]
X_train = train_data.drop(columns=[TARGET])

y_test = test_data[TARGET]
X_test = test_data.drop(columns=[TARGET])

columns_to_drop = ['last_name_x', 'last_name_y', 'first_name_x', 'first_name_y', 'name', 'role']
X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

numerical_features = ["actor_id", "director_id", "movie_id", "prob", "rank", 
                      "actor_movie_count", "director_movie_count", "director_actor_collaboration", "avg_rank_per_actor"]
categorical_features = ["year", "genre_y", "genre_x"]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)


lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', lgbm)])

model_pipeline.fit(X_train, y_train)
trn = model_pipeline

train_preds_proba = model_pipeline.predict_proba(X_train)[:, 1]
test_preds_proba = model_pipeline.predict_proba(X_test)[:, 1]
train_preds = model_pipeline.predict(X_train)
test_preds = model_pipeline.predict(X_test)

Train_AUC = roc_auc_score(y_train, train_preds_proba)
Train_Accuracy = accuracy_score(y_train, train_preds)
Train_F1_score = f1_score(y_train, train_preds)

Test_AUC = roc_auc_score(y_test, test_preds_proba)
Test_Accuracy = accuracy_score(y_test, test_preds)
Test_F1_score = f1_score(y_test, test_preds)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end