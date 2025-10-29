# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

categorical_features = ["c_3", "c_8", "c_1", "c_7", "c_4", "c_9", "c_6", "c_5", "c_2"]
target_column = "c_10"




def create_board_summary_features(df):
    """
    Engineers features based on the count of each type of piece on the board.
    """
    # (num_x: Count of 'x' pieces)
    # Usefulness: This feature provides a high-level summary of the game state. The number of 'x's on the board is a fundamental aspect of a tic-tac-toe game, indicating game progression and which player's turn it might be. This helps the model learn patterns related to board density and player advantage.
    df['num_x'] = df[categorical_features].apply(lambda row: (row == 'x').sum(), axis=1)

    # (num_o: Count of 'o' pieces)
    # Usefulness: Similar to num_x, this feature quantifies the opponent's presence on the board. The relative count of 'x's and 'o's is critical for determining the game's state and predicting the outcome, as it directly relates to the rules of tic-tac-toe.
    df['num_o'] = df[categorical_features].apply(lambda row: (row == 'o').sum(), axis=1)

    # (num_b: Count of blank spaces)
    # Usefulness: The number of blank spaces indicates how close the game is to completion. An endgame with few blanks has very different strategic implications than an early game with many blanks. This feature helps the model differentiate between various stages of the game.
    df['num_b'] = df[categorical_features].apply(lambda row: (row == 'b').sum(), axis=1)
    
    return df

train_data = create_board_summary_features(train_data)
test_data = create_board_summary_features(test_data)

engineered_numerical_features = ['num_x', 'num_o', 'num_b']

target_encoder = LabelEncoder()
target_encoder.fit(combined_data[target_column])

X_train = train_data.drop(columns=[target_column])
y_train = target_encoder.transform(train_data[target_column])

X_test = test_data.drop(columns=[target_column])
y_test = target_encoder.transform(test_data[target_column])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', engineered_numerical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])


pipeline.fit(X_train, y_train)


y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]

y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

Train_AUC = roc_auc_score(y_train, y_train_proba)
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)

Test_AUC = roc_auc_score(y_test, y_test_proba)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end