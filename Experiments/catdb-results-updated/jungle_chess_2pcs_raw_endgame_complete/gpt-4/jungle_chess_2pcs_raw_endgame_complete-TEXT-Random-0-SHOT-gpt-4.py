# python-import
# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_df = pd.read_csv("data/jungle_chess_2pcs_raw_endgame_complete/jungle_chess_2pcs_raw_endgame_complete_train.csv")
test_df = pd.read_csv("data/jungle_chess_2pcs_raw_endgame_complete/jungle_chess_2pcs_raw_endgame_complete_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: Total_strength
# Usefulness: This feature combines the strength of the white and black pieces. This could be a useful feature as the difference in strength between the pieces could influence the outcome of the game.
train_df['total_strength'] = train_df['white_piece0_strength'] + train_df['black_piece0_strength']
test_df['total_strength'] = test_df['white_piece0_strength'] + test_df['black_piece0_strength']
# end-added-column

# python-added-column
# Feature name and description: Total_rank
# Usefulness: This feature combines the rank of the white and black pieces. This could be a useful feature as the difference in rank between the pieces could influence the outcome of the game.
train_df['total_rank'] = train_df['white_piece0_rank'] + train_df['black_piece0_rank']
test_df['total_rank'] = test_df['white_piece0_rank'] + test_df['black_piece0_rank']
# end-added-column

# python-dropping-columns
# Explanation why the column white_piece0_file and black_piece0_file are dropped
# These columns are dropped as they represent the file where the piece is located, which may not be directly related to the outcome of the game.
train_df.drop(columns=['white_piece0_file', 'black_piece0_file'], inplace=True)
test_df.drop(columns=['white_piece0_file', 'black_piece0_file'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation why the solution is selected: RandomForestClassifier is a versatile and widely used algorithm that can handle multiclass classification problems. It is also less prone to overfitting.
# Encoding the target variable
le = LabelEncoder()
train_df['class'] = le.fit_transform(train_df['class'])
test_df['class'] = le.transform(test_df['class'])

X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df.drop('class', axis=1)
y_test = test_df['class']

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy =  accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation