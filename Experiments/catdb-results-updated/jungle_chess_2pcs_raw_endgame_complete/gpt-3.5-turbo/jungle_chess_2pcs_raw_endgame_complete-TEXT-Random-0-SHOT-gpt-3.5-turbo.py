# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import


# python-load-dataset 
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/jungle_chess_2pcs_raw_endgame_complete/jungle_chess_2pcs_raw_endgame_complete_train.csv')
test_data = pd.read_csv('data/jungle_chess_2pcs_raw_endgame_complete/jungle_chess_2pcs_raw_endgame_complete_test.csv')
# end-load-dataset

# python-added-column 
# Adding new column 'white_piece0_total_strength'
# Usefulness: This column provides the total strength of the white piece 0 by summing its strength and rank
train_data['white_piece0_total_strength'] = train_data['white_piece0_strength'] + train_data['white_piece0_rank']
test_data['white_piece0_total_strength'] = test_data['white_piece0_strength'] + test_data['white_piece0_rank']

# Adding new column 'black_piece0_total_strength'
# Usefulness: This column provides the total strength of the black piece 0 by summing its strength and rank
train_data['black_piece0_total_strength'] = train_data['black_piece0_strength'] + train_data['black_piece0_rank']
test_data['black_piece0_total_strength'] = test_data['black_piece0_strength'] + test_data['black_piece0_rank']
# end-added-column

# python-dropping-columns
# Dropping columns 'white_piece0_rank' and 'black_piece0_rank'
# Explanation: These columns are dropped as they may be redundant and hurt the predictive performance of the downstream classifier
train_data.drop(columns=['white_piece0_rank', 'black_piece0_rank'], inplace=True)
test_data.drop(columns=['white_piece0_rank', 'black_piece0_rank'], inplace=True)
# end-dropping-columns

# python-training-technique 
# Use a Random Forest Classifier for multiclass classification
# Explanation: Random Forest is selected as it is an effective algorithm for multiclass classification tasks
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 