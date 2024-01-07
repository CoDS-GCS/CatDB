# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/segment/segment_train.csv')
test_data = pd.read_csv('data/segment/segment_test.csv')
# end-load-dataset

# python-added-column
# Added column: 'mean_red_green_ratio'
# Usefulness: This column represents the ratio of mean red value to mean green value, which can provide information about the color balance in the image.
train_data['mean_red_green_ratio'] = train_data['rawred.mean'] / train_data['rawgreen.mean']
test_data['mean_red_green_ratio'] = test_data['rawred.mean'] / test_data['rawgreen.mean']

# Added column: 'mean_blue_green_ratio'
# Usefulness: This column represents the ratio of mean blue value to mean green value, which can provide information about the color balance in the image.
train_data['mean_blue_green_ratio'] = train_data['rawblue.mean'] / train_data['rawgreen.mean']
test_data['mean_blue_green_ratio'] = test_data['rawblue.mean'] / test_data['rawgreen.mean']

# Added column: 'line_density_ratio'
# Usefulness: This column represents the ratio of short line density at 5 pixels to short line density at 2 pixels, which can provide information about the line patterns in the image.
train_data['line_density_ratio'] = train_data['short.line.density.5'] / train_data['short.line.density.2']
test_data['line_density_ratio'] = test_data['short.line.density.5'] / test_data['short.line.density.2']
# end-added-column

# python-dropping-columns
# Dropping column: 'rawred.mean'
# Explanation: 'rawred.mean' is dropped as it is redundant with 'mean_red_green_ratio' and 'mean_blue_green_ratio' columns.
train_data.drop(columns=['rawred.mean'], inplace=True)
test_data.drop(columns=['rawred.mean'], inplace=True)

# Dropping column: 'rawgreen.mean'
# Explanation: 'rawgreen.mean' is dropped as it is redundant with 'mean_red_green_ratio' and 'mean_blue_green_ratio' columns.
train_data.drop(columns=['rawgreen.mean'], inplace=True)
test_data.drop(columns=['rawgreen.mean'], inplace=True)

# Dropping column: 'rawblue.mean'
# Explanation: 'rawblue.mean' is dropped as it is redundant with 'mean_blue_green_ratio' column.
train_data.drop(columns=['rawblue.mean'], inplace=True)
test_data.drop(columns=['rawblue.mean'], inplace=True)

# Dropping column: 'short.line.density.2'
# Explanation: 'short.line.density.2' is dropped as it is redundant with 'line_density_ratio' column.
train_data.drop(columns=['short.line.density.2'], inplace=True)
test_data.drop(columns=['short.line.density.2'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a Random Forest Classifier for multiclass classification
# Explanation: Random Forest Classifier is selected as it is known for its ability to handle high-dimensional datasets and capture complex relationships between features.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation
# 