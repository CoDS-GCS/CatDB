# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/mfeat-factors/mfeat-factors_train.csv')
test_data = pd.read_csv('data/mfeat-factors/mfeat-factors_test.csv')
# end-load-dataset

# python-added-column
# Add new columns based on real-world knowledge
train_data['att186_plus_att60'] = train_data['att186'] + train_data['att60']
train_data['att84_times_att67'] = train_data['att84'] * train_data['att67']
train_data['att71_div_att105'] = train_data['att71'] / train_data['att105']
train_data['att134_minus_att178'] = train_data['att134'] - train_data['att178']
train_data['att50_plus_att89'] = train_data['att50'] + train_data['att89']

test_data['att186_plus_att60'] = test_data['att186'] + test_data['att60']
test_data['att84_times_att67'] = test_data['att84'] * test_data['att67']
test_data['att71_div_att105'] = test_data['att71'] / test_data['att105']
test_data['att134_minus_att178'] = test_data['att134'] - test_data['att178']
test_data['att50_plus_att89'] = test_data['att50'] + test_data['att89']
# end-added-column

# python-dropping-columns
# Drop columns that may be redundant
train_data.drop(columns=['att186', 'att60', 'att84', 'att67', 'att71', 'att105', 'att134', 'att178', 'att50', 'att89'], inplace=True)
test_data.drop(columns=['att186', 'att60', 'att84', 'att67', 'att71', 'att105', 'att134', 'att178', 'att50', 'att89'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for multiclass classification
# Random Forest Classifier is selected as it can handle multiple classes and is robust to noisy data
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