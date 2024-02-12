# Import all required packages

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score



# Load the training and test datasets

train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")

test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")



# Remove low ration, static, and unique columns by getting statistic values

for column in train_data.columns:

    if train_data[column].nunique() == 1 or train_data[column].nunique() == train_data.shape[0]:

        train_data.drop(columns=[column], inplace=True)

        test_data.drop(columns=[column], inplace=True)



# Feature name and description: c_53_c_76_ratio

# Usefulness: This feature represents the ratio between c_53 and c_76, which might be useful for the classification task.

train_data['c_53_c_76_ratio'] = train_data['c_53'] / train_data['c_76']

test_data['c_53_c_76_ratio'] = test_data['c_53'] / test_data['c_76']



# Replace inf and -inf with NaN and then fill NaN with 0

train_data = train_data.replace([np.inf, -np.inf], np.nan).fillna(0)

test_data = test_data.replace([np.inf, -np.inf], np.nan).fillna(0)



# Explanation why the column c_53 is dropped: After creating the ratio feature, the original columns might be redundant and cause overfitting.

# Therefore, we drop these columns.

train_data.drop(columns=['c_53', 'c_76'], inplace=True)

test_data.drop(columns=['c_53', 'c_76'], inplace=True)



# Use a RandomForestClassifier technique

# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets.

# It can handle a large number of features, and it's also less likely to overfit than other classifiers.

X_train = train_data.drop(columns=['c_1'])

y_train = train_data['c_1']

X_test = test_data.drop(columns=['c_1'])

y_test = test_data['c_1']



clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

clf.fit(X_train, y_train)



# Report evaluation based on only test dataset

y_pred = clf.predict(X_test)



# Calculate the model accuracy

Accuracy = accuracy_score(y_test, y_pred)



# Calculate the model f1 score

F1_score = f1_score(y_test, y_pred)



# Print the accuracy result

print(f"Accuracy:{Accuracy}")



# Print the f1 score result

print(f"F1_score:{F1_score}")