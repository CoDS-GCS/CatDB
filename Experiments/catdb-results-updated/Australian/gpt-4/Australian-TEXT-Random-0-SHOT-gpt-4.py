# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# end-import

# python-load-dataset 
# load train and test datasets (csv file formats) here 
train = pd.read_csv('data/Australian/Australian_train.csv')
test = pd.read_csv('data/Australian/Australian_test.csv')
# end-load-dataset 

# python-added-column 
# Feature name and description: A6_A13_ratio 
# Usefulness: This feature represents the ratio of A6 to A13, which could be an important factor in predicting 'A15' depending on the real-world context of these features.
train['A6_A13_ratio'] = train['A6'] / train['A13']
test['A6_A13_ratio'] = test['A6'] / test['A13']
# end-added-column

# python-added-column 
# Feature name and description: A2_A14_ratio 
# Usefulness: This feature represents the ratio of A2 to A14, which could be an important factor in predicting 'A15' depending on the real-world context of these features.
train['A2_A14_ratio'] = train['A2'] / train['A14']
test['A2_A14_ratio'] = test['A2'] / test['A14']
# end-added-column

# python-added-column 
# Feature name and description: A3_A12_ratio 
# Usefulness: This feature represents the ratio of A3 to A12, which could be an important factor in predicting 'A15' depending on the real-world context of these features.
train['A3_A12_ratio'] = train['A3'] / train['A12']
test['A3_A12_ratio'] = test['A3'] / test['A12']
# end-added-column

# python-dropping-columns
# Explanation why the column A5 is dropped
# A5 may be dropped because it may not provide any significant information for predicting 'A15'. This is based on the assumption that A5 is not strongly correlated with 'A15'. 
train.drop(columns=['A5'], inplace=True)
test.drop(columns=['A5'], inplace=True)
# end-dropping-columns

# python-training-technique 
# Use a binary classification technique
# Explanation why the solution is selected 
# RandomForestClassifier is chosen because it usually provides high accuracy and works well for binary classification problems. It also handles overfitting.
X_train = train.drop(columns=['A15'])
y_train = train['A15']
X_test = test.drop(columns=['A15'])
y_test = test['A15']

# Scaling the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the model
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation