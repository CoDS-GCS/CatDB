# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/blood-transfusion-service-center/blood-transfusion-service-center_train.csv")
test = pd.read_csv("data/blood-transfusion-service-center/blood-transfusion-service-center_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: V1_V2_ratio - the ratio of V1 to V2
# Usefulness: This feature could capture the relationship between V1 and V2 that might be useful for predicting 'Class'.
train['V1_V2_ratio'] = train['V1'] / train['V2']
test['V1_V2_ratio'] = test['V1'] / test['V2']
# end-added-column

# python-added-column
# Feature name and description: V3_V4_ratio - the ratio of V3 to V4
# Usefulness: This feature could capture the relationship between V3 and V4 that might be useful for predicting 'Class'.
train['V3_V4_ratio'] = train['V3'] / train['V4']
test['V3_V4_ratio'] = test['V3'] / test['V4']
# end-added-column

# python-dropping-columns
# Explanation: V1 and V2 are dropped because the ratio V1_V2_ratio has been created which captures their relationship. Similarly, V3 and V4 are dropped because the ratio V3_V4_ratio has been created.
train.drop(columns=['V1', 'V2', 'V3', 'V4'], inplace=True)
test.drop(columns=['V1', 'V2', 'V3', 'V4'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation: RandomForestClassifier is selected because it can handle both categorical and numerical data and it has a good performance on a variety of datasets.
X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_test = test.drop('Class', axis=1)
y_test = test['Class']

# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation