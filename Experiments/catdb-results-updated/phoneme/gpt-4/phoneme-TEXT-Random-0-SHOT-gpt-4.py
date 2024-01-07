# python-import
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
df_train = pd.read_csv('data/phoneme/phoneme_train.csv')
df_test = pd.read_csv('data/phoneme/phoneme_test.csv')
# end-load-dataset

# python-added-column
# Feature name: V5_V3_ratio
# Usefulness: The ratio of V5 and V3 might have some correlation with the class label.
df_train['V5_V3_ratio'] = df_train['V5'] / df_train['V3']
df_test['V5_V3_ratio'] = df_test['V5'] / df_test['V3']
# end-added-column

# python-added-column
# Feature name: V1_V2_diff
# Usefulness: The difference between V1 and V2 might have some correlation with the class label.
df_train['V1_V2_diff'] = df_train['V1'] - df_train['V2']
df_test['V1_V2_diff'] = df_test['V1'] - df_test['V2']
# end-added-column

# python-dropping-columns
# Dropping 'V1' and 'V2' as they have been used to create a new feature and might not be necessary anymore
df_train.drop(columns=['V1', 'V2'], inplace=True)
df_test.drop(columns=['V1', 'V2'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# RandomForestClassifier is chosen because it can handle both numerical and categorical data, and it is less prone to overfitting.
X_train = df_train.drop('Class', axis=1)
y_train = df_train['Class']
X_test = df_test.drop('Class', axis=1)
y_test = df_test['Class']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation