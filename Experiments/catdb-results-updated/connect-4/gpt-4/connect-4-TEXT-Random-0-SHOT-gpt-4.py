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
df_train = pd.read_csv('data/connect-4/connect-4_train.csv')
df_test = pd.read_csv('data/connect-4/connect-4_test.csv')
# end-load-dataset

# python-added-column
# Feature name: total
# Usefulness: This feature represents the total sum of all the columns. This could be useful as it might show some underlying patterns in the data.
df_train['total'] = df_train.sum(axis=1)
df_test['total'] = df_test.sum(axis=1)
# end-added-column

# python-added-column
# Feature name: average
# Usefulness: This feature represents the average value of all the columns. This could be useful as it might show some underlying patterns in the data.
df_train['average'] = df_train.mean(axis=1)
df_test['average'] = df_test.mean(axis=1)
# end-added-column

# python-dropping-columns
# Column 'e1' is dropped because it has low variance and thus does not contribute much to the final prediction.
df_train.drop(columns=['e1'], inplace=True)
df_test.drop(columns=['e1'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a RandomForestClassifier as the multiclass classification technique
# RandomForestClassifier is selected because it is a versatile algorithm that can handle both continuous and categorical data. It also has built-in feature importance which can be useful for feature selection.
X_train = df_train.drop(columns=['class'])
y_train = df_train['class']
X_test = df_test.drop(columns=['class'])
y_test = df_test['class']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation