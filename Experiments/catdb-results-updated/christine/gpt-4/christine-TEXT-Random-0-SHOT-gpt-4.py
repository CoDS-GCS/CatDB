# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here 
df_train = pd.read_csv('data/christine/christine_train.csv')
df_test = pd.read_csv('data/christine/christine_test.csv')
# end-load-dataset

# python-added-column
# Feature name: V190_V838_diff
# Usefulness: The difference between V190 and V838 might capture some pattern that helps to classify 'class'.
df_train['V190_V838_diff'] = df_train['V190'] - df_train['V838']
df_test['V190_V838_diff'] = df_test['V190'] - df_test['V838']
# end-added-column

# python-added-column
# Feature name: V464_V1124_sum
# Usefulness: The sum of V464 and V1124 might capture some pattern that helps to classify 'class'.
df_train['V464_V1124_sum'] = df_train['V464'] + df_train['V1124']
df_test['V464_V1124_sum'] = df_test['V464'] + df_test['V1124']
# end-added-column

# python-added-column
# Feature name: V945_V868_ratio
# Usefulness: The ratio of V945 to V868 might capture some pattern that helps to classify 'class'.
df_train['V945_V868_ratio'] = df_train['V945'] / df_train['V868']
df_test['V945_V868_ratio'] = df_test['V945'] / df_test['V868']
# end-added-column

# python-dropping-columns
# Explanation why the column V10 is dropped: V10 might be highly correlated with other features, which can lead to multicollinearity in the model.
df_train.drop(columns=['V10'], inplace=True)
df_test.drop(columns=['V10'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that can handle a large number of features and is not prone to overfitting.
X_train = df_train.drop(columns=['class'])
y_train = df_train['class']
X_test = df_test.drop(columns=['class'])
y_test = df_test['class']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation