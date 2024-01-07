# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset 
# load train and test datasets (csv file formats) here 
train_data = pd.read_csv('data/helena/helena_train.csv')
test_data = pd.read_csv('data/helena/helena_test.csv')
# end-load-dataset 

# python-added-column 
# Feature name and description: V6_to_V21_ratio
# Usefulness: This feature might capture some interaction between V6 and V21 that could be useful for predicting 'class'.
train_data['V6_to_V21_ratio'] = train_data['V6'] / train_data['V21']
test_data['V6_to_V21_ratio'] = test_data['V6'] / test_data['V21']
# end-added-column

# python-added-column 
# Feature name and description: V18_to_V1_ratio
# Usefulness: This feature might capture some interaction between V18 and V1 that could be useful for predicting 'class'.
train_data['V18_to_V1_ratio'] = train_data['V18'] / train_data['V1']
test_data['V18_to_V1_ratio'] = test_data['V18'] / test_data['V1']
# end-added-column

# python-dropping-columns
# Explanation why the column V26 is dropped: Suppose V26 has a high correlation with other features such as V6 and V21, 
# then it may not add much information for the classifier. Also, dropping it can help reduce the dimensionality and potential overfitting.
train_data.drop(columns=['V26'], inplace=True)
test_data.drop(columns=['V26'], inplace=True)
# end-dropping-columns

# python-training-technique 
# Use a multiclass classification technique
# Explanation why the solution is selected: RandomForestClassifier is a versatile and widely used algorithm that can handle multiclass classification problems. 
# It is also less likely to overfit compared to other algorithms due to its ensemble nature.

# Split the data into features and target
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation