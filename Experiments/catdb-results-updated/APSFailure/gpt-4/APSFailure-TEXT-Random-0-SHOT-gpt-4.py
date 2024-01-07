# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_df = pd.read_csv("data/APSFailure/APSFailure_train.csv")
test_df = pd.read_csv("data/APSFailure/APSFailure_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: sum_of_all
# Usefulness: This feature will aggregate all the numerical columns which might help in differentiating the class.
train_df = train_df.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
test_df = test_df.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

train_df['sum_of_all'] = train_df.iloc[:,1:].sum(axis=1)
test_df['sum_of_all'] = test_df.iloc[:,1:].sum(axis=1)
# end-added-column

# python-added-column
# Feature name and description: mean_of_all
# Usefulness: This feature will calculate the mean of all the numerical columns which might help in differentiating the class.
train_df['mean_of_all'] = train_df.iloc[:,1:].mean(axis=1)
test_df['mean_of_all'] = test_df.iloc[:,1:].mean(axis=1)
# end-added-column

# python-dropping-columns
# Dropping columns that are highly correlated to avoid multicollinearity which can lead to overfitting.
corr_matrix = train_df.iloc[:,1:].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
train_df.drop(columns=to_drop, inplace=True)
test_df.drop(columns=to_drop, inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# RandomForestClassifier is used here as it can handle a large number of features and it is less likely to overfit.
# Label encoding the target variable
le = LabelEncoder()
train_df['class'] = le.fit_transform(train_df['class'])
test_df['class'] = le.transform(test_df['class'])

X_train = train_df.drop(columns=['class'])
y_train = train_df['class']
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation