# python-import
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here 
df_train = pd.read_csv('data/dilbert/dilbert_train.csv')
df_test = pd.read_csv('data/dilbert/dilbert_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: V732_V1272_ratio
# Usefulness: The ratio between two variables could provide additional information about their relationship and can be useful for classification.
df_train['V732_V1272_ratio'] = df_train['V732'] / df_train['V1272']
df_test['V732_V1272_ratio'] = df_test['V732'] / df_test['V1272']
# end-added-column

# python-added-column
# Feature name and description: V732_V1272_product
# Usefulness: The product of two variables can capture their joint effect on the target variable.
df_train['V732_V1272_product'] = df_train['V732'] * df_train['V1272']
df_test['V732_V1272_product'] = df_test['V732'] * df_test['V1272']
# end-added-column

# python-dropping-columns
# Explanation why the column V732 and V1272 are dropped: 
# These columns are dropped because they have been used to create new features and may not provide additional information.
df_train.drop(columns=['V732', 'V1272'], inplace=True)
df_test.drop(columns=['V732', 'V1272'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a RandomForestClassifier as a multiclass classification technique
# Explanation why the solution is selected:
# RandomForestClassifier is a powerful and flexible model that can capture complex patterns in the data. It also handles multicollinearity well, which might be present in our dataset due to the creation of new features.
X = df_train.drop('class', axis=1)
y = df_train['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(df_test.drop('class', axis=1))
accuracy = accuracy_score(df_test['class'], y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation