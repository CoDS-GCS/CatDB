# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_df = pd.read_csv("data/mfeat-factors/mfeat-factors_train.csv")
test_df = pd.read_csv("data/mfeat-factors/mfeat-factors_test.csv")
# end-load-dataset

# python-added-column
# Feature name: 'att186_60_sum'
# Usefulness: This feature adds useful real world knowledge to classify 'class' by combining 'att186' and 'att60' which might have a combined effect on the target variable.
train_df['att186_60_sum'] = train_df['att186'] + train_df['att60']
test_df['att186_60_sum'] = test_df['att186'] + test_df['att60']
# end-added-column

# python-added-column
# Feature name: 'att84_67_diff'
# Usefulness: This feature adds useful real world knowledge to classify 'class' by finding the difference between 'att84' and 'att67' which might have an inverse effect on the target variable.
train_df['att84_67_diff'] = train_df['att84'] - train_df['att67']
test_df['att84_67_diff'] = test_df['att84'] - test_df['att67']
# end-added-column

# python-dropping-columns
# Dropping 'att186' and 'att60' as they are now represented in the 'att186_60_sum' feature
# This helps to reduce redundancy and overfitting
train_df.drop(columns=['att186', 'att60'], inplace=True)
test_df.drop(columns=['att186', 'att60'], inplace=True)
# end-dropping-columns

# python-dropping-columns
# Dropping 'att84' and 'att67' as they are now represented in the 'att84_67_diff' feature
# This helps to reduce redundancy and overfitting
train_df.drop(columns=['att84', 'att67'], inplace=True)
test_df.drop(columns=['att84', 'att67'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Random Forest Classifier is chosen because it handles multiclass classification problems well and reduces the chance of overfitting by averaging the results of many decision trees.
# It also handles large datasets with many features effectively.
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df.drop('class', axis=1)
y_test = test_df['class']

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation