# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_3_rnc/dataset_3_rnc_train.csv')
test_data = pd.read_csv('data/dataset_3_rnc/dataset_3_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
for col in train_data.columns:
    if len(train_data[col].unique()) < 2:
        train_data.drop(col,inplace=True,axis=1)
# ```end

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'c_1' according to dataset description and attributes.) 
# (Some pandas code using 'c_53', 'c_76', ... to add a new column for each row in df)
# Here we can add new features or modify existing ones based on the dataset description and attributes. 
# As the schema and categorical data does not provide specific information about the columns, we cannot add or modify features here.
# ```end

# ```python-dropping-columns
# Explanation why the column XX is dropped
# df.drop(columns=['XX'], inplace=True)
# Here we can drop columns that are not useful for the prediction of 'c_1'. 
# As the schema and categorical data does not provide specific information about the columns, we cannot drop any columns here.
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets. It can handle a large number of features and it's not prone to overfitting.
X_train = train_data.drop('c_1', axis=1)
y_train = train_data['c_1']
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
X_test = test_data.drop('c_1', axis=1)
y_test = test_data['c_1']
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   

# Print the f1 score result
print(f"F1_score:{F1_score}") 
# ```end