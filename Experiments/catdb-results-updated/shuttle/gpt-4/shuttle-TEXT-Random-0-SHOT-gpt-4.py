# python-import
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv('data/shuttle/shuttle_train.csv')
test = pd.read_csv('data/shuttle/shuttle_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: 'A1_A2' - interaction between 'A1' and 'A2'
# Usefulness: This feature may capture the interaction effect between 'A1' and 'A2', which could be important for predicting 'class'.
train['A1_A2'] = train['A1'] * train['A2']
test['A1_A2'] = test['A1'] * test['A2']
# end-added-column

# python-added-column
# Feature name and description: 'A3_A4' - interaction between 'A3' and 'A4'
# Usefulness: This feature may capture the interaction effect between 'A3' and 'A4', which could be important for predicting 'class'.
train['A3_A4'] = train['A3'] * train['A4']
test['A3_A4'] = test['A3'] * test['A4']
# end-added-column

# python-dropping-columns
# Explanation why the column A7 is dropped: A7 might be correlated with other features, and therefore, might not bring any additional information to the model. It is dropped to avoid overfitting.
train.drop(columns=['A7'], inplace=True)
test.drop(columns=['A7'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique - RandomForestClassifier
# Explanation why the solution is selected: RandomForest is a popular and effective method for multiclass classification problems. It can handle interactions between features and doesn't require feature scaling.
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Create a pipeline with preprocessing and RandomForestClassifier
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the pipeline model
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation