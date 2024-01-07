# python-import
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# end-import

# python-load-dataset 
# load train and test datasets (csv file formats) here 
train = pd.read_csv('data/jannis/jannis_train.csv')
test = pd.read_csv('data/jannis/jannis_test.csv')
# end-load-dataset 

# python-added-column 
# Feature name and description: V19_V9_sum
# Usefulness: Combining features can sometimes reveal patterns that aren't visible from the individual features.
train['V19_V9_sum'] = train['V19'] + train['V9']
test['V19_V9_sum'] = test['V19'] + test['V9']
# end-added-column

# python-added-column 
# Feature name and description: V39_V47_diff
# Usefulness: The difference between two features can sometimes be more informative than the raw features.
train['V39_V47_diff'] = train['V39'] - train['V47']
test['V39_V47_diff'] = test['V39'] - test['V47']
# end-added-column

# python-dropping-columns
# Explanation why the column V1 is dropped: If a column has a high correlation with another column, then they are providing the same information to the model, so we can drop one of them.
train.drop(columns=['V1'], inplace=True)
test.drop(columns=['V1'], inplace=True)
# end-dropping-columns

# python-training-technique 
# Use a multiclass classification technique
# Explanation why the solution is selected: Random Forest is a versatile algorithm that can handle both numerical and categorical data. It also works well with large datasets and can handle missing values and outliers.

X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation