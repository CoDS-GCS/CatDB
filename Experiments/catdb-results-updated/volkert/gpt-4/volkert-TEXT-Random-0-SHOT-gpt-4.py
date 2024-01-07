# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/volkert/volkert_train.csv")
test = pd.read_csv("data/volkert/volkert_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: V107_V110_Sum
# Usefulness: Combining features can sometimes reveal patterns that are not apparent from the individual features.
train['V107_V110_Sum'] = train['V107'] + train['V110']
test['V107_V110_Sum'] = test['V107'] + test['V110']
# end-added-column

# python-dropping-columns
# Explanation: Dropping column V107 and V110 as they are now represented in the new feature V107_V110_Sum
train.drop(columns=['V107', 'V110'], inplace=True)
test.drop(columns=['V107', 'V110'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation: RandomForest is selected for its ability to handle large datasets with high dimensionality and 
# its robustness against overfitting through ensemble learning.
X_train = train.drop(columns=['class'])
y_train = train['class']
X_test = test.drop(columns=['class'])
y_test = test['class']

# Creating a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('scaling', StandardScaler()),  # Standardizing the features
    ('pca', PCA(n_components=0.95)),  # Reducing dimensionality
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))  # Model Training
])

pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation