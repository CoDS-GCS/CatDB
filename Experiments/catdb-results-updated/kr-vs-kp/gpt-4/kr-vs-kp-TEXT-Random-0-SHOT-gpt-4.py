# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/kr-vs-kp/kr-vs-kp_train.csv")
test = pd.read_csv("data/kr-vs-kp/kr-vs-kp_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: text_length
# Usefulness: The length of the text in each column might be useful for the classification task.
for column in train.columns:
    if column != 'class':
        train[column+'_len'] = train[column].apply(len)
        test[column+'_len'] = test[column].apply(len)
# end-added-column

# python-dropping-columns
# Explanation why the column XX is dropped
# No columns are dropped in this example as all columns are used for feature generation.
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation: RandomForestClassifier is chosen because it is a versatile algorithm that can handle both numerical and categorical data and perform well on large datasets. It also has built-in feature importance which can be useful for understanding the model.

# Separate features and target
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Create a pipeline for text and numerical features
text_features = [column for column in X_train.columns if column.endswith('_len')]
num_features = [column for column in X_train.columns if column not in text_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('text', CountVectorizer(), text_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Train the model
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
# end-evaluation
