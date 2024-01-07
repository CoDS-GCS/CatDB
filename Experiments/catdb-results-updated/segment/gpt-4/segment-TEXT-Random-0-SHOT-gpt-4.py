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
train = pd.read_csv('data/segment/segment_train.csv')
test = pd.read_csv('data/segment/segment_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: color_intensity_ratio
# Usefulness: The ratio of the mean intensities of different color channels can provide useful information about the color distribution in the image, which can be useful for classifying 'class'.
train['color_intensity_ratio'] = train['rawred.mean'] / (train['rawgreen.mean'] + train['rawblue.mean'] + 1e-8)
test['color_intensity_ratio'] = test['rawred.mean'] / (test['rawgreen.mean'] + test['rawblue.mean'] + 1e-8)
# end-added-column

# python-added-column
# Feature name and description: edge_density_ratio
# Usefulness: The ratio of the mean edge densities can provide useful information about the texture of the image, which can be useful for classifying 'class'.
train['edge_density_ratio'] = train['vedge.mean'] / (train['hedge.mean'] + 1e-8)
test['edge_density_ratio'] = test['vedge.mean'] / (test['hedge.mean'] + 1e-8)
# end-added-column

# python-dropping-columns
# We drop the 'vegde.sd' and 'hedge.sd' columns as they are standard deviations of the 'vedge.mean' and 'hedge.mean' respectively. 
# These may not provide additional useful information for the classifier and may lead to overfitting.
train.drop(columns=['vegde.sd', 'hedge.sd'], inplace=True)
test.drop(columns=['vegde.sd', 'hedge.sd'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# We use the RandomForestClassifier as it is a powerful and versatile model that performs well on many tasks. 
# It can handle both binary and multiclass classification problems, and it also provides feature importances which can be useful for interpretability.
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Create a pipeline with StandardScaler, PCA and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation