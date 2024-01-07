# python-import
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# end-import

# python-load-dataset 
# load train and test datasets (csv file formats) here 
df_train = pd.read_csv("data/Fashion-MNIST/Fashion-MNIST_train.csv")
df_test = pd.read_csv("data/Fashion-MNIST/Fashion-MNIST_test.csv")
# end-load-dataset 

# python-added-column 
# Feature name and description: 'sum_pixels' - the sum of all pixel values for a given image
# Usefulness: The sum of all pixel values could provide a measure of the overall brightness of the image, 
# which could potentially be a useful feature for classifying the 'class' of the image.
df_train['sum_pix'] = df_train.sum(axis=1)
df_test['sum_pix'] = df_test.sum(axis=1)
# end-added-column

# python-dropping-columns
# Drop columns that are not needed for the model. In this case, we will drop the original pixel columns 
# as we have created a new feature that aggregates the information contained in these columns.
# pixel_columns = df_train.columns[df_train.columns.str.contains('pixel')]
# df_train.drop(columns=pixel_columns, inplace=True)
# df_test.drop(columns=pixel_columns, inplace=True)
# end-dropping-columns

# python-training-technique 
# Use a multiclass classification technique
# RandomForestClassifier is selected because it is a powerful and versatile machine learning model 
# that performs well on many types of data, and it can handle multiclass classification tasks.

# Split the data into features and target variable

X_train = df_train.drop(columns=['class'])
y_train = df_train['class']

X_test = df_test.drop(columns=['class'])
y_test = df_test['class']

# Create a pipeline that first reduces the dimensionality using PCA and then fits a RandomForestClassifier
pipeline = Pipeline(steps=[('scl', StandardScaler()), 
                           ('pca', PCA(n_components=10)),
                           ('clf', RandomForestClassifier())])


# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation