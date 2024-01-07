# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here 
train_data = pd.read_csv('data/Fashion-MNIST/Fashion-MNIST_train.csv')
test_data = pd.read_csv('data/Fashion-MNIST/Fashion-MNIST_test.csv')
# end-load-dataset

# python-added-column
# Added column: pixel_sum
# Usefulness: This column calculates the sum of all pixel values for each row, providing an aggregate measure of the image's overall intensity.
train_data['pixel_sum'] = train_data.iloc[:, 10:].sum(axis=1)
test_data['pixel_sum'] = test_data.iloc[:, 10:].sum(axis=1)

# Added column: pixel_mean
# Usefulness: This column calculates the mean of all pixel values for each row, providing an average measure of the image's intensity.
train_data['pixel_mean'] = train_data.iloc[:, 10:].mean(axis=1)
test_data['pixel_mean'] = test_data.iloc[:, 10:].mean(axis=1)

# Added column: pixel_std
# Usefulness: This column calculates the standard deviation of all pixel values for each row, providing a measure of the image's variation in intensity.
train_data['pixel_std'] = train_data.iloc[:, 10:].std(axis=1)
test_data['pixel_std'] = test_data.iloc[:, 10:].std(axis=1)
# end-added-column

# python-dropping-columns
# Dropping column: pixel30
# Explanation: pixel30 is dropped as it is not contributing significantly to the prediction of the target class.
train_data.drop(columns=['pixel30'], inplace=True)
test_data.drop(columns=['pixel30'], inplace=True)

# Dropping column: pixel779
# Explanation: pixel779 is dropped as it is not contributing significantly to the prediction of the target class.
train_data.drop(columns=['pixel779'], inplace=True)
test_data.drop(columns=['pixel779'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a Random Forest Classifier for multiclass classification
# Explanation: Random Forest is selected as it is an ensemble method that combines multiple decision trees to improve prediction accuracy.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

# Train the Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation