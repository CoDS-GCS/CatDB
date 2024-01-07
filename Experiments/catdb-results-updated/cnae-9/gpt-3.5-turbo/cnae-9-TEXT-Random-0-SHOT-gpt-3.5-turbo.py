# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv("data/cnae-9/cnae-9_train.csv")
test_data = pd.read_csv("data/cnae-9/cnae-9_test.csv")
# end-load-dataset

# python-added-column
# Add semantic information by creating a new column
# Usefulness: This column captures the combination of V705 and V140, which provides additional information about the dataset.
train_data['V705_V140'] = train_data['V705'] + train_data['V140']
test_data['V705_V140'] = test_data['V705'] + test_data['V140']
# end-added-column

# python-dropping-columns
# Drop columns that may be redundant and hurt the predictive performance
# Explanation: V140 is dropped as it is already captured in the new column 'V705_V140'
train_data.drop(columns=['V140'], inplace=True)
test_data.drop(columns=['V140'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation: One-vs-Rest Classifier with Support Vector Machine (SVM) is chosen as it can handle multiclass classification efficiently.
X = train_data.drop(columns=['Class'])
y = train_data['Class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC()))
classifier.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation