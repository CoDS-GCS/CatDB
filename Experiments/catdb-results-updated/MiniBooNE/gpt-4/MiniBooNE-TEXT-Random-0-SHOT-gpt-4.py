# python-import
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/MiniBooNE/MiniBooNE_train.csv")
test = pd.read_csv("data/MiniBooNE/MiniBooNE_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: 'ParticleID_13_40_ratio'
# Usefulness: This ratio might provide useful information about the relationship between the two particles.
train['ParticleID_13_40_ratio'] = train['ParticleID_13'] / train['ParticleID_40']
test['ParticleID_13_40_ratio'] = test['ParticleID_13'] / test['ParticleID_40']
# end-added-column

# python-added-column
# Feature name and description: 'ParticleID_34_46_diff'
# Usefulness: The difference between these two particles might be a useful feature.
train['ParticleID_34_46_diff'] = train['ParticleID_34'] - train['ParticleID_46']
test['ParticleID_34_46_diff'] = test['ParticleID_34'] - test['ParticleID_46']
# end-added-column

# python-dropping-columns
# Dropping 'ParticleID_0' as it might not be contributing to the target variable 'signal' 
train.drop(columns=['ParticleID_0'], inplace=True)
test.drop(columns=['ParticleID_0'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# RandomForestClassifier is chosen because it is a robust and versatile classifier that can handle both numerical and categorical data, 
# and it is less likely to overfit than other classifiers.

X_train = train.drop('signal', axis=1)
y_train = train['signal']
X_test = test.drop('signal', axis=1)
y_test = test['signal']

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_accuracy * 100:.2f}")
# end-evaluation