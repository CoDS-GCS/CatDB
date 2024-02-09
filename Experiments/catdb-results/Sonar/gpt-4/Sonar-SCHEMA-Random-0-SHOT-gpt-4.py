# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/Sonar/Sonar_train.csv")
test_data = pd.read_csv("data/Sonar/Sonar_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.nunique() > 1]
# ```end

# ```python
# Drop columns that may be redundant and hurt the predictive performance of the downstream classifier
# Explanation: Columns with high correlation can lead to overfitting of the model. Hence, we drop one of them.
correlated_features = set()
correlation_matrix = train_data.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

train_data.drop(columns=correlated_features, inplace=True)
test_data.drop(columns=correlated_features, inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end