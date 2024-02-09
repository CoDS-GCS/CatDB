# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the training and test datasets
train_data = pd.read_csv("data/credit-g/credit-g_train.csv")
test_data = pd.read_csv("data/credit-g/credit-g_test.csv")

# Convert categorical columns to numerical values
le = LabelEncoder()
categorical_columns = ['c_8', 'c_11', 'c_18', 'c_16', 'c_17', 'c_14', 'c_21', 'c_20', 'c_9', 'c_7', 'c_10', 'c_6', 'c_12', 'c_19', 'c_15', 'c_3', 'c_1']

for column in categorical_columns:
    train_data[column] = train_data[column].astype(str)
    test_data[column] = test_data[column].astype(str)
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])

# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if len(train_data[column].unique()) == 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)

# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it can handle both categorical and numerical data, and it performs well on large datasets.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the model
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
clf.fit(X_train, y_train)

# Test the model
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
y_pred = clf.predict(X_test)

# Report evaluation based on only test dataset
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
