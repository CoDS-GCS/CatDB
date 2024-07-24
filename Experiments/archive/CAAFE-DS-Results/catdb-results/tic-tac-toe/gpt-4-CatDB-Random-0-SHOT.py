# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/tic-tac-toe/tic-tac-toe_train.csv')
test_data = pd.read_csv('../../../data/tic-tac-toe/tic-tac-toe_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# As per the given schema, there are no missing values and all the columns are of integer type. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# As per the given schema, all the columns are of integer type. So, no encoding is required.
# ```end

# ```python
# Select the appropriate features and target variables
features = ['middle-left-square', 'top-right-square', 'bottom-middle-square', 'bottom-left-square', 'middle-right-square', 'top-middle-square', 'bottom-right-square', 'middle-middle-square', 'top-left-square']
target = 'Class'
# ```end

# ```python
# Perform feature scaling
scaler = StandardScaler()
train_data[features] = scaler.fit_transform(train_data[features])
test_data[features] = scaler.transform(test_data[features])
# ```end

# ```python
# No columns are dropped as all the columns are necessary for the prediction of the target variable 'Class'
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile algorithm that can handle both categorical and numerical features. It also handles overfitting.
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(train_data[features], train_data[target])
# ```end

# ```python
# Report evaluation based on train and test dataset
train_predictions = clf.predict(train_data[features])
test_predictions = clf.predict(test_data[features])

Train_Accuracy = accuracy_score(train_data[target], train_predictions)
Test_Accuracy = accuracy_score(test_data[target], test_predictions)

Train_F1_score = f1_score(train_data[target], train_predictions)
Test_F1_score = f1_score(test_data[target], test_predictions)

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end