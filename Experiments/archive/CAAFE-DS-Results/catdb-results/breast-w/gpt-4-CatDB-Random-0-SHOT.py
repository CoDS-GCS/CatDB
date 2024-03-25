# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/breast-w/breast-w_train.csv')
test_data = pd.read_csv('../../../data/breast-w/breast-w_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Here we assume that the data is clean and does not contain any missing or erroneous values.
# If there were any, we would need to handle them appropriately (e.g., by removing rows with missing values or filling them with mean/median/mode, etc.)
# ```end

# ```python
# Perform feature processing
# Encode categorical values by dummyEncode
# Here we assume that all the columns are numerical and do not need encoding.
# If there were any categorical columns, we would need to encode them using one-hot encoding or label encoding, etc.
# ```end

# ```python
# Select the appropriate features and target variables for the question
features = ['Single_Epi_Cell_Size', 'Marginal_Adhesion', 'Bare_Nuclei', 'Mitoses', 'Bland_Chromatin', 'Normal_Nucleoli', 'Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity']
target = 'Class'

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]
# ```end

# ```python
# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# Here we use RandomForestClassifier as it is a powerful and versatile machine learning algorithm capable of performing both regression and classification tasks. 
# It is also one of the most used algorithms, because of its simplicity and the fact that it can be used for both classification and regression tasks. 
# In this case, we are using it for classification.
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end