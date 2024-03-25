# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
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
# Impute missing values in 'Bare_Nuclei' column with median
imputer = SimpleImputer(strategy='median')
train_data['Bare_Nuclei'] = imputer.fit_transform(train_data['Bare_Nuclei'].values.reshape(-1,1))
test_data['Bare_Nuclei'] = imputer.transform(test_data['Bare_Nuclei'].values.reshape(-1,1))
# ```end

# ```python
# Perform feature processing
# Scale numerical columns
scaler = MinMaxScaler()
numerical_cols = ['Clump_Thickness', 'Mitoses', 'Single_Epi_Cell_Size', 'Normal_Nucleoli', 'Cell_Shape_Uniformity', 'Cell_Size_Uniformity', 'Bare_Nuclei', 'Marginal_Adhesion', 'Bland_Chromatin']
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

# Encode categorical columns
encoder = LabelEncoder()
train_data['Class'] = encoder.fit_transform(train_data['Class'])
test_data['Class'] = encoder.transform(test_data['Class'])
# ```end

# ```python
# Select the appropriate features and target variables
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile and widely used algorithm that can handle both numerical and categorical data, and it also has a good performance in general.
clf = RandomForestClassifier(n_jobs=-1, random_state=42)
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