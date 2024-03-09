# Import all required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('/home/saeed/Documents/Github/CatDB/Experiments/data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('/home/saeed/Documents/Github/CatDB/Experiments/data/dataset_2_rnc/dataset_2_rnc_test.csv')

# Drop the column 'c_19' from the train and test datasets
train_data.drop(columns=['c_19'], inplace=True)
test_data.drop(columns=['c_19'], inplace=True)

# Impute missing values for the columns 'c_14' and 'c_10' in the train and test datasets
imputer = SimpleImputer(strategy='most_frequent')
train_data[['c_14', 'c_10']] = imputer.fit_transform(train_data[['c_14', 'c_10']])
test_data[['c_14', 'c_10']] = imputer.transform(test_data[['c_14', 'c_10']])

# Scale the numerical columns 'c_18', 'c_13', 'c_16', 'c_11', 'c_2', 'c_5', 'c_8' in the train and test datasets
scaler = MinMaxScaler()
train_data[['c_18', 'c_13', 'c_16', 'c_11', 'c_2', 'c_5', 'c_8']] = scaler.fit_transform(train_data[['c_18', 'c_13', 'c_16', 'c_11', 'c_2', 'c_5', 'c_8']])
test_data[['c_18', 'c_13', 'c_16', 'c_11', 'c_2', 'c_5', 'c_8']] = scaler.transform(test_data[['c_18', 'c_13', 'c_16', 'c_11', 'c_2', 'c_5', 'c_8']])

# Encode categorical values by dummyEncode for the columns 'c_18', 'c_16', 'c_11', 'c_8', 'c_15', 'c_3', 'c_12', 'c_14', 'c_9', 'c_10', 'c_7', 'c_6', 'c_20', 'c_17', 'c_1' in the train and test datasets

categorical_columns = ['c_15', 'c_3', 'c_12', 'c_14', 'c_9', 'c_10', 'c_7', 'c_6', 'c_20', 'c_17', 'c_1']
categorical_features = train_data.select_dtypes(include=['object']).columns

encoder = OneHotEncoder(handle_unknown='ignore')
for column in categorical_columns:
    # train_data[column] = encoder.fit_transform(train_data[column].astype(str))
    # test_data[column] = encoder.transform(test_data[column].astype(str))

    enc_df = pd.DataFrame(encoder.fit_transform(train_data[[column]]).toarray())
    train_data = train_data.join(enc_df)
    
    enc_df = pd.DataFrame(encoder.fit_transform(test_data[[column]]).toarray())
    train_data = test_data.join(enc_df)
    

encoder = LabelEncoder()
for column in categorical_features:
    if column in categorical_columns:
        continue
    train_data[column] = encoder.fit_transform(train_data[column].astype(str))
    test_data[column] = encoder.transform(test_data[column].astype(str))


# Select the appropriate features and target variables for the question
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']

# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both numerical and categorical data, and it also has methods for balancing error in class populations.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")