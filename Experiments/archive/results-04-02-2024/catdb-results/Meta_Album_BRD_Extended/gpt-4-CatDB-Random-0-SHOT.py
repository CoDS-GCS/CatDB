# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Meta_Album_BRD_Extended/Meta_Album_BRD_Extended_train.csv')
test_data = pd.read_csv('../../../data/Meta_Album_BRD_Extended/Meta_Album_BRD_Extended_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Drop the 'SUPER_CATEGORY' column as it is not needed
train_data.drop(columns=['SUPER_CATEGORY'], inplace=True)
test_data.drop(columns=['SUPER_CATEGORY'], inplace=True)
# ```end

# ```python
# Perform feature processing
# Encode all "object" columns by dummyEncode
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

train_data = dummyEncode(train_data)
test_data = dummyEncode(test_data)
# ```end

# ```python
# Select the appropriate features and target variables for the question
# The target variable is 'CATEGORY'
X_train = train_data.drop('CATEGORY', axis=1)
y_train = train_data['CATEGORY']

X_test = test_data.drop('CATEGORY', axis=1)
y_test = test_data['CATEGORY']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (regressor)
# We choose RandomForestRegressor as it is a versatile and widely used algorithm that can handle both categorical and numerical features
# It also has the advantage of being able to measure the relative importance of each feature
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on train and test dataset
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

Train_R_Squared = r2_score(y_train, y_train_pred)
Test_R_Squared = r2_score(y_test, y_test_pred)

Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")   
print(f"Train_RMSE:{Train_RMSE}") 
print(f"Test_R_Squared:{Test_R_Squared}")   
print(f"Test_RMSE:{Test_RMSE}") 
# ```end