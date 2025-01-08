import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Load the datasets
train_data = pd.read_csv("/home/saeed/Downloads/Airline_train.csv")
test_data = pd.read_csv("/home/saeed/Downloads/Airline_test.csv")

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocessing
# 1. Drop unnecessary columns + columns with all NaN values
combined_data = combined_data.drop(['id', 'Flight_Number_Marketing_Airline', 'DOT_ID_Marketing_Airline', 'Tail_Number', 'FlightDate', 'OriginCityName', 'DestCityName', 'DepTime', 'ArrTime', 'CancellationCode', 'DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay',
 'DivDistance', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID',
 'Div1WheelsOn', 'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff',
 'Div1TailNum', 'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID',
 'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime',
 'Div1Airport_Description', 'Div2Airport_Description',
 'Div1AirportSeq_Description', 'Div2AirportSeq_Description',
 'Cancellation_Description'], axis=1, errors='ignore')

# Drop columns with all NaN values
combined_data = combined_data.dropna(axis=1, how='all')

# 2. Define categorical and numerical features
categorical_cols = combined_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = combined_data.select_dtypes(include=['number']).columns.tolist()

# Remove the target variable from numerical cols
numerical_cols.remove('ArrDel15')

# 3. Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ]), numerical_cols)
    ],
    remainder='passthrough'  # Keep other columns
)

# 4. Split back into train and test sets
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Separate features and target
X_train = train_data.drop('ArrDel15', axis=1)
y_train = train_data['ArrDel15']
X_test = test_data.drop('ArrDel15', axis=1)
y_test = test_data['ArrDel15']

# 5. Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)

# Model evaluation
# Model evaluation
y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, y_train_prob)
Test_Log_loss = log_loss(y_test, y_test_prob)

Train_AUC_OVO = roc_auc_score(y_train, y_train_prob[:,1], multi_class='ovo')
Test_AUC_OVO = roc_auc_score(y_test, y_test_prob[:,1], multi_class='ovo')

Train_AUC_OVR = roc_auc_score(y_train, y_train_prob[:,1], multi_class='ovr')
Test_AUC_OVR = roc_auc_score(y_test, y_test_prob[:,1], multi_class='ovr')


print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
