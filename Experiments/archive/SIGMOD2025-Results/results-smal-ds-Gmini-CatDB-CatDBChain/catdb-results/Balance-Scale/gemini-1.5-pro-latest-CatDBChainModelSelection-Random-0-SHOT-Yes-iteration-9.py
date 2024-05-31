# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from threading import Thread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

def preprocess_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Feature Engineering: Calculate the total moments on each side
    df['left_moment'] = df['left-weight'] * df['left-distance']
    df['right_moment'] = df['right-weight'] * df['right-distance']

    # Define categorical features for one-hot encoding
    categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a pipeline for data preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Fit and transform the data
    transformed_data = pipeline.fit_transform(df)

    return transformed_data

train_data_path = '../../../data/Balance-Scale/Balance-Scale_train.csv'
train_thread = Thread(target=preprocess_data, args=(train_data_path,))
train_thread.start()

test_data_path = '../../../data/Balance-Scale/Balance-Scale_test.csv'
test_thread = Thread(target=preprocess_data, args=(test_data_path,))
test_thread.start()

train_thread.join()
test_thread.join()

print("Data augmentation is not applicable for this dataset.")

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

model = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, model.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, model.predict_proba(X_test))

Train_AUC_OVO = roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr')

Test_AUC_OVO = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 

print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end