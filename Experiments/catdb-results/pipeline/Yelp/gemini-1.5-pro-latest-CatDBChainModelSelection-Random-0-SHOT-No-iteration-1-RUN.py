# ```python
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

categorical_features = ['Saturday_18', 'Wednesday_6', 'Thursday_0',  # ... (All categorical columns)
                       'city', 'active', 'Business.stars']
numerical_features = ['Users.votes_funny', 'Users.votes_cool', 'review_count_y',
                   'Users.votes_useful', 'review_count_x', 'latitude',
                   'longitude', 'average_stars']

train_data['combined_review_count'] = train_data['review_count_y'] * train_data['review_count_x']
test_data['combined_review_count'] = test_data['review_count_y'] * test_data['review_count_x']
numerical_features.append('combined_review_count')  # Add new feature to the list


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_jobs=-1, objective='multi:softmax', num_class=5, random_state=42))  # Multi-threaded XGBoost
])

X_train = train_data.drop('stars', axis=1)
y_train = train_data['stars'] - 1 # Convert labels to 0-4
X_test = test_data.drop('stars', axis=1)
y_test = test_data['stars'] - 1 # Convert labels to 0-4

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
Test_Log_loss = log_loss(y_test, y_test_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_test_proba, multi_class='ovr')

print(f"Train_AUC_OVO: {Train_AUC_OVO}")
print(f"Train_AUC_OVR: {Train_AUC_OVR}")
print(f"Train_Accuracy: {Train_Accuracy}")
print(f"Train_Log_loss: {Train_Log_loss}")
print(f"Test_AUC_OVO: {Test_AUC_OVO}")
print(f"Test_AUC_OVR: {Test_AUC_OVR}")
print(f"Test_Accuracy: {Test_Accuracy}")
print(f"Test_Log_loss: {Test_Log_loss}")
# ```end