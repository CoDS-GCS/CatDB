import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_data = "train.csv"
test_data = "test.csv"

train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

categorical_cols = [
    "Seller of the Product",
    "The item is made up of high quality",
    "Price of the item (comparatively lower than the price that is offered in offline markets)",
    "Home and Furnishing",
    "Baby, Beauty and Personal Care",
    "Return and Exchange Policy",
    "Web series/ Movies",
    "Description of the product along with display picture from various angles",
    "Word of Mouth (recommended by a friend)",
    "Social Media like Facebook, Instagram, TikTok, YouTube, etc.",
    "Delivery Charges",
    "Reviews and Ratings",
    "Number of days of Delivery",
    "Conventional Advertisements like TV, Newspaper Ads, etc",
    "Grocery",
    "Electronics",
    "Books and Stationery",
    "Apparels (Clothes)",
    "Age ",
    "Gender",
    " Will you be more attracted to the platform with content in the native language?",
    "Annual Income",
    "Will you buy a subscription service offered by an online shopping platform that includes one-day delivery, exclusive deals, and offers, video streaming?",
    "Highest Education received ",
    "Which device do you use for Online Shopping? ",
    " Do you prefer pre-online payment if there is an extra discount on it?",
    "Which of the following value-addition services do you like the most on online shopping platform?",
    "Which online shopping website do you use the most?",
    "State",
    "City (from which you do online shopping)",
    "What is your “Average Cart Value”?",
    "How many online shopping orders do you place annually?",
    "Why do you prefer Offline shopping over Online Shopping? (Tick the first option if you prefer Online Shopping)? Select your Top 2 choices* ",
    "Thinking of the last time you put items in your shopping cart but did not finish the online purchase, which of the following describes why you didn’t complete the transaction?",
    "Which payment option do you prefer the most while making an online purchase?",
    " Which factors enhance your convenience for online shopping?",
    "Please specify the reason why you didn’t stick to one particular platform. Choose Option 1 if you have always shopped from one online platform",
    "Profession",
    "Email id",
    "Name",
]

string_cols = []

target_col = "What is the maximum cart value you ever shopped?"

le = LabelEncoder()
train_df[target_col] = le.fit_transform(train_df[target_col])
test_df[target_col] = le.transform(test_df[target_col])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", OneHotEncoder(handle_unknown="ignore"), [col for col in categorical_cols if col in train_df.columns]),
    ]
)

feature_engineer = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
)

ml_pipeline = Pipeline(
    steps=[
        ("feature_engineer", feature_engineer),
        ("classifier", RandomForestClassifier(n_estimators=100)),
    ]
)

X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

ml_pipeline.fit(X_train, y_train)

y_pred_train = ml_pipeline.predict(X_train)
y_pred_test = ml_pipeline.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
train_log_loss = log_loss(y_train, ml_pipeline.predict_proba(X_train))
train_auc_ovr = roc_auc_score(y_train, ml_pipeline.predict_proba(X_train), multi_class="ovr")
train_auc_ovo = roc_auc_score(y_train, ml_pipeline.predict_proba(X_train), multi_class="ovo")

test_accuracy = accuracy_score(y_test, y_pred_test)
test_log_loss = log_loss(y_test, ml_pipeline.predict_proba(X_test))
test_auc_ovr = roc_auc_score(y_test, ml_pipeline.predict_proba(X_test), multi_class="ovr")
test_auc_ovo = roc_auc_score(y_test, ml_pipeline.predict_proba(X_test), multi_class="ovo")

print(f"Train_Accuracy: {train_accuracy}")
print(f"Train_Log_loss: {train_log_loss}")
print(f"Train_AUC_OVR: {train_auc_ovr}")
print(f"Train_AUC_OVO: {train_auc_ovo}")
print(f"Test_Accuracy: {test_accuracy}")
print(f"Test_Log_loss: {test_log_loss}")
print(f"Test_AUC_OVR: {test_auc_ovr}")
print(f"Test_AUC_OVO: {test_auc_ovo}")
# 