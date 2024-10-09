# ```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data['Annual Income'].fillna(train_data['Annual Income'].mode()[0], inplace=True)
test_data['Annual Income'].fillna(test_data['Annual Income'].mode()[0], inplace=True)
train_data['Which online shopping website do you use the most?'].fillna(train_data['Which online shopping website do you use the most?'].mode()[0], inplace=True)
test_data['Which online shopping website do you use the most?'].fillna(test_data['Which online shopping website do you use the most?'].mode()[0], inplace=True)

train_data['Name'].fillna("Unknown", inplace=True)
test_data['Name'].fillna("Unknown", inplace=True)
train_data['Email id'].fillna("Unknown", inplace=True)
test_data['Email id'].fillna("Unknown", inplace=True)

train_data = train_data.drop(columns=['Timestamp'])
test_data = test_data.drop(columns=['Timestamp'])

categorical_cols = ["Web series/ Movies", "Price of the item (comparatively lower than the price that is offered in offline markets)", "Seller of the Product", "Description of the product along with display picture from various angles", "Electronics", "Delivery Charges", "Social Media like Facebook, Instagram, TikTok, YouTube, etc.", "Number of days of Delivery", "Reviews and Ratings", "Return and Exchange Policy", "Grocery", "The item is made up of high quality", "Apparels (Clothes)", "Home and Furnishing", "Books and Stationery", "Conventional Advertisements like TV, Newspaper Ads, etc", "Baby, Beauty and Personal Care", "Word of Mouth (recommended by a friend)", "Why do you prefer Offline shopping over Online Shopping? (Tick the first option if you prefer Online Shopping)? Select your Top 2 choices* ", "Highest Education received ", "Annual Income", "Thinking of the last time you put items in your shopping cart but did not finish the online purchase, which of the following describes why you didn’t complete the transaction?", "Please specify the reason why you didn’t stick to one particular platform. Choose Option 1 if you have always shopped from one online platform", "Profession", "How many online shopping orders do you place annually?", "Gender", "Which device do you use for Online Shopping?", " Will you be more attracted to the platform with content in the native language?", "Which of the following value-addition services do you like the most on online shopping platform?", "Age ", "Which payment option do you prefer the most while making an online purchase?", " Which factors enhance your convenience for online shopping?", " Do you prefer pre-online payment if there is an extra discount on it?", "What is your “Average Cart Value”?", "Will you buy a subscription service offered by an online shopping platform that includes one-day delivery, exclusive deals, and offers, video streaming?", "City (from which you do online shopping)", "State", "Which online shopping website do you use the most?",'Name', 'Email id']
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on the combined data
    le.fit(train_data[col].astype(str).values.reshape(-1, 1).ravel().tolist() + test_data[col].astype(str).values.reshape(-1, 1).ravel().tolist())
    # Transform the individual datasets
    train_data[col] = le.transform(train_data[col].astype(str).values.reshape(-1, 1).ravel())
    test_data[col] = le.transform(test_data[col].astype(str).values.reshape(-1, 1).ravel())

X_train = train_data.drop(columns=['What is the maximum cart value you ever shopped?'])
y_train = train_data['What is the maximum cart value you ever shopped?']
X_test = test_data.drop(columns=['What is the maximum cart value you ever shopped?'])
y_test = test_data['What is the maximum cart value you ever shopped?']

model = RandomForestClassifier(random_state=42)

trn = model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, model.predict(X_train))
Test_Accuracy = accuracy_score(y_test, y_pred)
Train_Log_loss = log_loss(y_train, model.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, y_pred_prob)
Train_AUC_OVO = roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end