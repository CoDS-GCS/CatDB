# ```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

combined_data = pd.concat([train_data, test_data], axis=0)

numerical_cols_impute = ["Annual brutto salary (without bonus and stocks) one year ago. Only answer if staying in the same country"]
imputer_numerical = SimpleImputer(strategy='median')
combined_data[numerical_cols_impute] = imputer_numerical.fit_transform(combined_data[numerical_cols_impute])

categorical_cols_impute = ["Age", "Company type", "Gender", "Employment status", "Your main technology / programming language",
                          "Have you lost your job due to the coronavirus outbreak?", "Position ", "Company size",
                          "Сontract duration", "Seniority level", "Years of experience in Germany", "Main language at work",
                          "Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR",
                          "Number of vacation days", "Total years of experience",
                          "Have you been forced to have a shorter working week (Kurzarbeit)? If yes, how many hours per week", "Other technologies/programming languages you use often"]
imputer_categorical = SimpleImputer(strategy='most_frequent')
combined_data[categorical_cols_impute] = imputer_categorical.fit_transform(combined_data[categorical_cols_impute])

string_cols_impute = ["Yearly bonus + stocks in EUR", "Annual bonus+stocks one year ago. Only answer if staying in same country"]
imputer_string = SimpleImputer(strategy='most_frequent')
combined_data[string_cols_impute] = imputer_string.fit_transform(combined_data[string_cols_impute])

numerical_cols_scaling = ["Annual brutto salary (without bonus and stocks) one year ago. Only answer if staying in the same country", "Yearly brutto salary (without bonus and stocks) in EUR"]
scaler = StandardScaler()
combined_data[numerical_cols_scaling] = scaler.fit_transform(combined_data[numerical_cols_scaling])

categorical_cols_encoding = ["Age", "Company type", "Gender", "Employment status",
                            "Your main technology / programming language",
                            "Have you lost your job due to the coronavirus outbreak?", "Company size",
                            "Сontract duration", "Seniority level", "Years of experience in Germany",
                            "Main language at work",
                            "Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR",
                            "City", "Number of vacation days", "Total years of experience",
                            "Have you been forced to have a shorter working week (Kurzarbeit)? If yes, how many hours per week", "Other technologies/programming languages you use often"]
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(combined_data[categorical_cols_encoding]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols_encoding))
combined_data = combined_data.reset_index(drop=True)
combined_data = combined_data.drop(categorical_cols_encoding, axis=1)
combined_data = pd.concat([combined_data, encoded_df], axis=1)

train_data = combined_data.iloc[:len(train_data), :]
test_data = combined_data.iloc[len(train_data):, :]

train_data.drop(columns=['Timestamp'], inplace=True)
test_data.drop(columns=['Timestamp'], inplace=True)

train_data.drop(columns=['Yearly bonus + stocks in EUR'], inplace=True)
test_data.drop(columns=['Yearly bonus + stocks in EUR'], inplace=True)

train_data.drop(columns=['Annual bonus+stocks one year ago. Only answer if staying in same country'], inplace=True)
test_data.drop(columns=['Annual bonus+stocks one year ago. Only answer if staying in same country'], inplace=True)

X_train = train_data.drop('Position ', axis=1)
y_train = train_data['Position ']
X_test = test_data.drop('Position ', axis=1)
y_test = test_data['Position ']

trn = RandomForestClassifier(random_state=42)
trn.fit(X_train, y_train)



Train_Accuracy = trn.score(X_train, y_train)
Test_Accuracy = trn.score(X_test, y_test)
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Test_Accuracy:{Test_Accuracy}")
# ```