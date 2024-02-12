# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")
test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if len(train_data[column].unique()) == 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# c_53 and c_11 are similar, so we can drop one of them
# df.drop(columns=['c_11'], inplace=True)

# c_76 and c_52 are similar, so we can drop one of them
# df.drop(columns=['c_52'], inplace=True)

# c_4 and c_10 are similar, so we can drop one of them
# df.drop(columns=['c_10'], inplace=True)

# c_31 and c_46 are similar, so we can drop one of them
# df.drop(columns=['c_46'], inplace=True)

# c_38 and c_59 are similar, so we can drop one of them
# df.drop(columns=['c_59'], inplace=True)

# c_74 and c_19 are similar, so we can drop one of them
# df.drop(columns=['c_19'], inplace=True)

# c_34 and c_42 are similar, so we can drop one of them
# df.drop(columns=['c_42'], inplace=True)

# c_65 and c_54 are similar, so we can drop one of them
# df.drop(columns=['c_54'], inplace=True)

# c_29 and c_78 are similar, so we can drop one of them
# df.drop(columns=['c_78'], inplace=True)

# c_63 and c_62 are similar, so we can drop one of them
# df.drop(columns=['c_62'], inplace=True)

# c_73 and c_12 are similar, so we can drop one of them
# df.drop(columns=['c_12'], inplace=True)

# c_18 and c_17 are similar, so we can drop one of them
# df.drop(columns=['c_17'], inplace=True)

# c_28 and c_66 are similar, so we can drop one of them
# df.drop(columns=['c_66'], inplace=True)

# c_69 and c_28 are similar, so we can drop one of them
# df.drop(columns=['c_28'], inplace=True)

# c_50 and c_55 are similar, so we can drop one of them
# df.drop(columns=['c_55'], inplace=True)

# c_33 and c_15 are similar, so we can drop one of them
# df.drop(columns=['c_15'], inplace=True)

# c_72 and c_71 are similar, so we can drop one of them
# df.drop(columns=['c_71'], inplace=True)

# c_75 and c_49 are similar, so we can drop one of them
# df.drop(columns=['c_49'], inplace=True)

# c_9 and c_43 are similar, so we can drop one of them
# df.drop(columns=['c_43'], inplace=True)

# c_58 and c_40 are similar, so we can drop one of them
# df.drop(columns=['c_40'], inplace=True)

# c_57 and c_24 are similar, so we can drop one of them
# df.drop(columns=['c_24'], inplace=True)

# c_35 and c_56 are similar, so we can drop one of them
# df.drop(columns=['c_56'], inplace=True)

# c_22 and c_2 are similar, so we can drop one of them
# df.drop(columns=['c_2'], inplace=True)

# c_48 and c_20 are similar, so we can drop one of them
# df.drop(columns=['c_20'], inplace=True)

# c_61 and c_16 are similar, so we can drop one of them
# df.drop(columns=['c_16'], inplace=True)

# c_77 and c_75 are similar, so we can drop one of them
# df.drop(columns=['c_75'], inplace=True)

# c_36 and c_31 are similar, so we can drop one of them
# df.drop(columns=['c_31'], inplace=True)

# c_39 and c_72 are similar, so we can drop one of them
# df.drop(columns=['c_72'], inplace=True)

# c_25 and c_79 are similar, so we can drop one of them
# df.drop(columns=['c_79'], inplace=True)

# c_64 and c_37 are similar, so we can drop one of them
# df.drop(columns=['c_37'], inplace=True)

# c_68 and c_13 are similar, so we can drop one of them
# df.drop(columns=['c_13'], inplace=True)

# c_6 and c_3 are similar, so we can drop one of them
# df.drop(columns=['c_3'], inplace=True)

# c_32 and c_27 are similar, so we can drop one of them
# df.drop(columns=['c_27'], inplace=True)

# c_67 and c_21 are similar, so we can drop one of them
# df.drop(columns=['c_21'], inplace=True)

# c_14 and c_2 are similar, so we can drop one of them
# df.drop(columns=['c_2'], inplace=True)

# c_23 and c_42 are similar, so we can drop one of them
# df.drop(columns=['c_42'], inplace=True)

# c_44 and c_7 are similar, so we can drop one of them
# df.drop(columns=['c_7'], inplace=True)

# c_26 and c_45 are similar, so we can drop one of them
# df.drop(columns=['c_45'], inplace=True)

# c_30 and c_45 are similar, so we can drop one of them
# df.drop(columns=['c_45'], inplace=True)

# c_60 and c_8 are similar, so we can drop one of them
# df.drop(columns=['c_8'], inplace=True)

# c_5 and c_2 are similar, so we can drop one of them
# df.drop(columns=['c_2'], inplace=True)

# c_11 and c_53 are similar, so we can drop one of them
# df.drop(columns=['c_53'], inplace=True)

# c_41 and c_51 are similar, so we can drop one of them
# df.drop(columns=['c_51'], inplace=True)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Separate the target variable
X_train = train_data.drop(columns=['c_1'])
y_train = train_data['c_1']
X_test = test_data.drop(columns=['c_1'])
y_test = test_data['c_1']

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

# Predict the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy and f1 score
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

# Print the results
print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end