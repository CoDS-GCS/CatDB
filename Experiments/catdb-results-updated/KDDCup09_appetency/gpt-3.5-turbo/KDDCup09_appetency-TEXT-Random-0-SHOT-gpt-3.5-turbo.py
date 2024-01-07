# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/KDDCup09_appetency/KDDCup09_appetency_train.csv')
test_data = pd.read_csv('data/KDDCup09_appetency/KDDCup09_appetency_test.csv')
# end-load-dataset

# python-added-column
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify 'APPETENCY' according to dataset description and attributes.)
train_data['Var1_plus_Var2'] = train_data['Var1'] + train_data['Var2']
train_data['Var3_times_Var4'] = train_data['Var3'] * train_data['Var4']
train_data['Var5_divided_by_Var6'] = train_data['Var5'] / train_data['Var6']

test_data['Var1_plus_Var2'] = test_data['Var1'] + test_data['Var2']
test_data['Var3_times_Var4'] = test_data['Var3'] * test_data['Var4']
test_data['Var5_divided_by_Var6'] = test_data['Var5'] / test_data['Var6']
# end-added-column

# python-dropping-columns
# Explanation why the column XX is dropped
train_data.drop(columns=['Var196', 'Var200', 'Var198', 'Var214', 'Var195', 'Var219', 'Var205', 'Var202', 'Var215', 'Var191', 'Var213', 'Var203', 'Var208', 'Var222', 'Var217', 'Var204', 'Var225', 'Var210', 'Var199', 'Var192', 'Var227', 'Var221', 'Var206', 'Var212', 'Var218', 'Var211', 'Var220', 'Var194', 'Var223', 'Var216', 'Var201', 'Var224', 'Var197', 'Var228', 'Var226', 'Var207', 'Var229', 'Var193'], inplace=True)
test_data.drop(columns=['Var196', 'Var200', 'Var198', 'Var214', 'Var195', 'Var219', 'Var205', 'Var202', 'Var215', 'Var191', 'Var213', 'Var203', 'Var208', 'Var222', 'Var217', 'Var204', 'Var225', 'Var210', 'Var199', 'Var192', 'Var227', 'Var221', 'Var206', 'Var212', 'Var218', 'Var211', 'Var220', 'Var194', 'Var223', 'Var216', 'Var201', 'Var224', 'Var197', 'Var228', 'Var226', 'Var207', 'Var229', 'Var193'], inplace=True)
# end-dropping-columns

train_data = train_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
test_data = test_data.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected
X_train = train_data.drop(columns=['APPETENCY'])
y_train = train_data['APPETENCY']

X_test = test_data.drop(columns=['APPETENCY'])
y_test = test_data['APPETENCY']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation