# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/adult/adult_train.csv')
test_data = pd.read_csv('data/adult/adult_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'age_group' based on age
# Usefulness: Age can be an important factor in determining the class. Grouping age into different categories can provide additional information.
train_data['age_group'] = pd.cut(train_data['age'], bins=[0, 30, 60, float('inf')], labels=['young', 'adult', 'senior'])
test_data['age_group'] = pd.cut(test_data['age'], bins=[0, 30, 60, float('inf')], labels=['young', 'adult', 'senior'])

# Add new column 'education_level' based on education and education-num
# Usefulness: Combining education and education-num can provide a more comprehensive representation of a person's education level.
train_data['education_level'] = train_data['education'] + ' (' + train_data['education-num'].astype(str) + ')'
test_data['education_level'] = test_data['education'] + ' (' + test_data['education-num'].astype(str) + ')'

# Add new column 'working_hours' based on workclass and hours-per-week
# Usefulness: Combining workclass and hours-per-week can provide insights into different working hour patterns and their relation to the class.
train_data['working_hours'] = train_data['workclass'] + ' (' + train_data['hours-per-week'].astype(str) + ')'
test_data['working_hours'] = test_data['workclass'] + ' (' + test_data['hours-per-week'].astype(str) + ')'
# end-added-column

# python-dropping-columns
# Drop columns that are not useful for classification
# Explanation: 'education-num' is dropped as it is already represented by 'education_level'
train_data.drop(columns=['education-num'], inplace=True)
test_data.drop(columns=['education-num'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for binary classification
# Explanation: Random Forest is a powerful ensemble learning algorithm that can handle both categorical and numerical features effectively.
# It also provides feature importance, which can be useful for feature selection.
target_column = 'class'
feature_columns = ['native-country', 'relationship', 'race', 'workclass', 'sex', 'capital-gain', 'age', 'fnlwgt',
                   'hours-per-week', 'capital-loss', 'marital-status', 'occupation', 'education', 'age_group',
                   'education_level', 'working_hours']

le = LabelEncoder()
# Convert categorical columns to numerical labels
for col in feature_columns:
    if train_data[col].dtype == 'object' or train_data[col].dtype == 'category':
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.fit_transform(test_data[col])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data[feature_columns], train_data[target_column], test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the model on the test dataset
y_pred = clf.predict(test_data[feature_columns])
accuracy = accuracy_score(test_data[target_column], y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 