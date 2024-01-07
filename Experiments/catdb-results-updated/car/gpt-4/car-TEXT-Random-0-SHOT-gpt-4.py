# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/car/car_train.csv")
test = pd.read_csv("data/car/car_test.csv")
# end-load-dataset

# python-added-column
# Feature name: safety_level
# Usefulness: This feature will map the safety level to a numerical scale. This will help the classifier to understand the importance of safety level in determining the class.
safety_map = {'low': 0, 'med': 1, 'high': 2}
train['safety_level'] = train['safety'].map(safety_map)
test['safety_level'] = test['safety'].map(safety_map)
# end-added-column

# python-added-column
# Feature name: buying_level
# Usefulness: This feature will map the buying level to a numerical scale. This will help the classifier to understand the importance of buying level in determining the class.
buying_map = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
train['buying_level'] = train['buying'].map(buying_map)
test['buying_level'] = test['buying'].map(buying_map)
# end-added-column

# python-dropping-columns
# Explanation: The 'safety' and 'buying' columns are dropped because they have been transformed into numerical columns 'safety_level' and 'buying_level'. The original columns are no longer needed.
train.drop(columns=['safety', 'buying'], inplace=True)
test.drop(columns=['safety', 'buying'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation: RandomForestClassifier is used because it can handle both categorical and numerical features. It also works well with multiclass classification problems. 

# Define preprocessing for numerical and categorical features
numerical_features = ['persons', 'safety_level', 'buying_level']
categorical_features = ['maint', 'lug_boot','doors']

numerical_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Combine preprocessing and model in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Split the training data
X_train = train.drop('class', axis=1)
y_train = train['class']

# Fit the model
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
X_test = test.drop('class', axis=1)
y_test = test['class']

# Predict the test data
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# end-evaluation