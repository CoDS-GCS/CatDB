# python-import
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/car/car_train.csv')
test_data = pd.read_csv('data/car/car_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'persons_doors' as a combination of 'persons' and 'doors'
train_data['persons_doors'] = train_data['persons'] + '_' + train_data['doors']
test_data['persons_doors'] = test_data['persons'] + '_' + test_data['doors']

# Add new column 'safety_maint' as a combination of 'safety' and 'maint'
train_data['safety_maint'] = train_data['safety'] + '_' + train_data['maint']
test_data['safety_maint'] = test_data['safety'] + '_' + test_data['maint']

# Add new column 'buying_lug_boot' as a combination of 'buying' and 'lug_boot'
train_data['buying_lug_boot'] = train_data['buying'] + '_' + train_data['lug_boot']
test_data['buying_lug_boot'] = test_data['buying'] + '_' + test_data['lug_boot']
# end-added-column

# python-dropping-columns
# Drop columns 'persons', 'doors', 'safety', 'maint', 'buying', 'lug_boot' as they are no longer needed
train_data.drop(columns=['persons', 'doors', 'safety', 'maint', 'buying', 'lug_boot'], inplace=True)
test_data.drop(columns=['persons', 'doors', 'safety', 'maint', 'buying', 'lug_boot'], inplace=True)
# end-dropping-columns

# python-training-technique
# Split the train_data into features (X) and target variable (y)
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

# Define the preprocessing steps for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['persons_doors', 'safety_maint', 'buying_lug_boot'])
    ])

# Create a pipeline with a RandomForestClassifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the pipeline on the test data
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
y_pred = pipeline.predict(X_test)

# Report the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 