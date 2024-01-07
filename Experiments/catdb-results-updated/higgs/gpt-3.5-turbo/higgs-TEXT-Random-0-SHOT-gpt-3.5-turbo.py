# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/higgs/higgs_train.csv')
test_data = pd.read_csv('data/higgs/higgs_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'jet2pt_squared' as the square of 'jet2pt'
train_data['jet2pt_squared'] = train_data['jet2pt'] ** 2
test_data['jet2pt_squared'] = test_data['jet2pt'] ** 2

# Add new column 'jet4pt_squared' as the square of 'jet4pt'
train_data['jet4pt_squared'] = train_data['jet4pt'] ** 2
test_data['jet4pt_squared'] = test_data['jet4pt'] ** 2

# Add new column 'm_wbb_minus_m_wwbb' as the difference between 'm_wbb' and 'm_wwbb'
train_data['m_wbb_minus_m_wwbb'] = train_data['m_wbb'] - train_data['m_wwbb']
test_data['m_wbb_minus_m_wwbb'] = test_data['m_wbb'] - test_data['m_wwbb']

# Add new column 'jet1pt_divided_by_jet3pt' as the division of 'jet1pt' by 'jet3pt'
train_data['jet1pt_divided_by_jet3pt'] = train_data['jet1pt'] / train_data['jet3pt']
test_data['jet1pt_divided_by_jet3pt'] = test_data['jet1pt'] / test_data['jet3pt']
# end-added-column

# python-dropping-columns
# Drop 'jet4b-tag' column as it may be redundant and hurt the predictive performance
train_data.drop(columns=['jet4b-tag'], inplace=True)
test_data.drop(columns=['jet4b-tag'], inplace=True)
# end-dropping-columns

# python-training-technique
# Create feature and target dataframes
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a column transformer for scaling the numerical columns
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), X_train.columns)])

# Create the pipeline with logistic regression as the classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the pipeline on the validation set
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation