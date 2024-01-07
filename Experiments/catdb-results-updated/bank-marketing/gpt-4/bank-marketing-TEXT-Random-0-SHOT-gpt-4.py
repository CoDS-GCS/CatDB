# python-import
# Import all required packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/bank-marketing/bank-marketing_train.csv")
test = pd.read_csv("data/bank-marketing/bank-marketing_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: TF-IDF features for each text column
# Usefulness: The TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect how important a word is to a document in a collection or corpus. It can add useful real world knowledge to classify 'Class' as it can capture the importance of certain words or phrases in the context of each attribute.

# Create a function to generate TF-IDF features for each text column
def generate_tfidf_features(df, columns):
    for column in columns:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df[column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        df = pd.concat([df, tfidf_df], axis=1)
    return df

# List of text columns
text_columns = ['V8', 'V16', 'V7', 'V3', 'V9', 'V11', 'V2', 'V5', 'V4']

# Generate TF-IDF features for each text column in train and test datasets
train = generate_tfidf_features(train, text_columns)
test = generate_tfidf_features(test, text_columns)
# end-added-column

# python-dropping-columns
# Explanation why the column XX is dropped
# The original text columns are dropped after the TF-IDF features are generated because they are no longer needed. The new TF-IDF features capture the important information from the text columns for the downstream classifier.
train.drop(columns=text_columns, inplace=True)
test.drop(columns=text_columns, inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: RandomForestClassifier is selected because it is a versatile and widely used machine learning algorithm that can handle a large number of features and is less prone to overfitting due to its ensemble nature.

# Define features and target variable for training
X_train = train.drop('Class', axis=1)
y_train = train['Class']

# Define a pipeline with a StandardScaler for preprocessing and a RandomForestClassifier for training
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
X_test = test.drop('Class', axis=1)
y_test = test['Class']

# Make predictions on the test dataset
y_pred = pipeline.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation