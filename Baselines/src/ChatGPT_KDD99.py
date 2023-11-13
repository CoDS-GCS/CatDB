from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from util.Reader import Reader
from util.InputArgs import InputArgs
import sys


class ChatGPT_KDD99(object):
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def runClassifier(self):
        # Assuming 'label' is the target variable
        X = df.drop(self.target, axis=1)
        y = df[self.target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create model pipeline (you can replace XGBClassifier with any other classifier)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        return  accuracy, 1

if __name__ == '__main__':
    inputs = InputArgs(sys.argv)

    reader = Reader(filename=inputs.dataset)
    df = reader.reader_CSV()

    gpt = ChatGPT_KDD99(df=df, target=inputs.target_attribute)
    accuracy, max_models = gpt.runClassifier()

    with open(inputs.log_file_name, "a") as logfile:
        dataset_name = inputs.dataset.split("/")
        name = dataset_name[len(dataset_name)-1]
        name = name.split(".")[0]
        logfile.write(f'ChatGPT_KDD99,{name},{accuracy},{inputs.time_left},{inputs.per_run_time_limit},{max_models}\n')
