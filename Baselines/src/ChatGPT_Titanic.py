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
import pandas as pd
import sys


class ChatGPT_Titanic(object):
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def runClassifier(self):
        # Data Cleaning and Feature Engineering
        self.df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.df['Age'].fillna(self.df['Age'].median(), inplace=True)
        self.df['Embarked'].fillna(self.df['Embarked'].mode()[0], inplace=True)
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch']

        # One-hot encoding for categorical variables
        self.df = pd.get_dummies(self.df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

        # Split the data into training and testing sets
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Data Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Building
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        return  accuracy, 1

if __name__ == '__main__':
    inputs = InputArgs(sys.argv)

    reader = Reader(filename=inputs.dataset)
    df = reader.reader_CSV()

    gpt = ChatGPT_Titanic(df, target=inputs.target_attribute)
    accuracy, max_models = gpt.runClassifier()

    with open(inputs.log_file_name, "a") as logfile:
        dataset_name = inputs.dataset.split("/")
        name = dataset_name[len(dataset_name)-1]
        name = name.split(".")[0]
        logfile.write(f'ChatGPT_Titanic,{name},{accuracy},{inputs.time_left},{inputs.per_run_time_limit},{max_models}\n')
