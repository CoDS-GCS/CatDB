import random

import sklearn.datasets
from sklearn.metrics import accuracy_score
import autosklearn.classification
from sklearn.model_selection import train_test_split
class SKLearnAutoML(object):
    def __init__(self, time_left, per_run_time_limit):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.time_left = time_left
        self.per_run_time_limit = per_run_time_limit

    def split_data(self, data, target, test_size, random_size):
        X = data.drop(target, axis=1)
        y = data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_size)

    def runClassifier(self):
        rand_val = random.randint(100000, 900000)
        tmp_file_name = f'/tmp/autosklearn_classification_{rand_val}'
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.time_left,
            per_run_time_limit=self.per_run_time_limit,
            tmp_folder=tmp_file_name,
        )
        automl.fit(self.X_train, self.y_train, dataset_name=f"dataset-{rand_val}")
        predictions = automl.predict(self.X_test)
        # print(automl.leaderboard())
        # print(predictions)

        return accuracy_score(self.y_test, predictions), len(automl.leaderboard())
