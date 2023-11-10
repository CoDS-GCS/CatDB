import h2o
from h2o.automl import H2OAutoML

class H2O_AutoML(object):
    def __init__(self, time_left, per_run_time_limit):
        self.x = None
        self.y = None

        self.df_train = None
        self.df_test = None
        self.df_valid = None

        h2o.init()
        self.time_left = time_left
        self.per_run_time_limit = per_run_time_limit

    def split_data(self, fname, target, test_size, random_size):
        data = h2o.import_file(fname)
        self.df_train, self.df_test, self.df_valid = data.split_frame(ratios=[float(1-test_size), random_size])
        self.X = self.df_train.columns
        self.X.remove(target)
        self.y = target

    def runClassifier(self):
        automl = H2OAutoML(max_runtime_secs=self.time_left,
                 max_runtime_secs_per_model=self.per_run_time_limit, exclude_algos=["StackedEnsemble", "DeepLearning"], stopping_metric='auc')

        automl.train(x=self.X, y=self.y, training_frame=self.df_train, validation_frame=self.df_valid)
        lb = automl.leaderboard
        print(lb)
        # titanic_pred = automl.leader.predict(self.df_test)
        # print(titanic_pred)
        acu = automl.leader.model_performance(self.df_test).auc()
        return acu, len(lb)
