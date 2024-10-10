from automl.AutoML import AutoML, result
from util.Config import Config
from util.Data import Dataset
from util.Namespace import Namespace as ns

import os
import time
import pandas as pd
import re
import h2o
from h2o.automl import H2OAutoML


class H2O(AutoML):
    def __init__(self, dataset: Dataset, config: Config, *args, **kwargs):
        AutoML.__init__(self, dataset=dataset, config=config)

    def extract_preds(self, h2o_preds, test, target):
        h2o_preds = h2o_preds.as_data_frame(use_pandas=False)
        preds = self.to_data_frame(arr=h2o_preds[1:], columns=h2o_preds[0])
        y_pred = preds.iloc[:, 0]

        h2o_truth = test[:, target].as_data_frame(use_pandas=False, header=False)
        y_truth = self.to_data_frame(h2o_truth)

        predictions = y_pred.values
        probabilities = preds.iloc[:, 1:].values
        prob_labels = h2o_labels = h2o_preds[0][1:]
        if all([re.fullmatch(r"p(-?\d)+", p) for p in prob_labels]):
            # for categories represented as numerical values, h2o prefixes the probabilities columns with p
            # in this case, we let the app setting the labels to avoid mismatch
            prob_labels = None
        truth = y_truth.values

        return ns(predictions=predictions,
                  truth=truth,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  h2o_labels=h2o_labels)

    def to_data_frame(self, arr, columns=None):
        return pd.DataFrame.from_records(arr, columns=columns)

    def output_subdir(self, name):
        subdir = os.path.join(self.config.output_dir, name)
        self.touch(subdir, as_dir=True)
        return subdir

    def write_csv(self, df, path):
        self.touch(path)
        df.to_csv(path, header=True, index=False)

    def save_model(self, model_id, dest_dir='.', mformat='json'):
        model = h2o.get_model(model_id)
        if mformat == 'mojo':
            return model.save_mojo(path=dest_dir)
        elif mformat == 'binary':
            return h2o.save_model(model, path=dest_dir)
        else:
            return model.save_model_details(path=dest_dir)

    def frame_name(self, fr_type):
        return '_'.join([fr_type, self.config.name])

    def write_preds(self, preds, path):
        df = self.to_data_frame(preds.probabilities, columns=preds.probabilities_labels)
        df = df.assign(predictions=preds.predictions)
        df = df.assign(truth=preds.truth)
        self.write_csv(df, path)

    def save_artifacts(self, automl):
        artifacts = ['leaderboard']
        try:
            models_artifacts = []
            lb_pat = re.compile(r"leaderboard(?:\[(.*)\])?")
            lb_match = next((lb_pat.fullmatch(a) for a in artifacts), None)
            if lb_match:
                lb_ext = list(filter(None, re.split("[,; ]", (lb_match.group(1) or ""))))
                lb = h2o.automl.get_leaderboard(automl, lb_ext).as_data_frame()
                models_dir = self.output_subdir("models")
                lb_path = os.path.join(models_dir, "leaderboard.csv")
                self.write_csv(lb, lb_path)
                models_artifacts.append(lb_path)
            else:
                lb = automl.leaderboard.as_data_frame()

            models_pat = re.compile(r"models(\[(json|binary|mojo)(?:,(\d+))?\])?")
            models = list(filter(models_pat.fullmatch, artifacts))
            for m in models:
                models_dir = self.output_subdir("models")
                all_models_se = next((mid for mid in lb['model_id'] if mid.startswith("StackedEnsemble_AllModels")),
                                     None)
                match = models_pat.fullmatch(m)
                mformat = match.group(2) or 'json'
                topN = int(match.group(3) or -1)
                if topN < 0 and mformat != 'json' and all_models_se:
                    models_artifacts.append(self.save_model(all_models_se, dest_dir=models_dir, mformat=mformat))
                else:
                    count = 0
                    for mid in lb['model_id']:
                        if topN < 0 or count < topN:
                            self.save_model(mid, dest_dir=models_dir, mformat=mformat)
                            count += 1
                        else:
                            break

                    models_archive = os.path.join(models_dir, f"models_{mformat}.zip")
                    self.zip_path(models_dir, models_archive, filter_=lambda p: p not in models_artifacts)
                    models_artifacts.append(models_archive)
                    self.clean_dir(models_dir,
                                   filter_=lambda p: p not in models_artifacts
                                                     and os.path.splitext(p)[1] in ['.json', '.zip', ''])

        except Exception:
            print("Error when saving artifacts.")

    def run(self):
        print(f"\n**** H2OAutoML AutoML [v{h2o.__version__}] ****\n")
        try:
            time_start = time.time()
            jvm_memory = str(
                round(self.config.jvm_memory * 2 / 3)) + "M"  # leaving 1/3rd of available memory for XGBoost
            max_port_range = 49151
            min_port_range = 1024
            rnd_port = os.getpid() % (max_port_range - min_port_range) + min_port_range
            h2o.init(nthreads=self.config.nthreads,
                     port=rnd_port,
                     min_mem_size=jvm_memory,
                     max_mem_size=jvm_memory)

            train = h2o.import_file(self.dataset.train_path)
            test = h2o.import_file(self.dataset.test_path)

            if self.dataset.task_type == 'binary' or self.dataset.task_type == 'multiclass':
                train[self.dataset.target_attribute] = train[self.dataset.target_attribute].asfactor()
                test[self.dataset.target_attribute] = test[self.dataset.target_attribute].asfactor()
                pref_metric = "AUC"
            else:
                pref_metric = "r2"

            ml = H2OAutoML(max_runtime_secs=self.config.max_runtime_seconds,
                           sort_metric=pref_metric,
                           seed=self.config.seed,
                           exclude_algos = ["StackedEnsemble", "DeepLearning"]
                           )

            ml.train(y=self.dataset.target_attribute, training_frame=train)

            if not ml.leader:
                raise Exception("H2OAutoML could not produce any model in the requested time.")

            time_end = time.time()
            time_execute = time_end - time_start

            # Extract Results
            if self.dataset.task_type == "binary":
                result_train = ml.leader.model_performance(test_data=train)
                result_test = ml.leader.model_performance(test_data=test)
                self.log_results.train_auc = result_train["AUC"]
                self.log_results.test_auc = result_test["AUC"]
                try:
                    self.log_results.train_log_loss = result_train["LogLoss"]
                    self.log_results.train_rmse = result_train["RMSE"]

                    self.log_results.test_log_loss = result_test["LogLoss"]
                    self.log_results.test_rmse = result_test["RMSE"]
                except:
                    pass

            elif self.dataset.task_type == "multiclass":
                result_train_ovo = ml.leader.model_performance(test_data=train, auc_type="macro_ovo")
                result_train_ovr = ml.leader.model_performance(test_data=train, auc_type="macro_ovr")

                result_test_ovo = ml.leader.model_performance(test_data=test, auc_type="macro_ovo")
                result_test_ovr = ml.leader.model_performance(test_data=test, auc_type="macro_ovr")

                self.log_results.train_auc_ovo = result_train_ovo["AUC"]
                self.log_results.train_auc_ovr = result_train_ovr["AUC"]
                self.log_results.train_log_loss = result_train_ovo["LogLoss"]
                self.log_results.train_rmse = result_train_ovo["RMSE"]

                self.log_results.test_auc_ovo = result_test_ovo["AUC"]
                self.log_results.test_auc_ovr = result_test_ovr["AUC"]
                self.log_results.test_log_loss = result_test_ovo["LogLoss"]
                self.log_results.test_rmse = result_test_ovo["RMSE"]

            elif self.dataset.task_type == "regression":
                result_train = ml.leader.model_performance(test_data=train)
                result_test = ml.leader.model_performance(test_data=test)

                self.log_results.train_r_squared = result_train.r2()
                self.log_results.train_rmse = result_train["RMSE"]

                self.log_results.test_r_squared = result_test.r2()
                self.log_results.test_rmse = result_test["RMSE"]

            self.log_results.number_iteration = self.config.iteration
            self.log_results.llm_model = self.config.llm_model
            self.log_results.status = "True"
            self.log_results.time_execution = time_execute
            self.log_results.config = "H2O"
            self.log_results.save_results(result_output_path=self.config.output_path)

        finally:
            con = h2o.connection()
            if con:
                con.close()
                if con.local_server:
                    con.local_server.shutdown()
