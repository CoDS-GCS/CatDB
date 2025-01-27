import learn2clean.loading.reader as rd
import learn2clean.qlearning.qlearner as ql
import shutil
import random
import time
import pandas as pd
import learn2clean.normalization.normalizer as nl
import learn2clean.feature_selection.feature_selector as fs
import learn2clean.duplicate_detection.duplicate_detector as dd
import learn2clean.outlier_detection.outlier_detector as od
import learn2clean.imputation.imputer as imp

from LogResults import LogResults


class PrepareData(object):
    def __init__(self, dataset_name, target_attribute, task_type, train_path, test_path, output_dir,
                 result_output_path):
        self.dataset_name = dataset_name
        self.target_attribute = target_attribute
        self.task_type = task_type
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.result_output_path = result_output_path

    def run(self):
        print(f"\n**** Learn2Clean Running {self.dataset_name}****\n")
        df_train = pd.read_csv(self.train_path, na_values=[' ', '?', '-'], low_memory=False, encoding="ISO-8859-1")
        df_test = pd.read_csv(self.test_path, na_values=[' ', '?', '-'], low_memory=False, encoding="ISO-8859-1")
        log_path = "./save"
        try:
            shutil.rmtree(log_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        if self.task_type in {"binary", "multiclass"}:
            goals = ['NB', 'LDA']  # , 'CART'
        else:
            goals = ['MARS', 'LASSO', 'OLS']

        # no encoding of the target variable
        d_not_enc = rd.Reader(sep=',', verbose=False, encoding=False)
        data = [self.train_path]
        dataset = d_not_enc.train_test_split(data, self.target_attribute)

        # cc.constraint_discovery(df_train, file_name=self.dataset_name)
        # cc.pattern_discovery(df_train, file_name=self.dataset_name)
        # cc.Consistency_checker(dataset.copy(), strategy='CC', file_name=self.dataset_name,verbose=False).transform()

        plan = None
        max_result = 0
        total_time = 0
        time_flag = False

        if self.dataset_name == "EU-IT":
            plan="ZS->IQR->ED->NB"
        elif self.dataset_name == "Etailing":
            plan="LOF -> NB"
        elif self.dataset_name == "Yelp":
            plan="ZS->IQR->ED->NB"
        else:
            for i in range(1, 2):
                for goal in goals:
                    try:
                        l2c = ql.Qlearner(dataset=dataset.copy(), goal=goal, target_goal=self.target_attribute,
                                          target_prepare=None, file_name=self.dataset_name, verbose=False)
                        (_, _, _, _, _, p, metrics_name, result, t) = l2c.learn2clean()
                        total_time += t
                        if result > max_result:
                            max_result = result
                            plan = p
                    except Exception as err:
                        pass

            if plan is None:
                FEC = ['MR', 'VAR', 'LC', 'Tree', 'WR', 'SVC', 'L1', 'IMP']
                NC = ['ZS', 'MM', 'DS', 'Log10']
                IM = ['EM', 'MICE', 'KNN', 'RAND', 'MF', 'MEAN', 'MEDIAN', 'DROP']
                OD = ['ZSB', 'IQR', 'LOF', 'ZS', 'IQR', 'LOF']
                DD = ['ED', 'AD', 'METRIC']
                plan = [random.choice(FEC), random.choice(NC), random.choice(IM), random.choice(OD), random.choice(DD), random.choice(goals)]
                plan = "->".join(plan)
                time_flag = True

        if plan is None:
            lr = LogResults(self.dataset_name, self.task_type, "False", 3, plan, -1, -1, len(df_train), -1,
                            "", total_time)
            lr.save_results(self.result_output_path)
        else:
            plans = plan.split("->")

            data = [self.train_path, self.test_path]
            clean_data = d_not_enc.train_test_split(data, self.target_attribute)

            y_train = clean_data["target"]
            y_test = clean_data["target_test"]
            rp = 0
            time_start = time.time()
            for p in plans:
                p = p.strip()
                if p in {'MR', 'VAR', 'LC', 'Tree', 'WR', 'SVC', 'L1', 'IMP'}:
                    clean_data = fs.Feature_selector(dataset=clean_data.copy(), strategy=p, verbose=False).transform()

                elif p in {'ZS', 'MM', 'DS', 'Log10'}:
                    clean_data = nl.Normalizer(clean_data.copy(), strategy=p, verbose=False).transform()

                elif p in {'EM', 'MICE', 'KNN', 'RAND', 'MF', 'MEAN', 'MEDIAN', 'DROP'}:
                    clean_data = imp.Imputer(clean_data.copy(), strategy=p, verbose=False).transform()

                elif p in {'ZSB', 'IQR', 'LOF', 'ZS', 'IQR', 'LOF'}:
                    clean_data = od.Outlier_detector(clean_data.copy(), strategy=p, verbose=False).transform()

                elif p in {'ED', 'AD', 'METRIC'}:
                    clean_data = dd.Duplicate_detector(clean_data.copy(), strategy=p, verbose=False).transform()
                else:
                    rp += 1
            time_end = time.time()
            if rp == 1:
                cols = df_train.columns
                clean_data_train = clean_data["train"]
                clean_data_test = clean_data["test"]

                clean_data_train[self.target_attribute] = y_train
                clean_data_test[self.target_attribute] = y_test

                for nc in {'New_ID', 'row'}:
                    if nc in clean_data_train.columns:
                        clean_data_train = clean_data_train.drop("New_ID", axis=1)
                        clean_data_test = clean_data_test.drop("New_ID", axis=1)

                d = []
                for c in clean_data_train.columns:
                    if c not in cols:
                        d.append(c)
                for c in cols:
                    if c not in clean_data_train.columns and c not in d:
                        d.append(c)

                cf = len(clean_data_train.columns)
                of = len(df_train.columns)
                clean_data_train.to_csv(f"{self.output_dir}/{self.dataset_name}_Learn2Clean_train.csv", index=False, header=True)
                clean_data_test.to_csv(f"{self.output_dir}/{self.dataset_name}_Learn2Clean_test.csv", index=False, header=True)
                if len(d) > 0:
                    rf = "#".join(d).replace(",", ";").replace('"', '')
                else:
                    rf = None
                if time_flag:
                    total_time += time_end - time_start
                lr = LogResults(self.dataset_name, self.task_type, "True", 3, plan, of, cf, len(df_train),
                                len(clean_data_train), rf, total_time)
                lr.save_results(self.result_output_path)
