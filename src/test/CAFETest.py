from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import os
import openai
import torch
from caafe import data
from sklearn.metrics import accuracy_score
from tabpfn.scripts import tabular_metrics
from functools import partial
from caafe.preprocessing import make_datasets_numeric

class CAFETest(object):
    def __init__(self):
        self. openai_key = "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"
        self.metric_used = tabular_metrics.auc_metric
        self.cc_test_datasets_multiclass = data.load_all_data()

        self.df_train = None
        self.df_test = None
        self.target_column_name =None
        self.dataset_description =None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.clf_no_feat_eng = None

    def selectDataset(self, id=5):
        ds = self.cc_test_datasets_multiclass[id]
        ds, self.df_train, self.df_test, _, _ = data.get_data_split(ds, seed=0)
        self.target_column_name = ds[4][-1]
        self.dataset_description = ds[-1]

        #print(self.df_train)

        # print('---------------------------------------')
        # print(ds[0])
        # print(target_column_name)
        # print('======================================')
        # print(dataset_description)

    def prepareDataset(self):
         self.df_train, self.df_test = make_datasets_numeric(self.df_train, self.df_test, self.target_column_name)
         self.train_x, self.train_y = data.get_X_y(self.df_train, self.target_column_name)
         self.test_x, self.test_y = data.get_X_y(self.df_test, self.target_column_name)

    def setupClassifier(self):
        self.clf_no_feat_eng = RandomForestClassifier()
        self.clf_no_feat_eng = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
        self.clf_no_feat_eng.fit = partial(self.clf_no_feat_eng.fit, overwrite_warning=True)

        self.clf_no_feat_eng.fit(self.train_x, self.train_y)
        pred = self.clf_no_feat_eng.predict(self.test_x)
        acc = accuracy_score(pred, self.test_y)

        return acc

    def setupOpenAI(self):
        openai.api_key = self.openai_key
        caafe_clf = CAAFEClassifier(base_classifier=self.clf_no_feat_eng,
                                    llm_model="gpt-4",
                                    iterations=2)

        print(self.dataset_description)
        caafe_clf.fit_pandas(self.df_train,
                             target_column_name=self.target_column_name,
                             dataset_description=self.dataset_description)

        pred = caafe_clf.predict(self.df_test)
        acc = accuracy_score(pred, self.test_y)
        return acc



if __name__ == '__main__':
    cafe = CAFETest()
    cafe.selectDataset(5)
    cafe.prepareDataset()
    before_acc = cafe.setupClassifier()
    after_acc = cafe.setupOpenAI()

    print(f'Accuracy Before = {before_acc}  -- Accuracy After={after_acc}')