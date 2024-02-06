import numpy as np
import pandas as pd
import os
import json
import psutil
from memory_profiler import memory_usage
from sklearn.preprocessing import LabelEncoder
import multiprocessing
import time
import sys
#import holoclean
#import datawig
#from detect import NullDetector, ViolationDetector
#from repair.featurize import *
#from operations.api import KGFarm
#kgfarm = KGFarm()
#import psycopg2
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


sys.path.append('../')
DIR_PATH = '.'
RANDOM_STATE = 30
RANDOM_STATE_SAMPLING = 30
np.random.seed(RANDOM_STATE)



def encode(X):
    categorical_cols = X.select_dtypes(include=['object']).columns
    print('categorical_cols',categorical_cols)
    le = LabelEncoder()
    for col in categorical_cols:
        # Replace NaN values with a placeholder (e.g., -1) before encoding
        X[col] = X[col].fillna('xxx')
        
        # Encode only non-placeholder values (not equal to -1)
        non_placeholder_mask = (X[col] != 'xxx')
        X.loc[non_placeholder_mask, col] = le.fit_transform(X.loc[non_placeholder_mask, col])
        
        # Optionally, revert NaN values back to NaN
        X[col] = X[col].replace('xxx', float('nan'))
    print('X',X)
    return X


def ml(X, y, ml_type):
    if ml_type == 'regression':
        score_f1_r2, score_acc_mse = ml_regression(X, y)
    elif ml_type == 'binary':
        score_f1_r2, score_acc_mse = ml_binary_classification(X, y)
    elif ml_type == 'multi-class':
        score_f1_r2, score_acc_mse = ml_multiclass_classification(X, y)
    return score_f1_r2, score_acc_mse

def ml_multiclass_classification(X, y):
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    score_f1_r2 = cross_val_score(rfc, X, y, cv=cv, scoring='f1_macro').mean()
    #score_f1_r2 = scores_f1_r2.mean()
    scorer = make_scorer(accuracy_score)
    score_acc_mse = cross_val_score(rfc, X, y, cv=cv, scoring=scorer).mean()
    return score_f1_r2, score_acc_mse

def ml_binary_classification(X, y):
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    score_f1_r2 = cross_val_score(rfc, X, y, cv=cv, scoring='f1_macro').mean()
    scorer = make_scorer(accuracy_score)
    score_acc_mse = cross_val_score(rfc, X, y, cv=cv, scoring=scorer).mean()
    return score_f1_r2, score_acc_mse

def ml_regression(X, y):
    rfc = RandomForestRegressor(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    score_f1_r2 = cross_val_score(rfc, X, y, cv=cv, scoring='r2').mean()
    scoring = make_scorer(mean_squared_error)
    scores = cross_val_score(rfc, X, y, cv=cv, scoring=scoring,random_state=RANDOM_STATE)
    score_acc_mse = np.mean(scores)
    return score_f1_r2, score_acc_mse
    
"""
def impute_holoclean(dataset_path,constraints_path):

    start_time = time.time()

    hc = holoclean.HoloClean(
                              db_name='holo',
                              domain_thresh_1=0,
                              domain_thresh_2=0,
                              weak_label_thresh=0.99,
                              max_domain=10000,
                              cor_strength=0.6,
                              nb_cor_strength=0.8,
                              epochs=10,
                              weight_decay=0.01,
                              learning_rate=0.001,
                              threads=1,
                              batch_size=1,
                              verbose=True,
                              timeout=3*60000,
                              feature_norm=False,
                              weight_norm=False,
                              print_fw=True                                                              
                          ).session

    hc.load_data(dataset_path[:-4],dataset_path)
    hc.load_dcs(constraints_path)
    hc.ds.set_constraints(hc.get_dcs())
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)
    hc.setup_domain()
    featurizers = [
                    InitAttrFeaturizer(),
                    OccurAttrFeaturizer(),
                    FreqFeaturizer(),
                    ConstraintFeaturizer(),
                    ]
    hc.repair_errors(featurizers)
    end_time = time.time()

    conn = psycopg2.connect(
                      host="localhost",
                      port="5432",
                      database="holo",
                      user="holocleanuser",
                      password="abcd1234"
                                                                                                           )

   # Define a SQL query to read a table
    query = 'SELECT * FROM "' + dataset_path[:-4] + '";'

    df = pd.read_sql(query, conn)
    df_clean = encode(df)

    return df_clean, end_time - start_time
"""

"""
def impute_datawig(X):
    start_time = time.perf_counter()
    print('start datawig')
    dw_dir = os.path.join(DIR_PATH,'datawig_imputers2')
    df = datawig.SimpleImputer.complete(X)#, output_path=dw_di)#r, hpo=False, verbose=2, num_epochs=5, iterations=1)
    end_time = time.perf_counter()
    X_imputed = encode(df)
    return X_imputed, end_time-start_time
"""
def impute_kgfarm(X):
    df = pd.DataFrame(X)
    df.columns = [str(c) for c in df.columns]
    start_time = time.perf_counter()
    cl = kgfarm.recommend_cleaning_operations(df)
    print('SUGG')
    df = kgfarm.clean(df, cl.index[0],clean_outliers = True)
    end_time = time.perf_counter()
    print(df)
    X_imputed= encode(df)
    return X_imputed, end_time-start_time


def get_data(data_loc, dependent_variable):
    df = pd.read_csv(data_loc)
    independent_variables = [feature for feature in list(df.columns) if feature != dependent_variable]
    X = df[independent_variables]
    y = df[dependent_variable]
    return X, y


def get_data_baseline(data_loc, dependent_variable):
    df = pd.read_csv(data_loc)
    memory_before = psutil.Process().memory_info().rss
    start_time = time.perf_counter()
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0)
    end_time = time.perf_counter()
    mem_usage = f'{abs(psutil.Process().memory_info().rss - memory_before) / (1024 * 1024)}'
    independent_variables = [feature for feature in list(df.columns) if feature != dependent_variable]
    X = df[independent_variables]
    y = df[dependent_variable]
    X = encode(X)
    return X, y, end_time-start_time, mem_usage



if __name__ == '__main__':
   
    original_df = pd.read_csv('/mnt/Niki/datasets4.csv')
    SDIR_PATH = '/mnt/Niki/cleaning-datasets/'
    le = LabelEncoder()
    with open(os.path.join(DIR_PATH, 'benchmark_results.json'), 'w') as fh:
        for index, row in original_df.iterrows():
            data_fn = row['Data cleaning dataset']
            ml_type = row['Task']
            dependent_variable = row['dependent_variable']
            d = data_fn + '/' + data_fn + '.csv'
            data_loc = os.path.join(SDIR_PATH,d)
            #print(data_loc)
            clean_time=0
            #"""
            # For baseline
            #X_clean, y, clean_time, mem_usage = get_data_baseline(data_loc, dependent_variable)
            res =  memory_usage((get_data_baseline, (data_loc, dependent_variable,)), retval=True)
            mem_usage_max = max(res[0])
            X_clean = res[1][0]
            y = res[1][1]
            clean_time = res[1][2]
            mem_usage = res[1][3]

            if y.dtypes=='object':
                y = le.fit_transform(y.astype(str))
            #"""

            """
            X, y = get_data(data_loc, dependent_variable)
            if y.dtypes=='object':
                y = le.fit_transform(y.astype(str))
            df = pd.DataFrame(X)
            """

            """
            # For Holoclean
            
            df.to_csv(str(data_fn)+'.csv', index=False)
            memory_before = psutil.Process().memory_info().rss
            X_clean, clean_time = impute_holoclean(str(data_fn)+'.csv', SDIR_PATH + 'constraints/' +  str(data_fn) + '_constraints.txt')
            mem_usage = f'{abs(psutil.Process().memory_info().rss - memory_before) / (1024 * 1024)}'
            """
            

            """
            # For Datawig
            memory_before = psutil.Process().memory_info().rss
            res =  memory_usage((impute_datawig, (df,)), retval=True)
            mem_usage_max = max(res[0])
            mem_usage = f'{abs(psutil.Process().memory_info().rss - memory_before) / (1024 * 1024)}'
            X_clean = res[1][0]
            clean_time = res[1][1]

            """

            #impute_kgfarm(df)
            """
            # For KGFarm
            memory_before = psutil.Process().memory_info().rss
            #X_clean, clean_time = impute_kgfarm(df)
            res =  memory_usage((impute_kgfarm, (df,)), retval=True)
            mem_usage_max = max(res[0])
            mem_usage = f'{abs(psutil.Process().memory_info().rss - memory_before) / (1024 * 1024)}'
            X_clean = res[1][0]
            clean_time = res[1][1]
            """
            """
            # For gnn
            # Create an instance of SimpleImputer with strategy='mean'
            X_clean = pd.DataFrame()
            print('cleaning_op',cleaning_op)
            if cleaning_op == 2:
                imputer = SimpleImputer(strategy='most_frequent')            
                # Fit and transform the DataFrame using the imputer
                X_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                X_clean = encode(X_clean)
                
            elif cleaning_op == 1:     
                X_clean = pd.DataFrame(KNNImputer().fit_transform(df), columns=df.columns)
                X_clean = encode(X_clean)

           elif cleaning_op == 0:     
                X_clean = pd.DataFrame(IterativeImputer().fit_transform(df), columns=df.columns)
                X_clean = encode(X_clean)            
            """            
            
            score_f1_r2, score_acc_mse = ml(X_clean,y,ml_type)
            result = {
                            'data': data_fn,
                            'time':clean_time,
                            #'total memory':mem_usage,
                            'max_memory':mem_usage_max,
                            'score_f1_r2': score_f1_r2,
                            'score_acc_mse': score_acc_mse

                            }

            print(result)
            fh.write(json.dumps(result) + "\n")


