from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score, log_loss, r2_score
from runner.LogResults import LogResults


def evaluate_score(pred, gt, metric):
    if metric == "accuracy":
        return accuracy_score(gt, pred)
    elif metric == "f1":
        unique_classes = sorted(list(np.unique(gt)))
        if 1 in unique_classes and 0 in unique_classes:
            pos_label = 1
        else:
            pos_label = unique_classes[0] if len(unique_classes) == 2 else None
        return f1_score(gt, pred, pos_label=pos_label)
    elif metric == "f1 weighted":
        return f1_score(gt, pred, average="weighted")
    elif metric == "auc":
        return roc_auc_score(gt, pred)
    elif metric == "log loss":
        return log_loss(gt, pred)
    
    elif metric == "auc ovo":
        return roc_auc_score(gt, pred, multi_class='ovo')
    elif metric == "auc ovr":
        return roc_auc_score(gt, pred, multi_class='ovr')
    elif metric == "r2":
        return r2_score(np.log1p(gt), np.log1p(pred), squared=False)
    elif metric == "rmse":
        return mean_squared_error(gt, pred, squared=False)
    elif metric == "log rmse":
        return mean_squared_error(np.log1p(gt), np.log1p(pred), squared=False)
    else:
        raise ValueError(f"Metric {metric} not supported")


def node_evaluate_score_sela(node, task_type:str, log_results:LogResults):
    test_preds = node.get_and_move_predictions("test")["target"]
    test_gt = node.get_gt("test")["target"]

    # train_preds = node.get_and_move_predictions("train")["target"]
    # train_gt = node.get_gt("train")["target"]
    
    if task_type == "binary":
    #    log_results.train_accuracy = evaluate_score(train_preds, train_gt, "accuracy")
    #    log_results.train_f1_score = evaluate_score(train_preds, train_gt, "f1")
    #    log_results.train_auc = evaluate_score(train_preds, train_gt, "auc") 

       log_results.test_accuracy = evaluate_score(test_preds, test_gt, "accuracy")
       log_results.test_f1_score = evaluate_score(test_preds, test_gt, "f1")
       log_results.test_auc = evaluate_score(test_preds, test_gt, "auc") 
       
    elif task_type == "multiclass":
        # log_results.train_accuracy = evaluate_score(train_preds, train_gt, "accuracy") 
        # log_results.train_f1_score = evaluate_score(train_preds, train_gt, "f1 weighted")            
        # log_results.train_log_loss = evaluate_score(train_preds, train_gt, "log loss") 
        # log_results.train_auc_ovo = evaluate_score(train_preds, train_gt, "auc ovo")  
        # log_results.train_auc_ovr = evaluate_score(train_preds, train_gt, "auc ovr")  

        log_results.test_accuracy = evaluate_score(test_preds, test_gt, "accuracy") 
        log_results.test_f1_score = evaluate_score(test_preds, test_gt, "f1 weighted")            
        log_results.test_log_loss = evaluate_score(test_preds, test_gt, "log loss") 
        log_results.test_auc_ovo = evaluate_score(test_preds, test_gt, "auc ovo")  
        log_results.test_auc_ovr = evaluate_score(test_preds, test_gt, "auc ovr") 
           
    elif task_type == "regression":
        # log_results.train_rmse  = evaluate_score(train_preds, train_gt, "rmse")  
        # log_results.train_r_squared = evaluate_score(train_preds, train_gt, "r2")  

        log_results.test_rmse  = evaluate_score(test_preds, test_gt, "rmse")  
        log_results.test_r_squared = evaluate_score(test_preds, test_gt, "r2")  