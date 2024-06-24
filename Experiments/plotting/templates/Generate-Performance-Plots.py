import os
import sys
import pandas as pd
import numpy as np


def read_template(temp_fname):
    # Open a file: file
    file = open(temp_fname,mode='r')
 
    # read all lines at once
    latex_script = file.read()
 
    # close the file
    file.close()

    return latex_script

def read_dataset(fname:str):
    df = pd.read_csv(fname)
    return df

def save_template(latex_script:str, fname:str):
    f = open(f"{fname}", 'w')
    f.write(latex_script)
    f.close()


def get_figure(name:str, caption:str):
    figure = """  
    \\tikzsetnextfilename{@NAME} 
        \\begin{figure}[!ht]
            \\centering
            \\input{aaa/@NAME.tex}
            \\caption{@CAPTION}
        \\end{figure}  
    """
    return figure.replace("@NAME", name).replace("@CAPTION", caption)


def extract_performance_plots(root, out):
    datasetIDs = [("Balance-Scale","multiclass"),
                  ("Breast-w","binary"),
                  ("CMC","multiclass"),
                  ("Credit-g","binary"),
                  ("Diabetes","binary"),
                  ("Tic-Tac-Toe","binary"),
                  ("Eucalyptus","multiclass"),
                  ("PC1","binary"),
                  ("Airlines","binary"),
                  ("Jungle-Chess","multiclass")
                ]
    # ,
    #               ("Higgs","oml_dataset_11_rnc",11),
    #               ("Skin","oml_dataset_12_rnc",12),
    #               ("Click-Prediction","oml_dataset_13_rnc",13),
    #               ("Census-Augmented","oml_dataset_14_rnc",14),
    #               ("Heart-Statlog","oml_dataset_15_rnc",15),
    #               ("KDDCup99","oml_dataset_16_rnc",16),
    #               ("Road-Safety","oml_dataset_17_rnc",17),
    #               ("Drug-Directory","oml_dataset_18_rnc",18),
    #               ("Traffic","oml_dataset_19_rnc",19),
    #               ("Walking-Activity","oml_dataset_20_rnc",20),
    #               ("Black-Friday","oml_dataset_21_rnc",21),
    #               ("Bike-Sharing","oml_dataset_22_rnc",22),
    #               ("House-Sales","oml_dataset_23_rnc",23),
    #               ("NYC","oml_dataset_24_rnc",24),
    #               ("Airlines-DepDelay","oml_dataset_25_rnc",25),
    #               ("Adult","oml_dataset_26_rnc",26)
    
   
    plot_template = read_template("Experiment1-Performance-Template.tex")
    exp_micro = read_template("Exp_Dataset_Attribute_Template.tex")
    
    datasets = []
    for ds in datasetIDs:
        ds_name, task_type = ds
        datasets.append((ds_name, f"{ds_name}-rnc", task_type))


    plots = []
    for (ds,ds_title, task_type) in datasets:
        metric="auc"
        if task_type == "multiclass":
            metric="auc_ovr"

        plot_performance = plot_template.replace("@DATASET",ds_title).replace("@METRIC",metric)

        plot_name = f"Experiment1-Performance-{ds}"
        save_template(plot_performance, f"{out}/{plot_name}.tex")        

        plots.append(get_figure(plot_name, f"Experiment1-Performance-{ds}"))

    plots_tex = "\n".join(plots)    
    inc_plots = exp_micro.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp1_micorbenchmark2.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../aaa"

    extract_performance_plots(root=root, out=out)


    
    