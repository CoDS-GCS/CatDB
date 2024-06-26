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
            \\input{Experiment1/@NAME.tex}
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
                  ("Jungle-Chess","multiclass")
                ]    
   
    plot_template = read_template("Experiment1-Performance-Template.tex")
    exp_micro = read_template("Exp_Dataset_Attribute_Template.tex")
    
    datasets = []
    for ds in datasetIDs:
        ds_name, task_type = ds
        datasets.append((ds_name, ds_name, task_type))


    plots = []
    for (ds,ds_title, task_type) in datasets:
        metric="auc"
        metric_lbl = ""        
        if task_type == "multiclass":
            metric="auc_ovr"
            metric_lbl="-ovr"

        plot_performance = plot_template.replace("@DATASET",ds_title).replace("@METRIC",metric).replace("@MLBL", metric_lbl)

        plot_name = f"Experiment1-Performance-{ds}"
        save_template(plot_performance, f"{out}/{plot_name}.tex")        

        plots.append(get_figure(plot_name, f"Experiment1-Performance-{ds}"))

    plots_tex = "\n".join(plots)    
    inc_plots = exp_micro.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp1_micorbenchmark2.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../Experiment1"

    extract_performance_plots(root=root, out=out)


    
    