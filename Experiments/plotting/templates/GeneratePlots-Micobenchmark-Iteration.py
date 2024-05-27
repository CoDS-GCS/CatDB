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
            \\input{exp_plots/@NAME.tex}
            \\caption{@CAPTION}
        \\end{figure}  
    """
    return figure.replace("@NAME", name).replace("@CAPTION", caption)

def extract_incremental_plots(root, out):
    inc_template = read_template("Incremental-Binary_Template.tex")
    exp_template = read_template("Exp_Incremental_Template.tex")
    datasets = ["dataset_1_rnc", "dataset_2_rnc", "dataset_3_rnc", "dataset_4_rnc", "dataset_5_rnc", "dataset_6_rnc"] 

    REP_TYPE = {"S":"Conf-1",
                "SDVC":"Conf-2",
                "SMVF":"Conf-3",
                "SSN":"Conf-4",
                "SCV":"Conf-5",
                "SDVCMVF":"Conf-6",
                "SDVCSN":"Conf-7",
                "SMVFSN":"Conf-8",
                "SMVFCV":"Conf-9",
                "SSNCV":"Conf-10",
                "ALL":"Conf-11",
                "CatDB":"CatDB"}
    

    plots = []
    width_ratio = 0.117
    h = 0
    for ds in datasets:
        path = f"results/Experiment1_LLM_Pipe_Gen_{ds}.dat"
        df = read_dataset(f"../{path}")

        df = df.sort_values(by=['accuracy'], ascending=False)
        rps = df['config'].tolist() #set(df['config'].unique())
 

        configs = []
        config_label = []
        for rp in rps:
            configs.append(rp)
            config_label.append(REP_TYPE[rp]) 
        # for rp in REP_TYPE:
        #     if rp in rps:
        #         configs.append(rp)
        #         config_label.append(REP_TYPE[rp]) 

        if len(configs) >6:
            R = 25
        configs = ",".join(configs)
        config_label = ",".join(config_label)

        plot = inc_template.replace("@DATASET",path).replace("@CONFIGS", configs).replace("@CONFIGLABELS", config_label).replace("@ROTATE",f"{R}")
        plot_name = f"CatDB-Incremental-Binary_{ds}"
        save_template(plot,f"{out}/{plot_name}.tex")

        plots.append(get_figure(name=plot_name, caption = ds.replace("_","\\_")))
    
    plots_tex = "\n".join(plots)
    inc_plots = exp_template.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp_incremental.tex")                
  

def extract_dataset_iterations(root, out):
    # datasets = [("Balance-Scale","multiclass"),("Breast-w","binary"),("CMC","multiclass"),("Credit-g","binary"),("Diabetes","binary"),("Tic-Tac-Toe","binary"),("Eucalyptus","multiclass"),("PC1","binary"),("Airlines","binary"),("Jungle-Chess","multiclass")]

    datasets = [("Balance-Scale","multiclass"),("Breast-w","binary"),("CMC","multiclass"),("Tic-Tac-Toe","binary"),("Eucalyptus","multiclass")]

    template = read_template("Experiment1-Iteration_Template.tex")
    exp_micorbenchmark = read_template("Exp_Micorbenchmark_Template.tex")

    plots = []
    for (ds,task_type) in datasets:
        if task_type == 'binary':
            measure = "auc"
        elif task_type == 'multiclass':
            measure = "auc_ovr" 
        else:
            measure = ""

        plot_it = template.replace("@DATASET",ds).replace("@MEASURE",measure)        
        plot_name = f"Experiment1-Iteration_{ds}"
        save_template(plot_it, f"{out}/{plot_name}.tex")        

        ds_title = ds.replace("_", "\_")
        plots.append(get_figure(plot_name, f"{ds_title}"))

    plots_tex = "\n".join(plots)
    inc_plots = exp_micorbenchmark.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp1_microbenchmark-iteration.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../exp_plots"

    #extract_incremental_plots(root=root, out=out)
    extract_dataset_iterations(root=root, out=out)

    
    