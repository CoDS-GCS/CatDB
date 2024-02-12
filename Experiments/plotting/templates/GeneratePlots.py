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


def get_figure(name:str):
    figure = """
    \\tikzsetnextfilename{@NAME}
        \\begin{figure}[h]
            \\centering
            \\input{exp_plots/@NAME.tex}
            \\caption{Accuracy}
        \\end{figure}  
    """
    return figure.replace("@NAME", name)

def extract_incremental_plots(root, out):
    inc_binary = f'{root}/Experiment_CatDB_Micro_Benchmark.dat'
    inc_template = read_template("Incremental-Binary_Template.tex")
    exp_template = read_template("Exp_Incremental_Template.tex")

    df_micro = read_dataset(inc_binary)
    datasets = df_micro['dataset'].unique()

    plots = []
    for ds in datasets:
        plot = inc_template.replace("@DATASET",ds)
        plot_name = f"CatDB-Incremental-Binary_{ds}"
        save_template(plot,f"{out}/{plot_name}.tex")

        plots.append(get_figure(plot_name))
    
    plots_tex = "\n".join(plots)
    inc_plots = exp_template.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp_incremental.tex")

def extract_dataset_propertis_mv(root, out):
    datasets = ["dataset_1_rnc", "dataset_2_rnc", "dataset_3_rnc", "dataset_4_rnc", "dataset_5_rnc", "dataset_6_rnc"]
    #datasets = ["Microsoft", "delays_zurich_transport", "federal_election"]
    miss_value_template = read_template("Missing_Values_Template.tex")
    distinct_value_template = read_template("Distinct_Values_Template.tex")
    exp_statistics = read_template("Exp_Dataset_Attribute_Template.tex")

    plots = []
    for ds in datasets:
        path = f"results/statistics/{ds}/statistics_2.csv"
        df = read_dataset(f"../{path}")
        ncols = len(df)
        rc = (1/float(ncols))/3
        plot_mv = miss_value_template.replace("@DATASET",path).replace("@XMAX",f"{ncols}").replace("@BARWIDTH",f"{rc}")
        plot_dv = distinct_value_template.replace("@DATASET",path).replace("@XMAX",f"{ncols}").replace("@BARWIDTH",f"{rc}")

        plot_name_mv = f"Statistics_{ds}_mv"
        save_template(plot_mv, f"{out}/{plot_name_mv}.tex")

        plot_name_dv = f"Statistics_{ds}_dv"
        save_template(plot_dv, f"{out}/{plot_name_dv}.tex")

        plots.append(get_figure(plot_name_mv))
        plots.append(get_figure(plot_name_dv))

    plots_tex = "\n".join(plots)
    inc_plots = exp_statistics.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp_statistics.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../exp_plots"

    extract_incremental_plots(root=root, out=out)
    #extract_dataset_propertis_mv(root=root, out=out)

    
    