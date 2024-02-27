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
                "ALL":"Conf-11"}
    

    plots = []
    width_ratio = 0.117
    for ds in datasets:
        path = f"results/Experiment1_LLM_Pipe_Gen_{ds}.dat"
        df = read_dataset(f"../{path}")
        rps = set(df['config'].unique())
        configs = []
        config_label = []
        for rp in REP_TYPE:
            if rp in rps:
                configs.append(rp)
                config_label.append(REP_TYPE[rp]) 

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

def extract_dataset_propertis_mv(root, out):
    #datasets = ["dataset_1_rnc", "dataset_2_rnc", "dataset_3_rnc", "dataset_4_rnc", "dataset_5_rnc", "dataset_6_rnc"]
    datasets = [f"dataset_{i}_rnc" for i in range(1,7)]
    #datasets = ["Microsoft", "delays_zurich_transport", "federal_election"]
    miss_value_template = read_template("Missing_Distinct_Values_Template.tex")
    distinct_value_template = read_template("Distinct_Values_Template.tex")
    exp_statistics = read_template("Exp_Dataset_Attribute_Template.tex")

    plots = []
    pindex = 0
    for ds in datasets:
        path = f"results/statistics/{ds}/statistics_2.csv"
        df = read_dataset(f"../{path}")
        ncols = len(df)
        rc = (1/float(ncols))/3
        ymax = max(df["number_rows"])
        plot_mv = miss_value_template.replace("@DATASET",path).replace("@XMAX",f"{ncols}").replace("@BARWIDTH",f"{rc}").replace("@YMAX",f"{ymax}")
        plot_dv = distinct_value_template.replace("@DATASET",path).replace("@XMAX",f"{ncols}").replace("@BARWIDTH",f"{rc}")

        plot_name_mv = f"Statistics_{ds}_mvdc"
        save_template(plot_mv, f"{out}/{plot_name_mv}.tex")

        # plot_name_dv = f"Statistics_{ds}_dv"
        # save_template(plot_dv, f"{out}/{plot_name_dv}.tex")

        ds_title = ds.replace("_", "\_")
        pindex +=1
        if pindex % 10 == 0:
            plots_tex = "\n".join(plots)
            si = pindex / 10
            inc_plots = exp_statistics.replace("@PLOT", plots_tex)
            save_template(inc_plots,f"../exp_statistics{int(si)}.tex")
            plots = []

        plots.append(get_figure(plot_name_mv, f"{ds_title}"))
        #plots.append(get_figure(plot_name_dv))

    plots_tex = "\n".join(plots)
    si = pindex / 10 +1
    inc_plots = exp_statistics.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp_statistics{int(si)}.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../exp_plots"

    #extract_incremental_plots(root=root, out=out)
    extract_dataset_propertis_mv(root=root, out=out)

    
    