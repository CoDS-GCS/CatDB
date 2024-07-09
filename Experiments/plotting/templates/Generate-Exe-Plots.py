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
            \\input{Experiment2/@NAME.tex}
            \\caption{@CAPTION}
        \\end{figure}  
    """
    return figure.replace("@NAME", name).replace("@CAPTION", caption)

def extract_performance_plots(root, out):
    datasets = ["Skin",
                #"Diabetes",
                "Tic-Tac-Toe",
                #"Breast-w",
                #"Credit-g",
                "Higgs",
                #"Nomao",
                "Balance-Scale",
                "Walking-Activity",
                "Jungle-Chess",
                "CMC",
                "Traffic",
                #"Gas-Drift",
                #"Volkert",
                "Black-Friday",
                "Bike-Sharing",
                "NYC",
                "House-Sales",
                   ]    
   
    plot_template = read_template("Experiment2-Exe-Template.tex")
    exp_micro = read_template("Exp_Dataset_Attribute_Template.tex")
    
    ytick_1="0,1,2,3,4,5,6,7,8,9,10"
    yticklabels_1="0,1,2,3,4,5,6,7,8,9,10" 

    ytick_2="0,10,20,30,40,50,60,70,80,90,100"
    yticklabels_2="0,10,20,30,40,50,60,70,80,90,100"

    ytick_3="0.1,1,10,100,1000,10000"
    yticklabels_3="0,1,10,$10^2$,$10^3$,$10^4$"
   
    plots = []
    for ds in datasets:
        if ds in {"Higgs","Nomao","Walking-Activity","Jungle-Chess","Traffic","Gas-Drift","Volkert"}:
           ytick = ytick_3
           yticklabels = yticklabels_3    
           plot_template = read_template("Experiment2-Exe-Template-Log.tex")            

        elif ds in {"Black-Friday", "Bike-Sharing", "House-Sales", "NYC"}:
             ytick = ytick_1
             yticklabels = yticklabels_1 
             plot_template = read_template("Experiment2-Exe-Template_Reg.tex")

        else:
             ytick = ytick_1
             yticklabels = yticklabels_1 
             plot_template = read_template("Experiment2-Exe-Template.tex")     
           

        plot_cost = plot_template.replace("@DATASET",ds).replace("@YTICKLABELs",yticklabels).replace("@YTICK",ytick)
        plot_name = f"Experiment2-Exe-{ds}"
        save_template(plot_cost, f"{out}/{plot_name}.tex")        

        plots.append(get_figure(plot_name, f"Experiment2-Exe-{ds}"))

        print("\\subfigure["+ds+"]{\\label{exp1a}\includegraphics[scale=0.55]{plots/Experiment2-Exe-"+ds+"}}")

    plots_tex = "\n".join(plots)    
    inc_plots = exp_micro.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp_exe_benchmark_it_1.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../Experiment2"

    extract_performance_plots(root=root, out=out)


    
    