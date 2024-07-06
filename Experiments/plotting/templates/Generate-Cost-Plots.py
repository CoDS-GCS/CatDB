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
    datasets = ["Balance-Scale", "Breast-w","CMC",
                  "Credit-g",
                  "Diabetes",
                  "Tic-Tac-Toe",
                  "Eucalyptus",
                  "PC1",
                  "Jungle-Chess",
                  "Higgs",
                  "Skin",
                  "Traffic",
                  "Walking-Activity",
                  "Black-Friday",
                  "Bike-Sharing",
                  "House-Sales",
                  "NYC",
                  "Airlines-DepDelay"
                ]    
   
    plot_template = read_template("Experiment1-Cost-Template.tex")
    exp_micro = read_template("Exp_Dataset_Attribute_Template.tex")
    
    ytick_1="0,5000,10000,15000,20000,25000,30000,35000,40000"
    yticklabels_1="0, 5k, 10k, 15k, 20k, 25k,30k,35k,40k" 

    ytick_2="0,2000,4000,6000,8000,10000,12000,14000,16000"
    yticklabels_2="0, 2k, 4k, 6k, 8k, 10k,12k,14k,16k"

    ytick_3="0,1000,2000,3000,4000,5000,6000,7000,8000"
    yticklabels_3="0, 1k, 2k, 3k, 4k, 5k,6k,7k,8k"
   
    plots = []
    for ds in datasets:
        if ds in {"Higgs","Traffic","Walking-Activity","House-Sales","Bike-Sharing","Airlines-DepDelay"}:
           ytick = ytick_1
           yticklabels = yticklabels_1
        elif ds in {"Skin","Black-Friday","NYC"}:
              ytick = ytick_2
              yticklabels = yticklabels_2 
        else:
             ytick = ytick_3
             yticklabels = yticklabels_3 

        if ds in {"Black-Friday","Bike-Sharing","House-Sales","NYC","Airlines-DepDelay"}:
             plot_template = read_template("Experiment1-Cost-Template-Reg.tex")
        else:
             plot_template = read_template("Experiment1-Cost-Template.tex")

                

        plot_cost = plot_template.replace("@DATASET",ds).replace("@YTICKLABELs",yticklabels).replace("@YTICK",ytick)
        plot_name = f"Experiment1-Cost-{ds}"
        save_template(plot_cost, f"{out}/{plot_name}.tex")        

        plots.append(get_figure(plot_name, f"Experiment1-Cost-{ds}"))

        print("\\subfigure["+ds+"]{\\label{exp1a}\includegraphics[scale=0.55]{plots/Experiment1-Cost-"+ds+"}}")

    plots_tex = "\n".join(plots)    
    inc_plots = exp_micro.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp1_micorbenchmark2.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../Experiment1"

    extract_performance_plots(root=root, out=out)


    
    