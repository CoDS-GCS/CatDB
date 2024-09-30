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
            \\input{Experiment0/@NAME.tex}
            \\caption{@CAPTION}
        \\end{figure}  
    """
    return figure.replace("@NAME", name).replace("@CAPTION", caption)

def getYTicks(max:int):
    ytick ="{}"
    yticklabels= "{}"

    if max <=100:
        ytick = "{0,20,40,80,100}"
        yticklabels= "{0,20,40,80,100}"
    
    elif max <=150:
        ytick = "{0,30,60,90,120,150}"
        yticklabels= "{0,30,60,90,120,150}"  

    elif max <=500:
        ytick = "{0,100,200,300,400,500}"
        yticklabels= "{0,100,200,300,400,500}"      

    elif max <=1000:
        ytick = "{0,200,400,600,800,1000}"
        yticklabels= "{0,200,400,600,800,1000}"    

    elif max <=1500:
        ytick = "{0,300,600,900,1200,1500}"
        yticklabels= "{0,300,600,900,1200,1500}"

    elif max <=2000:
        ytick = "{0,500,1000,1500,2000}"
        yticklabels= "{0,5e2,10e2,15e2,2e3}"

    elif max <=30000:
        ytick = "{0,7000,14000,21000,28000}"
        yticklabels= "{0,7e3,14e3,21e3,28e3}"

    elif max <=51000:
        ytick = "{0,10000,20000,30000,40000,50000}"
        yticklabels= "{0,1e4,2e4,3e4,4e4,5e4}"

    elif max <=150000:
        ytick = "{0,30000,60000,90000,120000,150000}"
        yticklabels= "{0,3e4,6e4,9e4,12e4,15e4}"       
            
    elif max <=400000:
        ytick = "{0,100000,200000,300000,400000}"
        yticklabels= "{0,1e5,2e5,3e5,4e5}" 
    
    elif max <=500000:
        ytick = "{0,100000,200000,300000,400000,500000}"
        yticklabels= "{0,1e5,2e5,3e5,4e5,5e5}"

    elif max <=1000000:
        ytick = "{0,200000,400000,600000,800000,1000000}"
        yticklabels= "{0,2e5,4e5,6e5,8e5,1e6}"   

    elif max <=1500000:
        ytick = "{0,300000,600000,900000,1200000,1500000}"
        yticklabels= "{0,3e5,6e5,9e5,12e5,15e5}"   

    elif max <=2000000:
        ytick = "{0,500000,1000000,1500000,2000000}"
        yticklabels= "{0,5e5,10e5,15e5,20e5}" 

    elif max <=5000000:
        ytick = "{0,1000000,2000000,3000000,4000000,5000000}"
        yticklabels= "{0,1e6,2e6,3e6,4e6,5e6}"     

    elif max <=11000000:
        ytick = "{0,2000000,4000000, 6000000, 8000000, 10000000}"
        yticklabels= "{0,2e6,4e6,6e6,8e6, 10e6}"   

    return ytick, yticklabels                       
    
def getXTicks(max:int):
    xtick ="{}"
    if max <=9 :
        xtick ="{1,2,3,4,5,6,7,8,9}"

    elif max <=10 :
        xtick ="{2,4,6,8,10}"    

    elif max <=15 :
        xtick ="{3,6,9,12,15}"
   
    elif max <=20 :
        xtick ="{1,5,10,15,20}"  

    elif max <=25 :
        xtick ="{1,5,10,15,20,25}" 

    elif max <=30 :
        xtick ="{1,6,12,18,24,30}"
    
    elif max <=40 :
        xtick ="{1,10,20,30,40}"

    elif max <=50 :
        xtick ="{1,10,20,30,40,50}"
        
    elif max <=60 :
        xtick ="{1,15,30,45,60}" 

    elif max <=80 :
        xtick ="{1,20,40,60,80}"

    elif max <=100 :
        xtick ="{1,20,40,60,80,100}"
    
    elif max <=200 :
        xtick ="{1,50,100,150,200}"

    return xtick

def extract_dataset_propertis_mv(root, out):
    # datasets = ["Breast-w","CMC","Credit-g","Diabetes","Tic-Tac-Toe","Nomao","Volkert","Gas-Drift","Jungle-Chess","Higgs","Skin","Traffic","Walking-Activity","Black-Friday","Bike-Sharing","House-Sales","NYC"]


    datasets = ["Midwest-Survey","WiFi","Utility","EU-IT","Etailing"]

    miss_value_template = read_template("Missing_Distinct_Values_Template.tex")
    exp_statistics = read_template("Exp_Dataset_Attribute_Template.tex")

    plots = []
    pindex = 0
    for ds in datasets:
        print(ds)
        path = f"../archive/VLDB2025/statistics/{ds}_statistics.dat"
        df = read_dataset(f"../{path}")
        ncols = len(df)
        rc = (1/float(ncols))/3
        ymax = max(df["number_rows"])
        ytick, yticklabels = getYTicks(ymax)
        xtiks = getXTicks(ncols)
        plot_mv = miss_value_template.replace("@DATASET",path).replace("@XMAX",f"{ncols}").replace("@BARWIDTH",f"{rc}").replace("@YMAX",f"{ymax}").replace("@YTICKLABELS",yticklabels).replace("@YTICK",ytick).replace('@XTICK',xtiks)      
        
        plot_name_mv = f"Statistics_{ds}"
        save_template(plot_mv, f"{out}/{plot_name_mv}.tex")        

        ds_title = ds.replace("_", "\_")
        # pindex +=1
        # if pindex % 11 == 0:
        #     plots_tex = "\n".join(plots)
        #     si = pindex / 11
        #     inc_plots = exp_statistics.replace("@PLOT", plots_tex)
        #     save_template(inc_plots,f"../exp_statistics{int(si)}.tex")
        #     plots = []

        plots.append(get_figure(plot_name_mv, f"{ds_title}"))

    plots_tex = "\n".join(plots)
    si = pindex / 10 +1
    inc_plots = exp_statistics.replace("@PLOT", plots_tex)
    save_template(inc_plots,f"../exp_statistics{int(si)}.tex")


if __name__ == '__main__':
    
    root = "../results"
    out = "../Experiment0"

    extract_dataset_propertis_mv(root=root, out=out)

    
    