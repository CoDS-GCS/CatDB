import sys
import pandas as pd
from argparse import ArgumentParser
import yaml
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)    
    parser.add_argument('--statistic-file-name', type=str, default=None) 
       
    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

     # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            try:
                args.data_source_train_path = "../../" + config_data[0].get('dataset').get('train').replace("{user}/", "")
                args.data_source_test_path = "../../" + config_data[0].get('dataset').get('test').replace("{user}/","")
                args.data_source_all_path = f"../../data/{args.dataset_name}/{args.dataset_name}.csv" 
            except Exception as ex:
                raise Exception(ex)
            
        except yaml.YAMLError as ex:
            raise Exception(ex)
   
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    # Read dataset
    data_path = [(args.data_source_all_path, "Original Dataset"),
                 (args.data_source_train_path, "Train Dataset"),
                 (args.data_source_test_path, "Test Dataset"),
                 ]
    
    df = pd.read_csv(args.data_source_all_path)
    classes = df[args.target_attribute].unique() 

    # set default values
    labels = ['Original Dataset', 'Train Dataset', 'Test Dataset']
    classes_ds = dict()
    for ds in labels:
       tmp_classes = dict()
       for c in classes:
            tmp_classes[c] = 0
       classes_ds[ds] = tmp_classes
      

    for (dsp, lbl) in data_path:
        df = pd.read_csv(dsp)
        tmp_classes = classes_ds[lbl]  
        for c in classes:
            tmp_classes[c] = len(df[df[args.target_attribute]  == c]) 

   
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots(figsize=(int(2.5 * len(classes)),3))
    a = width/2+  width * (len(classes) -2) 

    ci = 1  
    max_y = 0
    ratio = len(classes)  
    for c in classes:
      values = []
      for l in labels:
        values.append(classes_ds[l][c])
      
      w = width * (ci-1)
      rects = ax.bar(x-a+w, values, width, label=f'Class {ci}')  
      ax.bar_label(rects, padding=3) 
      ci +=1
      max_y = max(max_y, max(values))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(args.target_attribute)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, max_y+max_y*0.1])
    ax.legend()
    fig.tight_layout()
    #plt.show()

    fig.savefig(f"{args.statistic_file_name}", bbox_inches='tight')    