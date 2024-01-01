import sys
import pandas as pd



if __name__ == '__main__':
    df = pd.read_csv("/home/saeed/Documents/Github/CatDB/Experiments/data/dorothea/dorothea_train.csv")
    print( df.shape)