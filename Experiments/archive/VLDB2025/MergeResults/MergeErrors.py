import pandas as pd
from MergeResults import load_merge_all_errors


if __name__ == '__main__':
    
    root_path = "../results/"        
    df_errors = load_merge_all_errors(root_path=root_path)

    # df_errors.loc[df_errors['llm_model']=='gemini-1.5-pro-latest'] = 'gemini-1.5'
    # df_errors.loc[df_errors['llm_model']=='gemini-1.5-pro-exp-0827'] = 'gemini-1.5'
    # df_errors.loc[df_errors['llm_model']=='llama-3.1-70b-versatile'] = 'llama3'
    # df_errors.loc[df_errors['llm_model']=='llama3-70b-8192'] = 'llama3'


    # llms = df_errors["llm_model"].unique()
    err_class = df_errors["error_class"].value_counts().keys() #df_errors["error_class"].unique()
    
    
    # err_class_tmp = []
    # for i in range(len(err_class)-1, 0, -1):
    #     err_class_tmp.append(err_class[i])

    # err_class = err_class_tmp
    df_err_resul = pd.DataFrame(columns = ["error_class","llm_model","ratio","count","total"])
    # #print(err_class)

    
    for llm in {"gemini-1.5", "llama3"}:
        errs = dict()
        if llm == "gemini-1.5":
            df_tmp = df_errors.loc[(df_errors['llm_model'] == 'gemini-1.5-pro-latest') |
                                   (df_errors['llm_model'] == 'gemini-1.5-pro-exp-0827')]
        elif llm == 'llama3':
              df_tmp = df_errors.loc[(df_errors['llm_model'] == 'llama-3.1-70b-versatile') |
                                   (df_errors['llm_model'] == 'llama3-70b-8192')]  

        hide_list = []
        d = len(df_tmp)
        print(f"{llm}  >> {d}")
        flag = False
        for e in err_class:
            if e == "OpenMLError":
                continue
            count = len(df_tmp.loc[df_tmp['error_class'] == e])
            errs[e] = count
            ratio = count / d
            ratio = ratio * 100

            cindex = len(df_err_resul)
            row_entry = [e, llm, f"{ratio:0.3f}", count, d]
            df_err_resul.loc[cindex] = row_entry
            print(f"{e} >>>>>>>> {count}")
                                

    # print(df_err_resul)


    df_err_resul.to_csv(f"{root_path}/ErrorResults.csv", index=False) 
    
