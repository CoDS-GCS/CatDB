import os
import pandas as pd

def compare_orig_and_clean_updates(data_source_clean_path, orig_fname: str, clean_fname: str):
    # check the data clean is available:
    if os.path.isfile(data_source_clean_path):
        print("assss")
        orig_data = pd.read_csv(orig_fname)
        clean_data = pd.read_csv(clean_fname)

        print(orig_data.columns)
        print(clean_data.columns)

        cols = orig_data.columns.to_list()
        total_refined_cols = 0
        refine_cols = []
        total_diffs = 0
        for c in cols:
            on = orig_data[c].nunique()
            cn = clean_data[c].nunique()
            if on - cn != 0:
                total_refined_cols += 1
                refine_cols.append(f"{c}#{on}#{cn}#{on - cn}")
                total_diffs += on - cn

        return {"total_refined_cols": total_refined_cols, "refine_cols": ";".join(refine_cols),
                "total_diffs": total_diffs}

    return None

if __name__ == '__main__':
    root_path = "/home/saeed/Documents/Github/CatDB/Experiments/data/Violations/"
    a = compare_orig_and_clean_updates(data_source_clean_path=f"{root_path}/Violations_Google_clean.csv",
                                       orig_fname=f"{root_path}/Violations.csv",
                                       clean_fname=f"{root_path}/Violations_Google_clean.csv")

    print(a)