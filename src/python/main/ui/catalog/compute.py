import pandas as pd


def compute_statistics(catalog):
    schema_info = catalog.schema_info
    nbools = 0
    nstrings = 0
    nints = 0
    nfloats = 0

    for k in schema_info.keys():
        if schema_info[k] == "bool":
            nbools += 1
        elif schema_info[k] == "str":
            nstrings += 1
        elif schema_info[k] == "float":
            nfloats += 1
        elif schema_info[k] == "int":
            nints += 1
    ncols = nbools + nstrings + nfloats + nints

    df = pd.DataFrame({'count':[nstrings, nints, nfloats, nbools],
                         'pct':[nstrings/ncols, nints/ncols, nfloats/ncols, nbools/ncols]},
                        index=["String", "Integer", "Float", "Bool"])
    return df
