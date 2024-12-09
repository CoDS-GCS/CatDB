import pandas as pd
import os
from util.FileHandler import read_text_file_line_by_line
from dask.dataframe import from_pandas
import dask.dataframe as dd
import dask.array as da
from typing import Any, Dict


def compute_results(pipegen):
    df = pd.read_csv(pipegen["result_output_path"], low_memory=False, encoding='utf-8')
    try:
        df_errors = pd.read_csv(pipegen["error_output_path"], low_memory=False, encoding='utf-8')
        err_class = df_errors["error_class"].value_counts().keys()
        error_count = dict()
        for e in err_class:
            count = len(df_errors.loc[df_errors['error_class'] == e])
            error_count[e] = count
        keys = list(error_count.keys())
        count = []
        pct = []
        for k in keys:
            count.append(error_count[k])
            pct.append(error_count[k] / len(df_errors) * 100)
        df_err_resul = pd.DataFrame({'count': count,'pct': pct }, index=keys)
    except Exception as ex:
        df_err_resul = pd.DataFrame({'count': [], 'pct': []}, index=[])
    time_cols = ["number_iteration", "time_catalog_load", "time_pipeline_generate", "time_total", "time_execution"]
    performance_cols = ["number_iteration", "train_auc", "train_auc_ovo", "train_auc_ovr", "train_accuracy",
                        "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc", "test_auc_ovo",
                        "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared",
                        "test_rmse"]
    cost_cols = ["prompt_token_count", "all_token_count"]
    df_gen_tmp = df.loc[(df['status'] == True) & (df['dataset_name'] == pipegen["dataset_name"]) & (
                df['operation'] == 'Gen-and-Verify-Pipeline')]
    df_run_tmp = df.loc[(df['status'] == True) & (df['dataset_name'] == pipegen["dataset_name"]) & (df['operation'] == 'Run-Pipeline')]

    df_gen = df_gen_tmp[time_cols]
    df_run = df_run_tmp[["number_iteration", "time_execution"]].reset_index(drop=True)
    df_performance = df_run_tmp[performance_cols]
    df_cost = df_run_tmp[cost_cols]
    df_cost["Error Tokens"] = df_cost["all_token_count"] - df_cost["prompt_token_count"]
    df_cost = df_cost.rename(columns={"prompt_token_count": "Prompt Tokens"}).reset_index(drop=True)
    df_cost.index = [str(i) for i in range(1, len(df_cost) + 1)]
    df_cost = df_cost.drop('all_token_count', axis=1)


    df_gen["number_iteration"] = [str(i) for i in range(1, len(df_gen) + 1)]
    df_gen = df_gen.rename(columns={'time_execution': 'time_verify'}).reset_index(drop=True)

    df_run["number_iteration"] = [str(i) for i in range(1, len(df_run) + 1)]
    df_performance["number_iteration"] = [str(i) for i in range(1, len(df_performance) + 1)]


    # all prompts
    files = [f for f in os.listdir(pipegen["output_path"])]
    file_prompt = None
    run_files = []
    for f in files:
        f = f'{pipegen["output_path"]}/{f}'
        if f.endswith("Random-0-SHOT-No-iteration-1.prompt") and file_prompt is None:
            file_prompt = f
        elif f.endswith("_RUN.py"):
            run_files.append(f)

    try:
        with open(file_prompt) as f:
            prompt_lines = f.readlines()
    except Exception as ex:
        raise Exception(f"Error in reading file:\n {ex}")

    prompt = {"system_prompt": [], "usr_prompt": []}
    key = None
    for text in prompt_lines:
        if text.startswith("SYSTEM MESSAGE:"):
            key = "system_prompt"
            continue
        elif text.startswith("PROMPT TEXT:"):
            key = "usr_prompt"
            continue
        elif text.startswith("---------------------------------------"):
            continue
        else:
            prompt[key].append(text)

    codes = []
    for i in range(1, 100):
        for f in run_files:
            if f.endswith(f'Random-0-SHOT-No-iteration-{i}-RUN.py'):
                codes.append(read_text_file_line_by_line(f))
    df_runtime = pd.merge(df_gen, df_run, left_on="number_iteration", right_on='number_iteration', how='left').reset_index(drop=True)
    df_runtime["total_runtime"] = df_runtime["time_total"] + df_runtime["time_execution"]
    df_runtime = df_runtime.rename(columns={"number_iteration": "index",'time_pipeline_generate': 'Generation', 'time_verify': 'Verify', 'time_execution':'Execution'}).reset_index(drop=True)
    df_runtime = df_runtime.set_index('index')

    res = {
        "df_runtime": df_runtime,
        "df_gen": df_gen,
        "df_run": df_run,
        "df_performance": df_performance,
        "df_cost": df_cost,
        "df_error": df_err_resul,
        "system_prompt": "".join(prompt["system_prompt"]),
        "usr_prompt": "".join(prompt["usr_prompt"]),
        "codes": codes,
        "performance_auc": pd.DataFrame([_get_data(df_performance["train_auc"], "Train",100), _get_data(df_performance["test_auc"], "Test", 100)], index=[0, 1]),
        "performance_f1_score": pd.DataFrame([_get_data(df_performance["train_f1_score"], "Train", 100), _get_data(df_performance["test_f1_score"], "Test", 100)], index=[0, 1]),
        "performance_auc_ovr": pd.DataFrame([_get_data(df_performance["train_auc_ovr"], "Train", 100), _get_data(df_performance["test_auc_ovr"], "Test", 100)], index=[0, 1]),
        "performance_log_loss": pd.DataFrame([_get_data(df_performance["train_log_loss"], "Train"), _get_data(df_performance["test_log_loss"], "Test")], index=[0, 1]),
        "performance_r_squared": pd.DataFrame([_get_data(df_performance["train_r_squared"], "Train", 100), _get_data(df_performance["test_r_squared"], "Test", 100)], index=[0, 1]),
        "performance_rmse": pd.DataFrame([_get_data(df_performance["train_rmse"], "Train"), _get_data(df_performance["test_rmse"], "Test")], index=[0, 1]),
    }
    return res


def _get_data(series, grp, ratio=1):
    qmin, q1, q2, q3, qmax = series.quantile([0, 0.25, 0.5, 0.75, 1])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    mean = series.mean()

    out = series[(series > upper) | (series < lower)]

    if not out.empty:
        outlier = list(out.values)
    else:
        outlier = []
    data = {
        "grp": grp,
        "q1": q1 * ratio,
        "q2": q2 * ratio,
        "q3": q3 * ratio,
        "lw": max(lower * ratio, 0),
        "uw": min(upper* ratio, ratio),
        "otlrs": [outlier],
    }
    return data


# def _calc_box(srs: dd.Series, qntls: da.Array) -> Dict[str, Any]:
#     # quartiles
#     data = {f"qrtl{i + 1}": qntls.loc[qnt].sum() for i, qnt in enumerate((0.25, 0.5, 0.75))}
#
#     # inter-quartile range
#     iqr = data["qrtl3"] - data["qrtl1"]
#     srs_iqr = srs[srs.between(data["qrtl1"] - 1.5 * iqr, data["qrtl3"] + 1.5 * iqr)]
#     # lower and upper whiskers
#     data["lw"], data["uw"] = srs_iqr.min(), srs_iqr.max()
#
#     # outliers
#     otlrs = srs[~srs.between(data["qrtl1"] - 1.5 * iqr, data["qrtl3"] + 1.5 * iqr)]
#     # randomly sample at most 100 outliers from each partition without replacement
#     smp_otlrs = otlrs.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=otlrs)
#     data["otlrs"] = smp_otlrs.values
#
#     return data