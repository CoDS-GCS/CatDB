import pandas as pd


class LogResults(object):
    def __init__(self,
                 dataset_name: str,
                 config: str,
                 sub_task: str,
                 llm_model: str,
                 classifier: str,
                 task_type: str,
                 status: str,
                 number_iteration: int,
                 number_iteration_error: int,
                 has_description: str,
                 time_catalog_load: float,
                 time_pipeline_generate: float,
                 time_total: float,
                 time_execution: float,
                 train_auc: float = -2,
                 train_auc_ovo: float = -2,
                 train_auc_ovr: float = -2,
                 train_accuracy: float = -2,
                 train_f1_score: float = -2,
                 train_log_loss: float = -2,
                 train_r_squared: float = -2,
                 train_rmse: float = -2,
                 test_auc: float = -2,
                 test_auc_ovo: float = -2,
                 test_auc_ovr: float = -2,
                 test_accuracy: float = -2,
                 test_f1_score: float = -2,
                 test_log_loss: float = -2,
                 test_r_squared: float = -2,
                 test_rmse: float = -2,
                 prompt_token_count: int = 0,
                 all_token_count: int = 0,
                 operation: str = None,
                 number_of_samples: int = 0
                 ):
        self.config = config
        self.sub_task = sub_task
        self.llm_model = llm_model
        self.classifier = classifier
        self.task_type = task_type
        self.status = status
        self.number_iteration = number_iteration
        self.number_iteration_error = number_iteration_error
        self.has_description = has_description
        self.dataset_name = dataset_name
        self.time_catalog_load = time_catalog_load
        self.time_pipeline_generate = time_pipeline_generate
        self.time_total = time_total
        self.time_execution = time_execution
        self.train_auc = train_auc
        self.train_auc_ovo = train_auc_ovo
        self.train_auc_ovr = train_auc_ovr
        self.train_accuracy = train_accuracy
        self.train_f1_score = train_f1_score
        self.train_log_loss = train_log_loss
        self.train_r_squared = train_r_squared
        self.train_rmse = train_rmse
        self.test_auc = test_auc
        self.test_auc_ovo = test_auc_ovo
        self.test_auc_ovr = test_auc_ovr
        self.test_accuracy = test_accuracy
        self.test_f1_score = test_f1_score
        self.test_log_loss = test_log_loss
        self.test_r_squared = test_r_squared
        self.test_rmse = test_rmse
        self.prompt_token_count = prompt_token_count
        self.all_token_count = all_token_count
        self.operation = operation
        self.number_of_samples = number_of_samples

        self.columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                        "number_iteration", "number_iteration_error", "has_description", "time_catalog_load",
                        "time_pipeline_generate",
                        "time_total", "time_execution", "train_auc", "train_auc_ovo", "train_auc_ovr", "train_accuracy",
                        "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc", "test_auc_ovo",
                        "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared",
                        "test_rmse",
                        "prompt_token_count", "all_token_count", "operation", "number_of_samples"]

    def save_results(self, result_output_path: str):
        try:
            df_result = pd.read_csv(result_output_path)

        except Exception as err:
            df_result = pd.DataFrame(columns=self.columns)

        df_result.loc[len(df_result)] = [self.dataset_name,
                                         self.config,
                                         self.sub_task,
                                         self.llm_model,
                                         self.classifier,
                                         self.task_type,
                                         self.status,
                                         self.number_iteration,
                                         self.number_iteration_error,
                                         self.has_description,
                                         self.time_catalog_load,
                                         self.time_pipeline_generate,
                                         self.time_total,
                                         self.time_execution,
                                         self.train_auc,
                                         self.train_auc_ovo,
                                         self.train_auc_ovr,
                                         self.train_accuracy,
                                         self.train_f1_score,
                                         self.train_log_loss,
                                         self.train_r_squared,
                                         self.train_rmse,
                                         self.test_auc,
                                         self.test_auc_ovo,
                                         self.test_auc_ovr,
                                         self.test_accuracy,
                                         self.test_f1_score,
                                         self.test_log_loss,
                                         self.test_r_squared,
                                         self.test_rmse,
                                         self.prompt_token_count,
                                         self.all_token_count,
                                         self.operation,
                                         self.number_of_samples]

        df_result.to_csv(result_output_path, index=False)


def save_log(args, sub_task, final_status, iteration, iteration_error, time_catalog, time_generate, time_total,
             time_execute, prompt_token_count, all_token_count, operation_tag, run_mode, results_verified, results):
    from util.Config import __execute_mode
    log_results = LogResults(dataset_name=args.dataset_name, config=args.prompt_representation_type, sub_task=sub_task,
                             llm_model=args.llm_model, classifier="Auto", task_type=args.task_type,
                             status=f"{final_status}", number_iteration=iteration,
                             number_iteration_error=iteration_error,
                             has_description=args.dataset_description,
                             time_catalog_load=time_catalog, time_pipeline_generate=time_generate,
                             time_total=time_total,
                             time_execution=time_execute,
                             prompt_token_count=prompt_token_count,
                             all_token_count=all_token_count + prompt_token_count,
                             operation=operation_tag,
                             number_of_samples=args.prompt_number_samples)

    if run_mode == __execute_mode and results_verified:
        log_results.train_auc = results["Train_AUC"]
        log_results.train_auc_ovo = results["Train_AUC_OVO"]
        log_results.train_auc_ovr = results["Train_AUC_OVR"]
        log_results.train_accuracy = results["Train_Accuracy"]
        log_results.train_f1_score = results["Train_F1_score"]
        log_results.train_log_loss = results["Train_Log_loss"]
        log_results.train_r_squared = results["Train_R_Squared"]
        log_results.train_rmse = results["Train_RMSE"]
        log_results.test_auc = results["Test_AUC"]
        log_results.test_auc_ovo = results["Test_AUC_OVO"]
        log_results.test_auc_ovr = results["Test_AUC_OVR"]
        log_results.test_accuracy = results["Test_Accuracy"]
        log_results.test_f1_score = results["Test_F1_score"]
        log_results.test_log_loss = results["Test_Log_loss"]
        log_results.test_r_squared = results["Test_R_Squared"]
        log_results.test_rmse = results["Test_RMSE"]

    if final_status:
        log_results.save_results(result_output_path=args.result_output_path)


class LogCleaningResults(object):
    def __init__(self,
                 dataset_name: str,
                 sub_dataset_name: str,
                 llm_model: str,
                 status: str,
                 number_iteration: int,
                 number_iteration_error: int,
                 time_catalog_load: float,
                 time_pipeline_generate: float,
                 time_total: float,
                 time_execution: float,
                 prompt_token_count: int = 0,
                 all_token_count: int = 0,
                 operation: str = None,
                 total_refined_cols: int = 0,
                 refine_cols: str = None,
                 total_diffs: int = 0
                 ):
        self.llm_model = llm_model
        self.status = status
        self.number_iteration = number_iteration
        self.number_iteration_error = number_iteration_error
        self.dataset_name = dataset_name
        self.sub_dataset_name = sub_dataset_name
        self.time_catalog_load = time_catalog_load
        self.time_pipeline_generate = time_pipeline_generate
        self.time_total = time_total
        self.time_execution = time_execution
        self.prompt_token_count = prompt_token_count
        self.all_token_count = all_token_count
        self.operation = operation
        self.total_refined_cols = total_refined_cols
        self.refine_cols: str = refine_cols
        self.total_diffs = total_diffs

        self.columns = ["dataset_name", "sub_dataset_name", "llm_model", "status", "number_iteration", "number_iteration_error",
                        "time_catalog_load", "time_pipeline_generate", "time_total", "time_execution",
                        "prompt_token_count", "all_token_count", "operation", "total_refined_cols", "refine_cols",
                        "total_diffs"]

    def save_results(self, result_output_path: str):
        try:
            df_result = pd.read_csv(result_output_path)

        except Exception as err:
            df_result = pd.DataFrame(columns=self.columns)

        df_result.loc[len(df_result)] = [self.dataset_name,
                                         self.sub_dataset_name,
                                         self.llm_model,
                                         self.status,
                                         self.number_iteration,
                                         self.number_iteration_error,
                                         self.time_catalog_load,
                                         self.time_pipeline_generate,
                                         self.time_total,
                                         self.time_execution,
                                         self.prompt_token_count,
                                         self.all_token_count,
                                         self.operation,
                                         self.total_refined_cols,
                                         self.refine_cols,
                                         self.total_diffs]

        df_result.to_csv(result_output_path, index=False)


def save_cleaning_log(args, final_status, iteration, iteration_error, time_catalog, time_generate, time_total,
                      time_execute, prompt_token_count, all_token_count, operation_tag, total_refined_cols,
                      refine_cols, sub_dataset_name, total_diffs):
    log_results = LogCleaningResults(dataset_name=args.dataset_name,
                                     llm_model=args.llm_model,
                                     status=f"{final_status}",
                                     number_iteration=iteration,
                                     number_iteration_error=iteration_error,
                                     time_catalog_load=time_catalog,
                                     time_pipeline_generate=time_generate,
                                     time_total=time_total,
                                     time_execution=time_execute,
                                     prompt_token_count=prompt_token_count,
                                     all_token_count=all_token_count + prompt_token_count,
                                     operation=operation_tag,
                                     total_refined_cols=total_refined_cols,
                                     refine_cols=refine_cols,
                                     sub_dataset_name=sub_dataset_name,
                                     total_diffs=total_diffs)

    if final_status:
        log_results.save_results(result_output_path=args.result_output_path)
