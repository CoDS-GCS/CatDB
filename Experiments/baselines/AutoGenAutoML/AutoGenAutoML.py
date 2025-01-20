from autogen.coding import LocalCommandLineCodeExecutor
import tiktoken
from automl.AutoML import AutoML as CatDBAutoML
from util.Config import Config
from util.Data import Dataset
from os.path import dirname
import time

import autogen
from autogen import ConversableAgent
from autogen.code_utils import content_str
from util.LogResults import LogResults


class AutoGenAutoML(CatDBAutoML):
    def __init__(self, dataset: Dataset, config: Config, llm_model: str, iteration: int, *args, **kwargs):
        CatDBAutoML.__init__(self, dataset=dataset, config=config)
        self.llm_model = llm_model
        self.iteration = iteration
        self.config_path = dirname(__file__)
        self.config_list = autogen.config_list_from_json(
            f"{self.config_path}/OAI_CONFIG_LIST",
            filter_dict={
                "model": [self.llm_model],
            },
        )
        self.token_limit = 256
        if llm_model in {"gpt-4o"}:
            self.token_limit = 4096
        elif llm_model in {"llama-3.1-70b-versatile"}:
            self.token_limit = 131072
        elif llm_model in {"gemini-1.5-pro-latest"}:
            self.token_limit = 131072

        code_format = ""
        if self.dataset.task_type == "binary":
            code_format = """
                    Code formatting for binary classification evaluation:
                    # Report evaluation based on train and test dataset
                    # Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Train_Accuracy=..." and "Test_Accuracy=...".
                    # Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "Train_F1_score=..." and "Test_F1_score=...".
                    # Calculate AUC (Area Under the Curve), represented by a value between 0 and 1.
                    print(f"Train_AUC:{{Train_AUC}}")
                    print(f"Train_Accuracy:{{Train_Accuracy}}")   
                    print(f"Train_F1_score:{{Train_F1_score}}")
                    print(f"Test_AUC:{{Test_AUC}}")
                    print(f"Test_Accuracy:{{Test_Accuracy}}")   
                    print(f"Test_F1_score:{{Test_F1_score}}")
                    """
        elif self.dataset.task_type == "multiclass":
            code_format = """
                    Code formatting for multiclass classification evaluation:
                    # Report evaluation based on train and test dataset
                    # Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Train_Accuracy=..." and "Test_Accuracy=...".
                    # Calculate the model log loss, a lower log-loss value means better predictions. Store the  log loss value in a variable labeled as "Train_Log_loss=..." and "Test_Log_loss=...".
                    # Calculate AUC_OVO (Area Under the Curve One-vs-One), represented by a value between 0 and 1.
                    # Calculate AUC_OVR (Area Under the Curve One-vs-Rest), represented by a value between 0 and 1.
                    # print(f"Train_AUC_OVO:{{Train_AUC_OVO}}")
                    # print(f"Train_AUC_OVR:{{Train_AUC_OVR}}")
                    # print(f"Train_Accuracy:{{Train_Accuracy}}")   
                    # print(f"Train_Log_loss:{{Train_Log_loss}}") 
                    # print(f"Test_AUC_OVO:{{Test_AUC_OVO}}")
                    # print(f"Test_AUC_OVR:{{Test_AUC_OVR}}")
                    # print(f"Test_Accuracy:{{Test_Accuracy}}")   
                    # print(f"Test_Log_loss:{{Test_Log_loss}}")
                    """
        elif self.dataset.task_type == "regression":
            code_format = """
                        Code formatting for regression evaluation:
                        # Report evaluation based on train and test dataset
                        # Calculate the model R-Squared, represented by a value between 0 and 1, where 0 indicates low and 1 ndicates more variability is explained by the model. Store the R-Squared value in a variable labeled as "Train_R_Squared=..." and "Test_R_Squared=...".
                        # Calculate the model Root Mean Squared Error, where the lower the value of the Root Mean Squared Error, the better the model is.. Store the model Root Mean Squared Error value in a variable labeled as "Train_RMSE=..." and "Test_RMSE=...".
                        # print(f"Train_R_Squared:{{Train_R_Squared}}")   
                        # print(f"Train_RMSE:{{Train_RMSE}}") 
                        # print(f"Test_R_Squared:{{Test_R_Squared}}")   
                        # print(f"Test_RMSE:{{Test_RMSE}}")
                    """

        self.message = f"""
                           Your goal is to predict the target column `{self.dataset.target_attribute}`.
                           Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. 
                           ## Do not split the train_data into train and test sets. Use only the given datasets.
                           ## In order to avoid runtime error for unseen value on the target feature, do preprocessing based on union of train and test dataset.
                           ## Don't report model validation part (Only and Only report Train and Test model evaluation).
                           ## Don't change task type, for example don't change task tpe form multiclass classification to a regression task.

                           ##{code_format} 

                           ## Data dir
                           training (with labels): {self.dataset.train_path}
                           testing (with labels): {self.dataset.test_path}
                           """

    def save_log(self, status, results, iteration, number_iteration_error, total_time, prompt_tokens, total_tokens):
        pipeline_evl = {"Train_AUC": -2,
                        "Train_AUC_OVO": -2,
                        "Train_AUC_OVR": -2,
                        "Train_Accuracy": -2,
                        "Train_F1_score": -2,
                        "Train_Log_loss": -2,
                        "Train_R_Squared": -2,
                        "Train_RMSE": -2,
                        "Test_AUC": -2,
                        "Test_AUC_OVO": -2,
                        "Test_AUC_OVR": -2,
                        "Test_Accuracy": -2,
                        "Test_F1_score": -2,
                        "Test_Log_loss": -2,
                        "Test_R_Squared": -2,
                        "Test_RMSE": -2}
        if results is not None:
            raw_results = results.splitlines()
            for rr in raw_results:
                row = rr.strip().split(":")
                if row[0] in pipeline_evl.keys():
                    pipeline_evl[row[0]] = row[1].strip()



        log_results = LogResults(dataset_name=self.dataset.dataset_name,
                                 config="AutoGen", sub_task="", llm_model=self.llm_model, classifier="Auto", task_type= self.dataset.task_type,
                                 status=f"{status}", number_iteration=iteration, number_iteration_error=number_iteration_error, has_description="No", time_catalog_load=0,
                                 time_pipeline_generate=total_time, time_total=total_time, time_execution=total_time, train_auc=pipeline_evl["Train_AUC"],
                                 train_auc_ovo=pipeline_evl["Train_AUC_OVO"] , train_auc_ovr= pipeline_evl["Train_AUC_OVR"], train_accuracy=pipeline_evl["Train_Accuracy"],
                                 train_f1_score=pipeline_evl["Train_F1_score"], train_log_loss=pipeline_evl["Train_Log_loss"], train_r_squared=pipeline_evl["Train_R_Squared"], train_rmse=pipeline_evl["Train_RMSE"],
                                 test_auc=pipeline_evl["Test_AUC"], test_auc_ovo=pipeline_evl["Test_AUC_OVO"],  test_auc_ovr=pipeline_evl["Test_AUC_OVR"], test_accuracy=pipeline_evl["Test_Accuracy"],
                                 test_f1_score=pipeline_evl["Test_F1_score"], test_log_loss=pipeline_evl["Test_Log_loss"], test_r_squared=pipeline_evl["Test_R_Squared"], test_rmse=pipeline_evl["Test_RMSE"],
                                 prompt_token_count=prompt_tokens,all_token_count=total_tokens, operation="Run-Pipeline", number_of_samples=0)

        log_results.save_results(result_output_path=self.config.output_path)

    def get_number_tokens(self, message):
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(self.llm_model)
        token_integers = enc.encode(message)
        num_tokens = len(token_integers)
        return num_tokens

    def call_agent(self, message):

        start_time = time.time()
        code_writer_agent = ConversableAgent(
            name="code_writer_agent",
            system_message=message,
            max_consecutive_auto_reply=1,
            code_execution_config=False,
            human_input_mode="NEVER",
            llm_config={"config_list": self.config_list, "seed": 41, "temperature": 0, "cache_seed": None},
            is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0,
        )

        # Create a local command line code executor.
        executor = LocalCommandLineCodeExecutor(
            timeout=10 * 3600,
            work_dir=f"{self.config.output_dir}"
        )

        code_executor_agent = ConversableAgent(
            name="code_executor_agent",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the local command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
        )

        result = code_executor_agent.initiate_chat(
            code_writer_agent,
            message="Execute generated ML pipeline and print results.",
        )

        elapsed_time = time.time() - start_time
        prompt_tokens = completion_tokens = 0
        cost = 0
        try:
            if "gpt" in self.llm_model:
                prompt_tokens = self.get_number_tokens(message)
            else:
                prompt_tokens = result.cost['usage_including_cached_inference'][self.llm_model]['prompt_tokens']
                completion_tokens = result.cost['usage_including_cached_inference'][self.llm_model]['completion_tokens']
                cost = result.cost['usage_including_cached_inference'][self.llm_model]['cost']
            summary = result.summary
            history = result.chat_history
        except Exception as e:
            print(e)
            history = "I cannot solve this task."
            summary = "I cannot solve this task."

        # Find SRC:
        src = None
        error = None
        performance = None
        print(history)
        for h in history:
            try:
                for k in h.keys():
                    if "```python" in h[k]:
                        src = h[k]
                        if "gpt" in self.llm_model:
                            completion_tokens = self.get_number_tokens(src)

                        break
                    elif "exitcode: 1 (execution failed)" in h[k]:
                        error = h[k]
                        break
                    elif "exitcode: 0 (execution succeeded)" in h[k]:
                        performance = h[k]
            except Exception as err:
                pass
        print(f"Tokens: {prompt_tokens} + {completion_tokens}")
        return prompt_tokens + completion_tokens, cost, result, src, error, performance, summary, history, elapsed_time

    def run(self):
        print(f"\n**** AutoGen ****\n")
        total_tokens = 0
        prompt_tokens = 0
        error_tokens = 0
        total_cost = 0

        message = self.message
        status = False
        number_iteration_error = 0
        performance = ""
        elapsed_time = 0
        for i in range(1, 16):
            tokens, cost, reult, src, error, performance, summary, history, e_time = self.call_agent(
                message=message)
            elapsed_time += e_time
            if i == 1:
                prompt_tokens = tokens
            else:
                error_tokens += tokens
            total_tokens += tokens
            total_cost += cost
            if error is not None:
                message = f"""Fix the following ERROR from the pipeline and return full executable pipeline:
                ML Pipeline:
                {src}
                
                Error:
                {summary}
                """
                number_iteration_error = i
                continue
            else:
                number_iteration_error = i - 1
                status = True
                break

        self.save_log(status=status, results=performance, iteration=self.iteration, number_iteration_error=number_iteration_error, total_time=elapsed_time, prompt_tokens=prompt_tokens, total_tokens=total_tokens)
