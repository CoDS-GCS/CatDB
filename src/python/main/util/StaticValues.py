# # Data preprocessing Rules:
# dp_rule_task = "Task: Generate list of tasks are required for data preprocessing in Python 3.10."
#
# dp_rule_input = ("Input: A dataset in CSV format, a schema that describes the columns and data types of the dataset, "
#                  "and a data profiling info that summarizes the statistics and quality of the dataset.")
#
# dp_rule_output = "Output: A Python 3.10 code that performs the following steps:"
# dp_rule_1 = "Import the necessary libraries and modules."
# dp_rule_2 = ('Load the training and test datasets. For the training data, utilize the variable """train_data={}""", '
#           'and for the test data, employ the variable """test_data={}""". Utilize pandas\' CSV readers to load the '
#           'datasets.')
# dp_rule_3 = "Explicitly don't split the train_data into train and test sets. Use only the given datasets."
# dp_rule_4 = ('The user will provide the {} of the dataset with columns appropriately named as attributes, enclosed in '
#              'triple quotes, and preceded by the prefix "{}".')
#
# dp_rule_5 = 'If the question is not relevant to the dataset or the task, the output should be: "Insufficient information."'
# dp_rule_6 = "Explicitly analyze feature based on the provided data profiling information and implement data preprocessing tasks: i) handel missing values, ii) select appropriate data transfer for each column specifically (categorical values and numerical values based on min, max, mean, median values), ii) and apply data cleaning based on data profiling information."
# dp_rule_6_1 = "Explicitly implement Outlier removal for Train and Test data based on the provided data profiling information."
# dp_rule_6_2 = 'Explicitly do data augmentation techniques based on the data profiling information (use your knowledge to chose best technique).'
# dp_rule_7 = 'The target feature in the dataset is \"{}\".'
# dp_rule_8 = "Don't display the first few rows of the datasets."
#
# # Feature Engineering Rules:
# fe_rule_task = ('Task: Select the appropriate features and target variables for the question (Feature Engineering Task). '
#           'Additional columns add new semantic information, additional columns that are useful for a downstream algorithm'
#           'predicting "{}". They can e.g. be feature combinations, transformations, aggregations where the '
#           'new column is a function of the existing columns. Use appropriate scale factor for columns are need'
#           'to transfer.')
#
# fe_rule_input = ('Input: first draft version of pipline with a Data Preprocessing task enclosed in '
#                  '"<CODE> pipline code will be here. </CODE>", and a schema that describes the columns and data types '
#                  'of the dataset, and a data profiling info that summarizes the statistics and quality of the dataset.')
#
# fe_rule_output = "Output: A modified Python 3.10 code with additional feature enginnering tasks that performs the following steps:"
# fe_rule_1 = "Explicitly analyze feature based on the provided data profiling information and implement feature engineering task."
# fe_rule_1_1 = 'The target feature in the dataset is \"{}\".'
# fe_rule_2 = ("Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream {} "
#           "(Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the "
#           "dataset is small. The {} will be trained on the dataset with the generated columns and evaluated on a "
#           "holdout set.")
# fe_rule_3 = 'If the question is not relevant to the dataset or the task, the output should be: "Insufficient information."'
# fe_rule_4 = "Don't display the first few rows of the datasets."
#
# # Model Selection and Hyperparameters Tuning
# ms_rule_task = "Task: Select an appropriate {} Machine Learning model for the question."
# ms_rule_input = ('Input: first draft version of pipline with a Data Preprocessing and Feature Engineering task enclosed in '
#                  '"<CODE> pipline code will be here. </CODE>", and a schema that describes the columns and data types '
#                  'of the dataset, and a data profiling info that summarizes the statistics and quality of the dataset.')
# ms_rule_output = "Output: A modified Python 3.10 code with a Machine Learning algorithm task that performs the following steps:"
# ms_rule_1 =  'Explicitly select a best {} model based on the data profiling information for  predicting "{}".'
# ms_rule_2 = ('Select a suitable hyperparameters for the selected algorithm. If the algorithm is RandomForestClassifier '
#              'then pass max_leaf_nodes={} as parameter.')
# ms_rule_3 = "Don't report validation evaluation. We don't need it."
#
#
# rule_code_block = "Each codeblock ends with \"```end\" and starts with \"```python\"."
#
# chain_rule_1 = "Don't use \"if __name__ == '__main__':\" style, use only flat mode."
# #----------------------------------------------------------------------------------------------------------------------
#
# Rule_task = ("Task: Generate a data science pipeline in Python 3.10 that answers a question based on a "
#              "given dataset, {}.")
#
# Rule_input = ("Input: A dataset in CSV format, a schema that describes the columns and data types of the dataset, "
#               "and a data profiling info that summarizes the statistics and quality of the dataset. A question that "
#               "requires data analysis or modeling to answer.")
#
# Rule_output = "Output: A Python 3.10 code that performs the following steps:"
# Rule_1 = "Import the necessary libraries and modules."
# Rule_2 = ('Load the training and test datasets. For the training data, utilize the variable """train_data={}""", '
#           'and for the test data, employ the variable """test_data={}""". Utilize pandas\' CSV readers to load the '
#           'datasets.')
# Rule_3 = "Don't split the train_data into train and test sets. Use only the given datasets."
# Rule_4 = ('The user will provide the {} of the dataset with columns appropriately named as attributes, enclosed in '
#           'triple quotes, and preceded by the prefix "{}".')
# Rule_5 = "Explicitly analyze feature based on the provided data profiling information and implement data preprocessing tasks: i) handel missing values, ii) select appropriate data transfer for each column specifically (categorical values and numerical values based on min, max, mean, median values), ii) and apply data cleaning based on data profiling information."
# Rule_5_1 = "Explicitly implement Outlier removal for Train and Test data based on the provided data profiling information."
# Rule_5_2 = 'Explicitly do data augmentation techniques based on the data profiling information (use your knowledge to chose best technique).'
# Rule_5_3 = "Explicitly analyze feature based on the provided data profiling information and implement feature engineering task."
# Rule_6 = "Explicitly select a best model based on the data profiling information."
# Rule_7 = ('Select the appropriate features and target variables for the question. '
#           'Additional columns add new semantic information, additional columns that are useful for a downstream algorithm'
#           'predicting "{}". They can e.g. be feature combinations, transformations, aggregations where the '
#           'new column is a function of the existing columns. Use appropriate scale factor for columns are need'
#           'to transfer.')
#
# Rule_8 = (" Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream {} "
#           "(Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the "
#           "dataset is small. The {} will be trained on the dataset with the generated columns and evaluated on a "
#           "holdout set.")
#
# Rule_9 = ("In order to avoid runtime error for unseen value on the target feature, do preprocessing based on union of "
#           "train and test dataset.")
# Rule_10 = 'If the question is not relevant to the dataset or the task, the output should be: "Insufficient information."'
# Rule_11 = "Don't report validation evaluation. We don't need it."
#
# # Rule_13 = 'If the algorithm is RandomForestClassifier then pass max_leaf_nodes={} as parameter.'
# dataset_description = "Description of the dataset:\n{}\n"
#
# CODE_FORMATTING_IMPORT = f"""Code formatting for all required packages and pipeline format:
# ```python
# # Import all required packages
# # Don't use "if __name__ == '__main__':" style, use only flat mode.
# ```end
# """
#
# CODE_FORMATTING_PREPROCESSING = "Code formatting for data preprocessing:\n \
# ```python \n \
# # List all required tasks  \
# ```end"
#
#
# CODE_FORMATTING_ADDING = "Code formatting for each added column:\n \
# ```python \n \
# # (Feature name and description) \n \
# # Usefulness: (Description why this adds useful real world knowledge to classify '{}' according to dataset description and attributes.) \n \
# (Some pandas code using '{}', '{}', ... to add a new column for each row in df)\n \
# ```end"
#
# CODE_FORMATTING_DROPPING = f"""Code formatting for dropping columns:
# ```python-dropping-columns
# # Explanation why the column XX is dropped
# # df.drop(columns=['XX'], inplace=True)
# ```end-dropping-columns
# """
#
# CODE_FORMATTING_TECHNIQUE = ("Code formatting for training technique:\n \
# ```python \n \
# # Choose the suitable machine learning algorithm or technique ({}).\n \
# # Explanation why the solution is selected \n \
# trn = ... \n \
# ```end")
#
# CODE_FORMATTING_BINARY_EVALUATION = f"""Code formatting for binary classification evaluation:
# ```python
# # Report evaluation based on train and test dataset
# # Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Train_Accuracy=..." and "Test_Accuracy=...".
# # Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "Train_F1_score=..." and "Test_F1_score=...".
# # Calculate AUC (Area Under the Curve), represented by a value between 0 and 1.
# # Print the train AUC result: print(f"Train_AUC:{{Train_AUC}}")
# # Print the train accuracy result: print(f"Train_Accuracy:{{Train_Accuracy}}")
# # Print the train f1 score result: print(f"Train_F1_score:{{Train_F1_score}}")
# # Print the test AUC result: print(f"Test_AUC:{{Test_AUC}}")
# # Print the test accuracy result: print(f"Test_Accuracy:{{Test_Accuracy}}")
# # Print the test f1 score result: print(f"Test_F1_score:{{Test_F1_score}}")
# ```end
# """
#
# CODE_FORMATTING_MULTICLASS_EVALUATION = f"""Code formatting for multiclass classification evaluation:
# ```python
# # Report evaluation based on train and test dataset
# # Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Train_Accuracy=..." and "Test_Accuracy=...".
# # Calculate the model log loss, a lower log-loss value means better predictions. Store the  log loss value in a variable labeled as "Train_Log_loss=..." and "Test_Log_loss=...".
# # Calculate AUC_OVO (Area Under the Curve One-vs-One), represented by a value between 0 and 1.
# # Calculate AUC_OVR (Area Under the Curve One-vs-Rest), represented by a value between 0 and 1.
# # Print the train AUC One-vs-One result: print(f"Train_AUC_OVO:{{Train_AUC_OVO}}")
# # Print the train AUC One-vs-Rest result: print(f"Train_AUC_OVR:{{Train_AUC_OVR}}")
# # Print the train accuracy result: print(f"Train_Accuracy:{{Train_Accuracy}}")
# # Print the train log loss result: print(f"Train_Log_loss:{{Train_Log_loss}}")
# # Print the test AUC One-vs-One result: print(f"Test_AUC_OVO:{{Test_AUC_OVO}}")
# # Print the test AUC One-vs-Rest result: print(f"Test_AUC_OVR:{{Test_AUC_OVR}}")
# # Print the test accuracy result: print(f"Test_Accuracy:{{Test_Accuracy}}")
# # Print the test log loss result: print(f"Test_Log_loss:{{Test_Log_loss}}")
# ```end
# """
#
# CODE_FORMATTING_REGRESSION_EVALUATION = f"""Code formatting for regression evaluation:
# ```python
# # Report evaluation based on train and test dataset
# # Calculate the model R-Squared, represented by a value between 0 and 1, where 0 indicates low and 1 ndicates more variability is explained by the model. Store the R-Squared value in a variable labeled as "Train_R_Squared=..." and "Test_R_Squared=...".
# # Calculate the model Root Mean Squared Error, where the lower the value of the Root Mean Squared Error, the better the model is.. Store the model Root Mean Squared Error value in a variable labeled as "Train_RMSE=..." and "Test_RMSE=...".
# # Print the train accuracy result: print(f"Train_R_Squared:{{Train_R_Squared}}")
# # Print the train log loss result: print(f"Train_RMSE:{{Train_RMSE}}")
# # Print the test accuracy result: print(f"Test_R_Squared:{{Test_R_Squared}}")
# # Print the test log loss result: print(f"Test_RMSE:{{Test_RMSE}}")
# ```end
# """
