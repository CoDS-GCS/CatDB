Rule_task = ("Task: Generate a data science pipeline in Python 3.11 that answers a question based on a "
             "given dataset, {}.")

Rule_input = ("Input: A dataset in CSV format, a schema that describes the columns and data types of the dataset, "
              "and a data profiling info that summarizes the statistics and quality of the dataset. A question that "
              "requires data analysis or modeling to answer.")

Rule_output = "Output: A Python 3.10 code that performs the following steps:"
Rule_1 = "Import the necessary libraries and modules."
Rule_2 = ('Load the raining and test datasets. For the training data, utilize the variable """train_data={}""", '
          'and for the test data, employ the variable """test_data={}""". Utilize pandas\' CSV readers to load the '
          'datasets.')
Rule_3 = "Don't split the train_data into train and test sets. Use only the given datasets."
Rule_4 = ('The user will provide the {} of the dataset with columns appropriately named as attributes, enclosed in '
          'triple quotes, and preceded by the prefix "{}".')
Rule_5 = "Perform data cleaning and preprocessing."
Rule_6 = "Perform feature processing (e.g., encode categorical values by dummyEncode)."
Rule_7 = ('Select the appropriate features and target variables for the question. '
          'Additional columns add new semantic information, additional columns that are useful for a downstream algorithm'
          'predicting "{}". They can e.g. be feature combinations, transformations, aggregations where the '
          'new column is a function of the existing columns. Use appropriate scale factor for columns are need'
          'to transfer.')

Rule_8 = (" Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream {} "
          "(Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the "
          "dataset is small. The {} will be trained on the dataset with the generated columns and evaluated on a "
          "holdout set.")

Rule_9 = ("In order to avoid runtime error for unseen value on the target feature, do preprocessing based on union of "
          "train and test dataset.")
Rule_10 = 'If the question is not relevant to the dataset or the task, the output should be: "Insufficient information."'
Rule_11 = "Don't report validation evaluation. We don't need it."

CODE_FORMATTING_IMPORT = f"""Code formatting for all required packages:
```python
# Import all required packages
```end
"""

CODE_FORMATTING_ADDING = "Code formatting for each added column:\n \
```python \n \
# (Feature name and description) \n \
# Usefulness: (Description why this adds useful real world knowledge to classify '{}' according to dataset description and attributes.) \n \
(Some pandas code using '{}', '{}', ... to add a new column for each row in df)\n \
```end"

CODE_FORMATTING_DROPPING = f"""Code formatting for dropping columns:
```python-dropping-columns
# Explanation why the column XX is dropped
# df.drop(columns=['XX'], inplace=True)
```end-dropping-columns
"""

CODE_FORMATTING_TECHNIQUE = "Code formatting for training technique:\n \
```python \n \
# Choose the suitable machine learning algorithm or technique ({}). Don't used Random forest Classification algorithm.\n \
# Explanation why the solution is selected \n \
trn = ... \n \
```end"

CODE_FORMATTING_BINARY_EVALUATION = f"""Code formatting for binary classification evaluation:
```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{{Accuracy}}")   
# Print the f1 score result: print(f"F1_score:{{F1_score}}") 
```end
"""

CODE_FORMATTING_MULTICLASS_EVALUATION = f"""Code formatting for multiclass classification evaluation:
```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model log loss, a lower log-loss value means better predictions. Store the  log loss value in a variable labeled as "Log_loss=...".
# Print the accuracy result: print(f"Accuracy:{{Accuracy}}")   
# Print the log loss result: print(f"Log_loss:{{Log_loss}}") 
```end
"""

CODE_FORMATTING_REGRESSION_EVALUATION = f"""Code formatting for regression evaluation:
```python
# Report evaluation based on only test dataset
# Calculate the model R-Squared, represented by a value between 0 and 1, where 0 indicates low and 1 ndicates more variability is explained by the model. Store the R-Squared value in a variable labeled as "R_Squared=...".
# Calculate the model Root Mean Squared Error, where the lower the value of the Root Mean Squared Error, the better the model is.. Store the model Root Mean Squared Error value in a variable labeled as "RMSE=...".
# Print the accuracy result: print(f"R_Squared:{{R_Squared}}")   
# Print the log loss result: print(f"RMSE:{{RMSE}}") 
```end
"""
