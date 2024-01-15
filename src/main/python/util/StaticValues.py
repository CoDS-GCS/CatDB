CATEGORICAL_RATIO: float = 0.01


class REPRESENTATION_TYPE:
    SCHEMA = "SCHEMA"
    SCHEMA_STATISTIC = "SCHEMA_STATISTIC"
    SCHEMA_DTYPE_FD = "SCHEMA_FD" # TODO:
    SCHEMA_ID = "SCHEMA_ID" # TODO:
    SCHEMA_FD_ID = "SCHEMA_FD_ID" # TODO:
    SCHEMA_ABSTRACT = "SCHEMA_ABSTRACT" #TODO:


PROMPT_DESCRIPTION = "Create a comprehensive Python 3.10 pipeline ({}) using the following format. This pipeline generates additional columns that are useful for a downstream {} algorithm predicting \"{}\"." \
                         "Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns." \
                         "The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes." \
                         "This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small." \
                         "The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected. " \
                         "Added columns can be used in other codeblocks, dropped columns are not available anymore."

CODE_FORMATTING_IMPORT = f"""Code formatting for all required packages:
```python
# Import all required packages
```end
"""

CODE_FORMATTING_REQUIREMENTS = f"""Code formatting for requirements.txt file:
```python-requirements.txt
# list all required packages here
```end-requirements.txt
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
# Use a {} technique\n \
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

CODE_FORMATTING_DATASET = 'Load datasets from the following path:\n \
"""train_data = {}""" \n \
"""test_data = {}"""'