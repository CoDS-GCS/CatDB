class REPRESENTATION_TYPE:
    SCHEMA = "SCHEMA"
    DISTINCT = "DISTINCT"
    MISSING_VALUE = "MISSING_VALUE"
    NUMERIC_STATISTIC = "NUMERIC_STATISTIC"
    CATEGORICAL_VALUE = "CATEGORICAL_VALUE"

    DISTINCT_MISSING_VALUE = "DISTINCT_MISSING_VALUE"
    DISTINCT_NUMERIC_STATISTIC = "DISTINCT_NUMERIC_STATISTIC"
    MISSING_VALUE_NUMERIC_STATISTIC = "MISSING_VALUE_NUMERIC_STATISTIC"
    MISSING_VALUE_CATEGORICAL_VALUE = "MISSING_VALUE_CATEGORICAL_VALUE"
    NUMERIC_STATISTIC_CATEGORICAL_VALUE= "NUMERIC_STATISTIC_CATEGORICAL_VALUE"

    FULL = "FULL"


# Rule 1: need 2 parameters. 1) type of info, e.g., statistical info, min-max values,... 2)  total number of steps
Rule_1 = ('You will be given a dataset, a {} of the dataset, and a question. Your task is to generate a data '
          'science pipeline. You should answer only by generating code. You should follow Steps 1 to 11 to answer '
          'the question. You should return a data science pipeline in Python 3.10 programming language. If you do not '
          'have a relevant answer to the question, simply write: "Insufficient information."')

# Rule 2: need 2 parameters. 1) path to train data 2) path to test data
Rule_2 = ('Load the raining and test datasets. For the training data, utilize the variable """train_data={}""", '
          'and for the test data, employ the variable """test_data={}""". Utilize pandas\' CSV readers to load the '
          'datasets.')

# Rule 3: without extra parameter
Rule_3 = "Don't split the train_data into train and test sets. Use only the given datasets."

# Rule 4: need 2 parameters. 2) name of metadata 2) prefix label
Rule_4 = ('The user will provide the {} of the dataset with columns appropriately named as attributes, enclosed in '
          'triple quotes, and preceded by the prefix "{}".')

# Rule_5: 1) target column name 2) specific algorithm 3) prefix label 4) classifier/regressor 5) same as 4
Rule_5 = ('This pipeline generates additional columns that are useful for a downstream {} algorithm predicting "{}".'
          'Additional columns add new semantic information, that is they use real world knowledge on the dataset '
          'mentioned in """{}""". They can e.g. be feature combinations, transformations, aggregations where the new column '
          'is a function of the existing columns. The scale of columns and offset does not matter. Make sure all used '
          'columns exist. Follow the above description of columns closely and consider the datatypes and meanings of '
          'classes. This code also drops columns, if these may be redundant and hurt the predictive performance of '
          'the downstream {} (Feature selection). Dropping columns may help as the chance of overfitting is lower, '
          'especially if the dataset is small. The {} will be trained on the dataset with the generated columns and '
          'evaluated on a holdout set.')

# Rule 6:
Rule_6 = "Remove low ration, static, and unique columns by getting statistic values."

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
