CATEGORICAL_RATIO: float = 0.01


class REPRESENTATION_TYPE:
    TEXT = "TEXT"
    # --------------------
    NUMBER_SIGN = "NUMBERSIGN"
    BASELINE = "BASELINE"
    INSTRUCTION = "INSTRUCTION"
    NUMBER_SIGN_WFK = "NUMBERSIGNWFK"
    BASELINE_WOFK = "BASELINEWOFK"
    TEXT_WFK = "TEXTWFK"
    INSTRUCTION_WFK = "INSTRUCTIONWFK"
    NUMBER_SIGN_WORULE = "NUMBERSIGNWORULE"
    SQL_WRULE = "SQLWRULE"
    INSTRUCTION_WRULE = "INSTRUCTIONWRULE"
    TEXT_WRULE = "TEXTWRULE"
    SQL_COT = "SQLCOT"
    TEXT_COT = "TEXTCOT"
    NUMBER_SIGN_COT = "NUMBERSIGNCOT"
    INSTRUCTION_COT = "INSTRUCTIONCOT"
    CBR = "CBR"

PROMPT_DESCRIPTION = "Create a comprehensive Python 3.10 pipeline ({}) using the following format. This pipeline generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{}\"." \
                         "Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns." \
                         "The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes." \
                         "This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small." \
                         "The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected. " \
                         "Added columns can be used in other codeblocks, dropped columns are not available anymore."

CODE_FORMATTING_IMPORT = f"""Code formatting for all required packages:
```python-import
# Import all required packages
```end-import
"""

CODE_FORMATTING_REQUIREMENTS = f"""Code formatting for requirements.txt file:
```python-requirements.txt
# list all required packages here
```end-requirements.txt
"""

CODE_FORMATTING_ADDING = "Code formatting for each added column:\n \
```python-added-column \n \
# (Feature name and description) \n \
# Usefulness: (Description why this adds useful real world knowledge to classify '{}' according to dataset description and attributes.) \n \
(Some pandas code using '{}', '{}', ... to add a new column for each row in df)\n \
```end-added-column"

CODE_FORMATTING_DROPPING = f"""Code formatting for dropping columns:
```python-dropping-columns
# Explanation why the column XX is dropped
# df.drop(columns=['XX'], inplace=True)
```end-dropping-columns
"""

CODE_FORMATTING_TECHNIQUE = "Code formatting for training technique:\n \
```python-training-technique \n \
# Use a {} technique\n \
# Explanation why the solution is selected \n \
trn = ... \n \
```end-training-technique"

CODE_FORMATTING_OTHER = f"""Code formatting for somthing else:
```python-other
# Explanation why this line of code is required
```end-other
"""

CODE_FORMATTING_LOAD_DATASET = "Code formatting for loading datasets: \n \
```python-load-dataset \n \
# load train and test datasets ({} file formats) here \n \
```end-load-dataset "

CODE_FORMATTING_EVALUATION = f"""Code formatting for evaluation:
```python-evaluation
# Report evaluation based on only test dataset  
```end-evaluation
"""