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


PROMPT_ADDITIONAL_TEXT = "This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"Class\"." \
                         "Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns." \
                         "The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes." \
                         "This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small." \
                         "The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected. " \
                         "Added columns can be used in other codeblocks, dropped columns are not available anymore."
