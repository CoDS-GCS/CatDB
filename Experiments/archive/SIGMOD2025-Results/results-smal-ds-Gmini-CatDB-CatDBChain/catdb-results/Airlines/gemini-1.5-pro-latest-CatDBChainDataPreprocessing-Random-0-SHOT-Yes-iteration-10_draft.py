# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

categorical_features = ["DayOfWeek", "Airline"]
numerical_features = ["AirportTo", "AirportFrom", "Flight", "Length", "Time"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ]
)
pipeline.fit(train_data)

train_transformed = pipeline.transform(train_data)
test_transformed = pipeline.transform(test_data)
# ```end