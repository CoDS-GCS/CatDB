# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

categorical_features = ["DayOfWeek", "Airline"]
numerical_features = ["AirportTo", "AirportFrom", "Flight", "Length", "Time"]

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

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
X_train = pipeline.fit_transform(train_data.drop("Delay", axis=1))
y_train = train_data["Delay"]
X_test = pipeline.transform(test_data.drop("Delay", axis=1))
y_test = test_data["Delay"]
# ```end