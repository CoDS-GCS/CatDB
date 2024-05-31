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
        ("cat", OneHotEncoder(), categorical_features),
    ],
    remainder="passthrough"
)

data_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ]
)
X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]
X_train_transformed = data_pipeline.fit_transform(X_train)

X_test = test_data.drop("Delay", axis=1)
y_test = test_data["Delay"]
X_test_transformed = data_pipeline.transform(X_test)
# ```end