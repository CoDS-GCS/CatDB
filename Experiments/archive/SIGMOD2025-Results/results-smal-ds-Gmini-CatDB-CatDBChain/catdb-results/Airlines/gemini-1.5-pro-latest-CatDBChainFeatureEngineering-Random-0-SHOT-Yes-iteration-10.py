# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
import multiprocessing

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

categorical_features = ["DayOfWeek", "Airline"]
numerical_features = ["AirportTo", "AirportFrom", "Flight", "Length", "Time"]

target = ["Delay"]

numerical_transformer = make_pipeline(StandardScaler())
categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

preprocessor = make_column_transformer(
    (numerical_transformer, numerical_features),
    (categorical_transformer, categorical_features),
)

pipeline = make_pipeline(
    preprocessor,
)

X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]
X_test = test_data.drop("Delay", axis=1)
y_test = test_data["Delay"]

with multiprocessing.Pool() as pool:
    pipeline.fit(X_train)

train_transformed = pipeline.transform(X_train)
test_transformed = pipeline.transform(X_test)
# ```end