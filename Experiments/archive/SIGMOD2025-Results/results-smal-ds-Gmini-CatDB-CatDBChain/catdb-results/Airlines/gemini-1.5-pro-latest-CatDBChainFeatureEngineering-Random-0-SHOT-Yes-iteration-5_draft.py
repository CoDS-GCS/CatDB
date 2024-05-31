# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

categorical_features = ["DayOfWeek", "Airline"]
numerical_features = ["AirportTo", "AirportFrom", "Flight", "Length", "Time"]

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ]
)
X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]

X_train_transformed = pipeline.fit_transform(X_train)

X_test_transformed = pipeline.transform(test_data)
# ```end