# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

categorical_features = ["DayOfWeek", "Airline"]

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough",  # Keep the remaining columns
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(max_leaf_nodes=500, n_jobs=multiprocessing.cpu_count()))
    ]
)

n_threads = multiprocessing.cpu_count()

pipeline.fit(train_data.drop("Delay", axis=1), train_data["Delay"])

train_predictions = pipeline.predict(train_data.drop("Delay", axis=1))
test_predictions = pipeline.predict(test_data.drop("Delay", axis=1))

Train_Accuracy = accuracy_score(train_data["Delay"], train_predictions)
Test_Accuracy = accuracy_score(test_data["Delay"], test_predictions)
Train_F1_score = f1_score(train_data["Delay"], train_predictions)
Test_F1_score = f1_score(test_data["Delay"], test_predictions)
Train_AUC = roc_auc_score(train_data["Delay"], train_predictions)
Test_AUC = roc_auc_score(test_data["Delay"], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end