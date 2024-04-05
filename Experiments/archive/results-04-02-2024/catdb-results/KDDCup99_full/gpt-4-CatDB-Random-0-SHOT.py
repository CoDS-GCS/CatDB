# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/KDDCup99_full/KDDCup99_full_train.csv')
test_data = pd.read_csv('../../../data/KDDCup99_full/KDDCup99_full_test.csv')
# ```end

# ```python
# Drop the unnecessary column
train_data.drop(columns=['lnum_outbound_cmds'], inplace=True)
test_data.drop(columns=['lnum_outbound_cmds'], inplace=True)
# ```end-dropping-columns

# ```python
# Define the preprocessing steps
numeric_features = ['wrong_fragment', 'lnum_shells', 'num_failed_logins', 'urgent', 'duration', 'src_bytes', 'lnum_access_files', 'lsu_attempted', 'dst_host_srv_count', 'hot', 'srv_count', 'dst_host_count', 'count', 'lnum_root', 'lnum_compromised', 'lnum_file_creations', 'dst_bytes', 'diff_srv_rate', 'dst_host_serror_rate', 'dst_host_diff_srv_rate', 'dst_host_rerror_rate', 'srv_serror_rate', 'serror_rate', 'srv_diff_host_rate', 'rerror_rate', 'dst_host_srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_srv_rerror_rate']
categorical_features = ['service', 'protocol_type', 'flag', 'lroot_shell', 'logged_in', 'land', 'is_guest_login', 'is_host_login']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# ```end

# ```python
# Define the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Combine preprocessing and classifier into a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
# ```end

# ```python
# Train the model
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
clf.fit(X_train, y_train)
# ```end

# ```python
# Evaluate the model
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, clf.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, clf.predict_proba(X_test))

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end