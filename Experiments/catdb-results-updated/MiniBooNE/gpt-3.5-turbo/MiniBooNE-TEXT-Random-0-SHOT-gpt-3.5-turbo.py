# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/MiniBooNE/MiniBooNE_train.csv')
test_data = pd.read_csv('data/MiniBooNE/MiniBooNE_test.csv')
# end-load-dataset

# python-added-column
# Add new columns
train_data['ParticleID_13_40'] = train_data['ParticleID_13'] + train_data['ParticleID_40']
train_data['ParticleID_34_46'] = train_data['ParticleID_34'] + train_data['ParticleID_46']
train_data['ParticleID_1_26'] = train_data['ParticleID_1'] + train_data['ParticleID_26']
train_data['ParticleID_17_3'] = train_data['ParticleID_17'] + train_data['ParticleID_3']
train_data['ParticleID_45_11'] = train_data['ParticleID_45'] + train_data['ParticleID_11']
train_data['ParticleID_12_10'] = train_data['ParticleID_12'] + train_data['ParticleID_10']
train_data['ParticleID_39_8'] = train_data['ParticleID_39'] + train_data['ParticleID_8']
train_data['ParticleID_24_33'] = train_data['ParticleID_24'] + train_data['ParticleID_33']
train_data['ParticleID_9_23'] = train_data['ParticleID_9'] + train_data['ParticleID_23']
train_data['ParticleID_28_15'] = train_data['ParticleID_28'] + train_data['ParticleID_15']
train_data['ParticleID_0_14'] = train_data['ParticleID_0'] + train_data['ParticleID_14']
train_data['ParticleID_35_29'] = train_data['ParticleID_35'] + train_data['ParticleID_29']
train_data['ParticleID_4_7'] = train_data['ParticleID_4'] + train_data['ParticleID_7']
train_data['ParticleID_41_16'] = train_data['ParticleID_41'] + train_data['ParticleID_16']
train_data['ParticleID_32_37'] = train_data['ParticleID_32'] + train_data['ParticleID_37']
train_data['ParticleID_5_42'] = train_data['ParticleID_5'] + train_data['ParticleID_42']
train_data['ParticleID_30_38'] = train_data['ParticleID_30'] + train_data['ParticleID_38']
train_data['ParticleID_43_49'] = train_data['ParticleID_43'] + train_data['ParticleID_49']
train_data['ParticleID_2_31'] = train_data['ParticleID_2'] + train_data['ParticleID_31']
train_data['ParticleID_20_19'] = train_data['ParticleID_20'] + train_data['ParticleID_19']
train_data['ParticleID_48_44'] = train_data['ParticleID_48'] + train_data['ParticleID_44']
train_data['ParticleID_18_6'] = train_data['ParticleID_18'] + train_data['ParticleID_6']
train_data['ParticleID_27_25'] = train_data['ParticleID_27'] + train_data['ParticleID_25']
train_data['ParticleID_47_22'] = train_data['ParticleID_47'] + train_data['ParticleID_22']
train_data['ParticleID_21_36'] = train_data['ParticleID_21'] + train_data['ParticleID_36']

test_data['ParticleID_13_40'] = test_data['ParticleID_13'] + test_data['ParticleID_40']
test_data['ParticleID_34_46'] = test_data['ParticleID_34'] + test_data['ParticleID_46']
test_data['ParticleID_1_26'] = test_data['ParticleID_1'] + test_data['ParticleID_26']
test_data['ParticleID_17_3'] = test_data['ParticleID_17'] + test_data['ParticleID_3']
test_data['ParticleID_45_11'] = test_data['ParticleID_45'] + test_data['ParticleID_11']
test_data['ParticleID_12_10'] = test_data['ParticleID_12'] + test_data['ParticleID_10']
test_data['ParticleID_39_8'] = test_data['ParticleID_39'] + test_data['ParticleID_8']
test_data['ParticleID_24_33'] = test_data['ParticleID_24'] + test_data['ParticleID_33']
test_data['ParticleID_9_23'] = test_data['ParticleID_9'] + test_data['ParticleID_23']
test_data['ParticleID_28_15'] = test_data['ParticleID_28'] + test_data['ParticleID_15']
test_data['ParticleID_0_14'] = test_data['ParticleID_0'] + test_data['ParticleID_14']
test_data['ParticleID_35_29'] = test_data['ParticleID_35'] + test_data['ParticleID_29']
test_data['ParticleID_4_7'] = test_data['ParticleID_4'] + test_data['ParticleID_7']
test_data['ParticleID_41_16'] = test_data['ParticleID_41'] + test_data['ParticleID_16']
test_data['ParticleID_32_37'] = test_data['ParticleID_32'] + test_data['ParticleID_37']
test_data['ParticleID_5_42'] = test_data['ParticleID_5'] + test_data['ParticleID_42']
test_data['ParticleID_30_38'] = test_data['ParticleID_30'] + test_data['ParticleID_38']
test_data['ParticleID_43_49'] = test_data['ParticleID_43'] + test_data['ParticleID_49']
test_data['ParticleID_2_31'] = test_data['ParticleID_2'] + test_data['ParticleID_31']
test_data['ParticleID_20_19'] = test_data['ParticleID_20'] + test_data['ParticleID_19']
test_data['ParticleID_48_44'] = test_data['ParticleID_48'] + test_data['ParticleID_44']
test_data['ParticleID_18_6'] = test_data['ParticleID_18'] + test_data['ParticleID_6']
test_data['ParticleID_27_25'] = test_data['ParticleID_27'] + test_data['ParticleID_25']
test_data['ParticleID_47_22'] = test_data['ParticleID_47'] + test_data['ParticleID_22']
test_data['ParticleID_21_36'] = test_data['ParticleID_21'] + test_data['ParticleID_36']
# end-added-column

# python-dropping-columns
# Drop redundant columns
train_data.drop(columns=['ParticleID_13', 'ParticleID_40', 'ParticleID_34', 'ParticleID_46', 'ParticleID_1', 
                         'ParticleID_26', 'ParticleID_17', 'ParticleID_3', 'ParticleID_45', 'ParticleID_11', 
                         'ParticleID_12', 'ParticleID_10', 'ParticleID_39', 'ParticleID_8', 'ParticleID_24', 
                         'ParticleID_33', 'ParticleID_9', 'ParticleID_23', 'ParticleID_28', 'ParticleID_15', 
                         'ParticleID_0', 'ParticleID_14', 'ParticleID_35', 'ParticleID_29', 'ParticleID_4', 
                         'ParticleID_7', 'ParticleID_41', 'ParticleID_16', 'ParticleID_32', 'ParticleID_37', 
                         'ParticleID_5', 'ParticleID_42', 'ParticleID_30', 'ParticleID_38', 'ParticleID_43', 
                         'ParticleID_49', 'ParticleID_2', 'ParticleID_31', 'ParticleID_20', 'ParticleID_19', 
                         'ParticleID_48', 'ParticleID_44', 'ParticleID_18', 'ParticleID_6', 'ParticleID_27', 
                         'ParticleID_25', 'ParticleID_47', 'ParticleID_22', 'ParticleID_21', 'ParticleID_36'], inplace=True)

test_data.drop(columns=['ParticleID_13', 'ParticleID_40', 'ParticleID_34', 'ParticleID_46', 'ParticleID_1', 
                        'ParticleID_26', 'ParticleID_17', 'ParticleID_3', 'ParticleID_45', 'ParticleID_11', 
                        'ParticleID_12', 'ParticleID_10', 'ParticleID_39', 'ParticleID_8', 'ParticleID_24', 
                        'ParticleID_33', 'ParticleID_9', 'ParticleID_23', 'ParticleID_28', 'ParticleID_15', 
                        'ParticleID_0', 'ParticleID_14', 'ParticleID_35', 'ParticleID_29', 'ParticleID_4', 
                        'ParticleID_7', 'ParticleID_41', 'ParticleID_16', 'ParticleID_32', 'ParticleID_37', 
                        'ParticleID_5', 'ParticleID_42', 'ParticleID_30', 'ParticleID_38', 'ParticleID_43', 
                        'ParticleID_49', 'ParticleID_2', 'ParticleID_31', 'ParticleID_20', 'ParticleID_19', 
                        'ParticleID_48', 'ParticleID_44', 'ParticleID_18', 'ParticleID_6', 'ParticleID_27', 
                        'ParticleID_25', 'ParticleID_47', 'ParticleID_22', 'ParticleID_21', 'ParticleID_36'], inplace=True)
# end-dropping-columns

# python-training-technique
# Prepare the data for training
X_train = train_data.drop(columns=['signal'])
y_train = train_data['signal']

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate on the test set
X_test = test_data.drop(columns=['signal'])
y_test = test_data['signal']

X_test_scaled = scaler.transform(X_test)
y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation