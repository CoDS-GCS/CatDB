# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_path = "/home/saeed/Documents/Github/CatDB/Experiments/data/airlines/airlines_train.csv"
test_path = "/home/saeed/Documents/Github/CatDB/Experiments/data/airlines/airlines_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
# end-load-dataset

# python-added-column
# Combine 'AirportFrom' and 'AirportTo' to create a new column 'AirportRoute'
# Usefulness: The 'AirportRoute' column captures the combination of origin and destination airports, providing additional information about the flight route.
df_train['AirportRoute'] = df_train['AirportFrom'] + ' - ' + df_train['AirportTo']
df_test['AirportRoute'] = df_test['AirportFrom'] + ' - ' + df_test['AirportTo']

# Calculate the average length and time of flights for each airline and create new columns 'AvgLengthByAirline' and 'AvgTimeByAirline'
# Usefulness: The average length and time of flights for each airline can provide insights into the typical duration and timing of flights operated by different airlines.
df_train['AvgLengthByAirline'] = df_train.groupby('Airline')['Length'].transform('mean')
df_train['AvgTimeByAirline'] = df_train.groupby('Airline')['Time'].transform('mean')

df_test['AvgLengthByAirline'] = df_test.groupby('Airline')['Length'].transform('mean')
df_test['AvgTimeByAirline'] = df_test.groupby('Airline')['Time'].transform('mean')

# Calculate the average length and time of flights for each route and create new columns 'AvgLengthByRoute' and 'AvgTimeByRoute'
# Usefulness: The average length and time of flights for each route can provide insights into the typical duration and timing of flights between different airport pairs.
df_train['AvgLengthByRoute'] = df_train.groupby('AirportRoute')['Length'].transform('mean')
df_train['AvgTimeByRoute'] = df_train.groupby('AirportRoute')['Time'].transform('mean')

df_test['AvgLengthByRoute'] = df_test.groupby('AirportRoute')['Length'].transform('mean')
df_test['AvgTimeByRoute'] = df_test.groupby('AirportRoute')['Time'].transform('mean')
# end-added-column

# python-dropping-columns
# Drop the unnecessary columns 'AirportFrom', 'AirportTo', 'Flight' 
# Explanation: These columns are dropped as they do not contribute useful information for predicting 'Delay'.
df_train.drop(columns=['AirportFrom', 'AirportTo', 'Flight'], inplace=True)
df_test.drop(columns=['AirportFrom', 'AirportTo', 'Flight'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for binary classification of 'Delay'
# Explanation: Random Forest Classifier is selected for its ability to handle a mix of categorical and numerical features, and its robustness against overfitting.
X_train = df_train.drop(columns=['Delay'])
y_train = df_train['Delay']

X_test = df_test.drop(columns=['Delay'])
y_test = df_test['Delay']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
acc
# end-evaluation
# 