# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/covertype/covertype_train.csv')
test_data = pd.read_csv('data/covertype/covertype_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'Hillshade_Difference' which represents the difference between Hillshade_9am and Hillshade_3pm
# Usefulness: This column captures the difference in hillshade between morning and afternoon, which can provide information about the terrain and vegetation density.
train_data['Hillshade_Difference'] = train_data['Hillshade_9am'] - train_data['Hillshade_3pm']
test_data['Hillshade_Difference'] = test_data['Hillshade_9am'] - test_data['Hillshade_3pm']

# Add new column 'Distance_To_Water' which represents the combined vertical and horizontal distance to hydrology
# Usefulness: This column captures the overall distance to water bodies, which can provide information about the proximity to water sources.
train_data['Distance_To_Water'] = train_data['Vertical_Distance_To_Hydrology'] + train_data['Horizontal_Distance_To_Hydrology']
test_data['Distance_To_Water'] = test_data['Vertical_Distance_To_Hydrology'] + test_data['Horizontal_Distance_To_Hydrology']

# Add new column 'Elevation_Slope' which represents the product of elevation and slope
# Usefulness: This column captures the interaction between elevation and slope, which can provide information about the steepness of the terrain at different elevations.
train_data['Elevation_Slope'] = train_data['Elevation'] * train_data['Slope']
test_data['Elevation_Slope'] = test_data['Elevation'] * test_data['Slope']
# end-added-column

# python-dropping-columns
# Drop the column 'Hillshade_Noon' as it may be redundant and not contribute significantly to the classification task
train_data.drop(columns=['Hillshade_Noon'], inplace=True)
test_data.drop(columns=['Hillshade_Noon'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for multiclass classification
# Explanation: Random Forest is a robust and effective algorithm for multiclass classification tasks, and it can handle both numerical and boolean features.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 