# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/MD_MIX_Mini_Copy/MD_MIX_Mini_Copy_train.csv')
test_data = pd.read_csv('../../../data/MD_MIX_Mini_Copy/MD_MIX_Mini_Copy_test.csv')

# Drop the unnecessary columns
drop_cols = ['font_size','image_width_resolution','font_weight','image_height_resolution',
             'morph_gradient_kernel_size','image_mode','background_random_color_composition_params',
             'morph_gradient_kernel_shape','background_polygon_fill_color','background_polygon_outline_color',
             'margin_right','pre_elastic','margin_bottom','margin_top','margin_left']
train_data.drop(columns=drop_cols, inplace=True)
test_data.drop(columns=drop_cols, inplace=True)

# Define preprocessing for numerical columns
num_cols = ['outline_image_crop_y_plus_height','outline_image_crop_x_plus_width','background_image_crop_x',
            'foreground_image_crop_x_plus_width','background_image_crop_y_plus_height','background_image_crop_x_plus_width',
            'foreground_image_crop_y','outline_image_crop_x','foreground_image_crop_x','outline_image_crop_y',
            'background_image_crop_y','foreground_image_crop_y_plus_height','shear_x']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

# Define preprocessing for categorical columns
cat_cols = ['SUPER_CATEGORY','outline_image_name','foreground_image_name','outline','background_image_name',
            'outline_image_resized_height','outline_image_resized_width','outline_image_original_width',
            'foreground_image_original_width','foreground_image_resized_height','outline_image_original_height',
            'foreground_image_resized_width','background_image_resized_height','foreground_image_original_height',
            'background_image_resized_width','background_image_original_width','background_image_original_height',
            'outline_size','foreground','style_name','background','background_color','image_blending_method',
            'stroke_fill','variable_font_weight','shear_y']
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('convert_to_str', FunctionTransformer(lambda x: x.astype(str))),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Define the model 
model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Preprocessing of training data, fit model 
my_pipeline.fit(train_data.drop('CATEGORY', axis=1), train_data['CATEGORY'])

# Preprocessing of test data, get predictions
preds_test = my_pipeline.predict(test_data.drop('CATEGORY', axis=1))

# Evaluation
Train_R_Squared = my_pipeline.score(train_data.drop('CATEGORY', axis=1), train_data['CATEGORY'])
Test_R_Squared = my_pipeline.score(test_data.drop('CATEGORY', axis=1), test_data['CATEGORY'])
Train_RMSE = np.sqrt(mean_squared_error(train_data['CATEGORY'], my_pipeline.predict(train_data.drop('CATEGORY', axis=1))))
Test_RMSE = np.sqrt(mean_squared_error(test_data['CATEGORY'], preds_test))

print(f"Train_R_Squared:{Train_R_Squared}")   
print(f"Train_RMSE:{Train_RMSE}") 
print(f"Test_R_Squared:{Test_R_Squared}")   
print(f"Test_RMSE:{Test_RMSE}") 
# ```