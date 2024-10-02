# ```python
import pandas as pd
original_data = 'original_data.csv'
clean_data_path = 'clean_data.csv'
df = pd.read_csv(original_data)
df['state'] = df['state'].str.strip()
df['city'] = df['city'].str.strip()
state_mapping = {
    'AZ': 'AZ',
    'SC': 'SC',
    'CO': 'CO',
    'CA': 'CA'
}
city_mapping = {
    'Ahwatukee': 'Ahwatukee',
    'Anthem': 'Anthem',
    'Apache Junction': 'Apache Junction',
    'Avondale': 'Avondale',
    'Buckeye': 'Buckeye',
    'Carefree': 'Carefree',
    'Casa Grande': 'Casa Grande',
    'Cave Creek': 'Cave Creek',
    'Chandler': 'Chandler',
    'Charleston': 'Charleston',
    'Coolidge': 'Coolidge',
    'El Mirage': 'El Mirage',
    'Florence': 'Florence',
    'Fort McDowell': 'Fort McDowell',
    'Fountain Hills': 'Fountain Hills',
    'Fountain Hls': 'Fountain Hills',
    'Gila Bend': 'Gila Bend',
    'Gilbert': 'Gilbert',
    'Glendale': 'Glendale',
    'Glendale Az': 'Glendale',
    'Gold Canyon': 'Gold Canyon',
    'Goldfield': 'Goldfield',
    'Goodyear': 'Goodyear',
    'Good Year': 'Goodyear',
    'Grand Junction': 'Grand Junction',
    'Guadalupe': 'Guadalupe',
    'Higley': 'Higley',
    'Laveen': 'Laveen',
    'Litchfield Park': 'Litchfield Park',
    'Maricopa': 'Maricopa',
    'Mesa': 'Mesa',
    'Morristown': 'Morristown',
    'North Pinal': 'North Pinal',
    'North Scottsdale': 'North Scottsdale',
    'Paradise Valley': 'Paradise Valley',
    'Peoria': 'Peoria',
    'Pheonix': 'Phoenix',
    'Phoenix': 'Phoenix',
    'Queen Creek': 'Queen Creek',
    'Rio Verde': 'Rio Verde',
    'Saguaro Lake': 'Saguaro Lake',
    'San Tan Valley': 'San Tan Valley',
    'Scottsdale': 'Scottsdale',
    'Scottsdale ': 'Scottsdale',
    'Sun City': 'Sun City',
    'Sun City Anthem': 'Sun City Anthem',
    'Sun City West': 'Sun City West',
    'Sun Lakes': 'Sun Lakes',
    'Surprise': 'Surprise',
    'Tempe': 'Tempe',
    'Tolleson': 'Tolleson',
    'Tonopah': 'Tonopah',
    'Tonto Basin': 'Tonto Basin',
    'Tortilla Flat': 'Tortilla Flat',
    'Tucson': 'Tucson',
    'Waddell': 'Waddell',
    'Wickenburg': 'Wickenburg',
    'Wittmann': 'Wittmann',
    'Youngtown': 'Youngtown',
    'Yuma': 'Yuma'
}
df['state'] = df['state'].map(state_mapping)
df['city'] = df['city'].map(city_mapping)

df['categories'] = df['categories'].str.split(',')
df = df.explode('categories')
df = pd.concat([df,pd.get_dummies(df['categories'].str.strip(), prefix='categories_')], axis=1)
df = df.drop(columns=['categories'])
df.to_csv(clean_data_path, index=False)
# ```