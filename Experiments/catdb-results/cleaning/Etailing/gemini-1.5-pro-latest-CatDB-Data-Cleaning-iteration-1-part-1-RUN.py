# ```python
import pandas as pd
original_data = pd.read_csv('/home/saeed/Documents/Github/CatDB/Experiments/data/Etailing/Etailing_Google_clean.csv')
original_data['Which among the following E-Commerce Website have you ever opened?']=original_data['Which among the following E-Commerce Website have you ever opened?'].str.replace("Firstcry.com", "FirstCry")
original_data['Which among the following E-Commerce Website have you ever opened?']=original_data['Which among the following E-Commerce Website have you ever opened?'].str.replace("Lenskart.com", "Lenskart")
original_data['Which among the following E-Commerce Website have you ever opened?']=original_data['Which among the following E-Commerce Website have you ever opened?'].str.replace("Home Shop18", "HomeShop18")
original_data['Which among the following E-Commerce Website have you ever opened?']=original_data['Which among the following E-Commerce Website have you ever opened?'].str.replace("PayTm Mall", "PaytmMall")
original_data['Which among the following E-Commerce Website have you ever opened?']=original_data['Which among the following E-Commerce Website have you ever opened?'].str.replace("Bestylish", "Bewakoof.com")
original_data.to_csv('/home/saeed/Documents/Github/CatDB/Experiments/data/Etailing/Etailing_Google_clean.csv', index=False)
# ```end