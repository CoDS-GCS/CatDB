#!/bin/bash

log_file_name='buildCatalog.dat'
data_path='data'

imdb_data="ImdbName.csv,ImdbTitleAkas.csv,ImdbTitleBasics.csv,ImdbTitleCrew.csv,ImdbTitleEpisode.csv,ImdbTitlePrincipals.csv,ImdbTitleRatings.csv"
imdb_root="IMDB"
imdb_format="csv"

yelp_data="yelp_academic_dataset_business.json,yelp_academic_dataset_review.json,yelp_academic_dataset_user.json,photos.json,yelp_academic_dataset_checkin.json,yelp_academic_dataset_tip.json"
yelp_data2="yelp_academic_dataset_business.json"

yelp_root="YELP"
yelp_format="json"


#SCRIPT="python Catalog.py ${data_path} ${imdb_root} ${imdb_data} ${imdb_format}"
#echo $SCRIPT
#start=$(date +%s%N)
#$SCRIPT
#end=$(date +%s%N)
#echo "Catalog,"${imdb_root}","$((($end - $start) / 1000000)) >>results/$log_file_name


SCRIPT="python Catalog.py ${data_path} ${yelp_root} ${yelp_data} ${yelp_format}"
echo $SCRIPT
start=$(date +%s%N)
$SCRIPT
end=$(date +%s%N)
echo "Catalog,"${yelp_root}","$((($end - $start) / 1000000)) >>results/$log_file_name