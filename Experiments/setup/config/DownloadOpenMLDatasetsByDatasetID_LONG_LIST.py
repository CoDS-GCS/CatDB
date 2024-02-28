import sys
import openml
from sklearn.model_selection import train_test_split
from pathlib import Path
from argparse import ArgumentParser
from DatasetPrepare import split_data_save
from DatasetPrepare import get_metadata
from DatasetPrepare import rename_col_names
from DatasetPrepare import save_config
import pandas as pd


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-out-path', type=str, default=None)
    parser.add_argument('--setting-out-path', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.data_out_path is None:
        raise Exception("--data-out-path is a required parameter!")    
    
    if args.setting_out_path is None:
        raise Exception("--setting-out-path is a required parameter!")
    
    return args


if __name__ == '__main__':
    args = parse_arguments()    
    
    # datasetIDs = [ (45693,'simulated_electricity','binary', 1),
    #                (23513,'KDD98','binary', 2),                  
    #                (45570,'Higgs','binary', 3),
    #                (45072,'airlines','binary', 4),
    #                (40514,'BNG_credit_g','binary', 5),
    #                (45579,'Microsoft','multiclass', 6),
    #                (45056,'cmc','multiclass', 7),
    #                (37,'diabetes','multiclass', 8),
    #                (43476,'3-million-Sudoku-puzzles-with-ratings','multiclass', 9),
    #                (155,'pokerhand','multiclass', 10),
    #                (4549,'Buzzinsocialmedia_Twitter','regression', 11),
    #                (45045,'delays_zurich_transport','regression', 12),
    #                (44065,'nyc-taxi-green-dec-2016','regression', 13),
    #                (44057,'black_friday','regression', 14),
    #                (42080,'federal_election','regression', 15),
    #               ]

    datasetIDs = [(45570,"Higgs","binary",1),
                (45668,"bates_classif_100","binary",2),
                (45665,"colon","binary",3),
                (45660,"simulated_electricity","binary",4),
                (45656,"simulated_adult","binary",5),
                (45704,"simulated_covertype","binary",6),
                (45703,"simulated_bank_marketing","binary",7),
                (45693,"simulated_electricity","binary",8),
                (45689,"simulated_adult","binary",9),
                (45672,"prostate","binary",10),
                (45669,"breast","binary",11),
                (42732,"sf-police-incidents","binary",12),
                (1218,"Click_prediction_small","binary",13),
                (1240,"AirlinesCodrnaAdult","binary",14),
                (354,"poker","binary",15),
                (1377,"BNG_kr-vs-kp","binary",16),
                (153,"Hyperplane_10_1E-4","binary",17),
                (152,"Hyperplane_10_1E-3","binary",18),
                (267,"BNG_heart-statlog","binary",19),
                (146,"BNG_ionosphere","binary",20),
                (142,"BNG_hepatitis","binary",21),
                (40514,"BNG_credit-g","binary",22),
                (40515,"BNG_spambase","binary",23),
                (140,"BNG_heart-statlog","binary",24),
                (43489,"Census-Augmented","binary",25),
                (262,"BNG_cylinder-bands","binary",26),
                (264,"BNG_sonar","binary",27),
                (45583,"click","binary",28),
                (161,"SEA_50","binary",29),
                (162,"SEA_50000","binary",30),
                (246,"BNG_labor","binary",31),
                (135,"BNG_spambase","binary",32),
                (257,"BNG_colic","binary",33),
                (256,"BNG_colic-ORIG","binary",34),
                (1211,"BNG_SPECT","binary",35),
                (1180,"BNG_spect_test","binary",36),
                (1235,"Agrawal1","binary",37),
                (41228,"Klaverjas2018","binary",38),
                (42742,"porto-seguro","binary",39),
                (293,"covertype","binary",40),
                (1169,"airlines","binary",41),
                (42344,"sf-police-incidents","binary",42),
                (1241,"codrnaNorm","binary",43),
                (351,"codrna","binary",44),
                (41147,"albert","binary",45),
                (43948,"covertype","binary",46),
                (1219,"Click_prediction_small","binary",47),
                (45567,"hcdr_main","binary",48),
                (1597,"creditcard","binary",49),
                (42397,"CreditCardFraudDetection","binary",50),
                (1110,"KDDCup99_full","multiclass",51),
                (42746,"KDDCup99","multiclass",52),
                (42553,"BitcoinHeist_Ransomware","multiclass",53),
                (42089,"vancouver_employee","multiclass",54),
                (42088,"beer_reviews","multiclass",55),
                (42132,"Traffic_violations","multiclass",56),
                (149,"CovPokElec","multiclass",57),
                (45274,"PASS","multiclass",58),
                (45579,"Microsoft","multiclass",59),
                (1567,"poker-hand","multiclass",60),
                (160,"RandomRBF_50_1E-4","multiclass",61),
                (159,"RandomRBF_50_1E-3","multiclass",62),
                (158,"RandomRBF_10_1E-4","multiclass",63),
                (157,"RandomRBF_10_1E-3","multiclass",64),
                (156,"RandomRBF_0_0","multiclass",65),
                (154,"LED_50000","multiclass",66),
                (42468,"hls4ml_lhc_jets_hlf","multiclass",67),
                (155,"pokerhand","multiclass",68),
                (1226,"Click_prediction_small","multiclass",69),
                (1596,"covertype","multiclass",70),
                (41960,"seattlecrime6","multiclass",71),
                (1113,"KDDCup99","multiclass",72),
                (44317,"Meta_Album_PLK_Extended","multiclass",73),
                (41167,"dionis","multiclass",74),
                (42803,"road-safety","multiclass",75),
                (1503,"spoken-arabic-digit","multiclass",76),
                (44340,"Meta_Album_INS_Extended","multiclass",77),
                (1483,"ldpa","multiclass",78),
                (1509,"walking-activity","multiclass",79),
                (44343,"Meta_Album_BTS_Extended","multiclass",80),
                (44327,"Meta_Album_PLT_NET_Extended","multiclass",81),
                (43044,"drug-directory","multiclass",82),
                (45282,"AfriSenti","multiclass",83),
                (180,"covertype","multiclass",84),
                (42396,"aloi","multiclass",85),
                (4541,"Diabetes130US","multiclass",86),
                (40672,"fars","multiclass",87),
                (41168,"jannis","multiclass",88),
                (44326,"Meta_Album_INS_2_Extended","multiclass",89),
                (42345,"Traffic_violations","multiclass",90),
                (40668,"connect-4","multiclass",91),
                (41169,"helena","multiclass",92),
                (45548,"Otto-Group-Product-Classification-Challenge","multiclass",93),
                (41166,"volkert","multiclass",94),
                (40685,"shuttle","multiclass",95),
                (44321,"Meta_Album_PLT_VIL_Extended","multiclass",96),
                (42734,"okcupid-stem","multiclass",97),
                (44320,"Meta_Album_BRD_Extended","multiclass",98),
                (40985,"tamilnadu-electricity","multiclass",99),
                (279,"meta_stream_intervals.arff","multiclass",100),
                (41001,"jungle_chess_2pcs_endgame_complete","multiclass",101),
                (41027,"jungle_chess_2pcs_raw_endgame_complete","multiclass",102),
                (44341,"Meta_Album_RSD_Extended","multiclass",103),
                (44338,"Meta_Album_AWA_Extended","multiclass",104),
                (44333,"Meta_Album_RSICB_Extended","multiclass",105),
                (45714,"PriceRunner","multiclass",106),
                (44324,"Meta_Album_RESISC_Extended","multiclass",107),
                (1537,"volcanoes-c1","multiclass",108),
                (45049,"MD_MIX_Mini_Copy","multiclass",109),
                (1481,"kr-vs-k","multiclass",110),
                (184,"kropt","multiclass",111),
                (45067,"okcupid_stem","multiclass",112),
                (44337,"Meta_Album_TEX_ALOT_Extended","multiclass",113),
                (44331,"Meta_Album_DOG_Extended","multiclass",114),
                (41671,"microaggregation2","multiclass",115),
                (6,"letter","multiclass",116),
                (45517,"timing-attack-dataset-30-micro-seconds-delay-2022-09-08","multiclass",117),
                (45501,"timing-attack-dataset-20-micro-seconds-delay-2022-09-15","multiclass",118),
                (45490,"timing-attack-dataset-15-micro-seconds-delay-2022-09-14","multiclass",119),
                (45484,"timing-attack-dataset-10-micro-seconds-delay-2022-09-20","multiclass",120),
                (45528,"timing-attack-dataset-35-micro-seconds-delay-2022-09-09","multiclass",121),
                (45498,"timing-attack-dataset-20-micro-seconds-delay-2022-09-09","multiclass",122),
                (45480,"timing-attack-dataset-10-micro-seconds-delay-2022-09-14","multiclass",123),
                (45481,"timing-attack-dataset-10-micro-seconds-delay-2022-09-15","multiclass",124),
                (45499,"timing-attack-dataset-20-micro-seconds-delay-2022-09-13","multiclass",125),
                (45511,"timing-attack-dataset-25-micro-seconds-delay-2022-09-15","multiclass",126),
                (45521,"timing-attack-dataset-30-micro-seconds-delay-2022-09-15","multiclass",127),
                (45509,"timing-attack-dataset-25-micro-seconds-delay-2022-09-13","multiclass",128),
                (45534,"timing-attack-dataset-35-micro-seconds-delay-2022-09-20","multiclass",129),
                (45488,"timing-attack-dataset-15-micro-seconds-delay-2022-09-09","multiclass",130),
                (45489,"timing-attack-dataset-15-micro-seconds-delay-2022-09-13","multiclass",131),
                (45478,"timing-attack-dataset-10-micro-seconds-delay-2022-09-09","multiclass",132),
                (45487,"timing-attack-dataset-15-micro-seconds-delay-2022-09-08","multiclass",133),
                (45497,"timing-attack-dataset-20-micro-seconds-delay-2022-09-08","multiclass",134),
                (45519,"timing-attack-dataset-30-micro-seconds-delay-2022-09-13","multiclass",135),
                (45531,"timing-attack-dataset-35-micro-seconds-delay-2022-09-15","multiclass",136)           
                  ]
       
   
    #dataset_list = 'row,orig_dataset_name,dataset_name,nrows,ncols,file_format,task_type,number_classes,original_url,target_feature,description\n'
    dataset_list =  pd.DataFrame(columns=["Row","ID","dataset_name", "orig_name","nrows","ncols","nclasses","target"])
    
    script_list =""
    for (dataset_id,dataset_name,task_type, dataset_index) in datasetIDs:        
        print(f" Downloading Dataset: dataset name={dataset_name}, dataset ID={dataset_id} \n")

        dataset = openml.datasets.get_dataset(dataset_id, download_qualities=False)
        data, y, categorical_indicator, attribute_names = dataset.get_data()
        target_attribute = dataset.default_target_attribute

        n_classes = data[dataset.default_target_attribute].nunique()
        if n_classes == 2:
                task_type = "binary"
        elif n_classes < 300 :
             task_type = "multiclass"
        else:
                task_type = "regression"            

         # Split and save original dataset
        # nrows, ncols, number_classes = get_metadata(data=data, target_attribute=target_attribute)
        # split_data_save(data=data, ds_name=dataset_name,out_path= args.data_out_path)
        # save_config(dataset_name=dataset_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

        # # Split and rename dataset-name, then save  
        # dataset_out_name = f"oml_dataset_{dataset_index}"
        # split_data_save(data=data, ds_name=dataset_out_name, out_path= args.data_out_path)
        # save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path)

        # Rename cols and dataset name, then split and save it
        dataset_out_name = f"oml_dataset_{dataset_index}_rnc"
        target_attribute, nrows, ncols, number_classes = rename_col_names(data=data, ds_name=dataset_out_name, target_attribute=target_attribute, out_path=args.data_out_path)
        save_config(dataset_name=dataset_out_name, target=target_attribute, task_type=task_type, data_out_path=args.data_out_path, setting_out_path=args.setting_out_path) 

        # "Row","ID","dataset_name", "orig_name","nrows","ncols","nclasses","target"
        dataset_list.loc[len(dataset_list)] = [dataset_index, dataset_id, dataset_out_name, dataset_name, nrows, ncols, number_classes, target_attribute]
        dataset_list.to_csv(f"{args.setting_out_path}/dataset_list.csv") 