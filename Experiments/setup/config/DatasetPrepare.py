import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
import re
import numpy as np
import yaml

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-root-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--target-attribute', type=str, default=None)
    parser.add_argument('--task-type', type=str, default=None)
    parser.add_argument('--target-table', type=str, default=None)
    parser.add_argument('--multi-table', type=str, default=None)
    parser.add_argument('--mtos', type=str, default=None)
    parser.add_argument('--dataset-description', type=str, default="")
    parser.add_argument('--data-out-path', type=str, default=None)    
    parser.add_argument('--catalog-root-path', type=str, default=None)
    
    
    args = parser.parse_args()
    return args


def split_data_save(data, ds_name, out_path, target_table: str=None, write_data: bool=True):
    if target_table is None:
        target_table = ds_name
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
    _, data_verify =  train_test_split(data_train, test_size=0.1, random_state=42)
    Path(f"{out_path}/{ds_name}").mkdir(parents=True, exist_ok=True)

    if write_data:
        data.to_csv(f'{out_path}/{ds_name}/{target_table}.csv', index=False)
    data_train.to_csv(f'{out_path}/{ds_name}/{target_table}_train.csv', index=False)
    data_test.to_csv(f'{out_path}/{ds_name}/{target_table}_test.csv', index=False)
    data_verify.to_csv(f'{out_path}/{ds_name}/{target_table}_verify.csv', index=False)

def get_metadata(data, target_attribute):
    (nrows, ncols) = data.shape
    number_classes = 'N/A'
    n_classes = data[target_attribute].nunique()
    number_classes = f'{n_classes}' 

    return nrows, ncols, number_classes

def rename_col_names(data, ds_name, target_attribute, out_path):

    nrows, ncols, number_classes = get_metadata(data=data, target_attribute=target_attribute)

    colnams = data.columns
    new_colnams=dict()
    i = 1
    for col in colnams:
        new_colnams[col]= f"c_{i}"
        i +=1
    data = data.rename(columns=new_colnams)  
    target_attribute = new_colnams[target_attribute] 

    split_data_save(data=data, ds_name=ds_name, out_path=out_path)
    
    return target_attribute, nrows, ncols, number_classes    


def refactor_openml_description(description):
    """Refactor the description of an openml dataset to remove the irrelevant parts."""
    if description is None:
        return None
    splits = re.split("\n", description)
    blacklist = [
        "Please cite",
        "Author",
        "Source",
        "Author:",
        "Source:",
        "Please cite:",
    ]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n", np.array(splits)[sel].tolist())

    splits = re.split("###", description)
    blacklist = ["Relevant Papers"]
    sel = ~np.array(
        [
            np.array([blacklist_ in splits[i] for blacklist_ in blacklist]).any()
            for i in range(len(splits))
        ]
    )
    description = str.join("\n\n", np.array(splits)[sel].tolist())
    return description


def save_config(dataset_name,target, task_type, data_out_path, description=None, multi_table: bool=False, target_table: str=None):
    if target_table is None:
        target_table=dataset_name

    config_strs = [f"- name: {dataset_name}",
                       "  dataset:",
                       f"    multi_table: {multi_table}",
                       f"    train: \'{dataset_name}/{target_table}_train.csv\'",
                       f"    test: \'{dataset_name}/{target_table}_test.csv\'",
                       f"    verify: \'{dataset_name}/{target_table}_verify.csv\'",
                       f"    target_table: {target_table}",
                       f"    target: '{target}'",
                       f"    type: {task_type}"
                       "\n"]
    config_str = "\n".join(config_strs)

    yaml_file_local = f'{data_out_path}/{dataset_name}/{dataset_name}.yaml'
    f_local = open(yaml_file_local, 'w')
    f_local.write("--- \n \n")
    f_local.write(config_str)
    f_local.close() 

    des = description
    if description is None:
        des = ""
    else:
        des = refactor_openml_description(description=description)    
        
    description_file = f'{data_out_path}/{dataset_name}/{dataset_name}.txt'
    f = open(description_file, 'w')
    f.write(des)
    f.close()


def load_dependency_info(dependency_file: str, datasource_name: str):
    with open(dependency_file, "r") as f:
        try:
            dep = yaml.load(f, Loader=yaml.FullLoader)
            ds_name = dep[0].get('name')
            if ds_name == datasource_name:
                tbls = dict()
                for k, v in dep[0].get('tables').items():                    
                    FKs = dep[0].get('tables').get(k).get('FK')
                    d = []
                    if FKs is not None:
                        d = FKs.split(",")

                    tbls[k] = d

                return tbls

        except yaml.YAMLError as ex:
            raise Exception(ex)
        except:
            return None
        
def create_join(root_path: str, dataset_name: str, main_table: str ,relations: dict):
    if dataset_name == "Yelp":
        return create_join_Yelp(root_path=root_path, dataset_name=dataset_name)
    elif dataset_name == "Accidents":
        return create_join_Accidents(root_path=root_path, dataset_name=dataset_name)
    elif dataset_name == "Airline":
        return create_join_Airline(root_path=root_path, dataset_name=dataset_name)
    elif dataset_name == "Financial":
        return create_join_Financial(root_path=root_path, dataset_name=dataset_name)
    elif dataset_name == "IMDB":
        return create_join_IMDB(root_path=root_path, dataset_name=dataset_name)
    elif dataset_name == "IMDB-IJS":
        return create_join_IMDB_IJS(root_path=root_path, dataset_name=dataset_name)    
    elif dataset_name == "Walmart":
        return create_join_Walmart(root_path=root_path, dataset_name=dataset_name)     
    else:
        return None
    # path = f'{root_path}/{dataset_name}/{main_table}.csv'
    # data = pd.read_csv(path, low_memory=False, encoding="ISO-8859-1")
    
    # for fks in relations[main_table]:
    #     fk_item = fks.split(" ")
    #     fk = fk_item[0][1: len(fk_item[0])-1]
    #     ref_tbl = fk_item[2]
    #     id = fk_item[3][1: len(fk_item[3])-1]
    #     ref_tbl_data = create_join(root_path=root_path, dataset_name=dataset_name, main_table=ref_tbl, relations=relations)
    #     cols_ref_tbl_data =[v for v in list(ref_tbl_data.columns.values) if v != id]
    #     cols_data = [ v for v in list(data.columns.values) if v != fk]        
    #     interset = list(set(cols_ref_tbl_data) & set(cols_data))
    #     if len(interset) > 0:
    #         for c in interset:
    #             ref_tbl_data = ref_tbl_data.rename(columns={c : f'{ref_tbl}.{c}'})
    #             data = data.rename(columns={c : f'{main_table}.{c}'})

    #     ref_tbl_data = ref_tbl_data.rename(columns={id : f'{ref_tbl}.{id}'})
    #     data = data.rename(columns={fk : f'{main_table}.{fk}'})    

    #     data = pd.merge(data,ref_tbl_data, how='left',left_on=[fk],right_on=[f'{ref_tbl}.{id}'])
    #     data = data.drop(f'{ref_tbl}.{id}', axis=1)
    #     data = data.rename(columns={f'{main_table}.{fk}': fk})

    # return data           

def create_join_Yelp(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    Business = pd.read_csv(f"{path}/Business.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"stars":"Business.stars"})
    Checkins = pd.read_csv(f"{path}/Checkins.csv", low_memory=False, encoding="ISO-8859-1")
    Reviews = pd.read_csv(f"{path}/Reviews.csv", low_memory=False, encoding="ISO-8859-1")
    Users = pd.read_csv(f"{path}/Users.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"votes_funny":"Users.votes_funny", "votes_useful":"Users.votes_useful", "votes_cool":"Users.votes_cool"})

    Business_Checkins = pd.merge(Business, Checkins, how='left',left_on=["business_id"],right_on=['business_id'])
    data = pd.merge(Reviews,Business_Checkins, how='left',left_on=["business_id"],right_on=['business_id'])
    data = pd.merge(data,Users, how='left',left_on=["user_id"],right_on=['user_id'])
    return data

def create_join_Accidents(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    upravna_enota = pd.read_csv(f"{path}/upravna_enota.csv", low_memory=False, encoding="ISO-8859-1")
    nesreca = pd.read_csv(f"{path}/nesreca.csv", low_memory=False, encoding="ISO-8859-1")
    oseba = pd.read_csv(f"{path}/oseba.csv", low_memory=False, encoding="ISO-8859-1")

    nesreca_upravna_enota = pd.merge(nesreca,upravna_enota, how='left',left_on=["upravna_enota"],right_on=['id_upravna_enota']).rename(columns={"ime_upravna_enota":"nesreca.ime_upravna_enota","st_prebivalcev":"nesreca.st_prebivalcev","povrsina":"nesreca.povrsina"})

    oseba_upravna_enota = pd.merge(oseba,upravna_enota, how='left',left_on=["upravna_enota"],right_on=['id_upravna_enota']).rename(columns={"ime_upravna_enota":"oseba.ime_upravna_enota","st_prebivalcev":"oseba.st_prebivalcev","povrsina":"oseba.povrsina"})

    data = pd.merge(oseba_upravna_enota,nesreca_upravna_enota, how='left',left_on=["id_nesreca"],right_on=['id_nesreca'])

    return data

def create_join_Airline(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    L_AIRLINE_ID = pd.read_csv(f"{path}/L_AIRLINE_ID.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"Airline"})

    L_AIRPORT = pd.read_csv(f"{path}/L_AIRPORT.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"Des_Description"})

    L_AIRPORT_ID = pd.read_csv(f"{path}/L_AIRPORT_ID.csv", low_memory=False, encoding="ISO-8859-1")
    L_AIRPORT_ID_DestAirport = L_AIRPORT_ID.copy(deep=True).rename(columns={"Description":"DestAirport_Description"})
    L_AIRPORT_ID_Div1Airport = L_AIRPORT_ID.copy(deep=True).rename(columns={"Description":"Div1Airport_Description"})
    L_AIRPORT_ID_Div2Airport = L_AIRPORT_ID.copy(deep=True).rename(columns={"Description":"Div2Airport_Description"})
    L_AIRPORT_ID_OriginAirport = L_AIRPORT_ID.copy(deep=True).rename(columns={"Description":"OriginAirport_Description"})


    L_AIRPORT_SEQ_ID = pd.read_csv(f"{path}/L_AIRPORT_SEQ_ID.csv", low_memory=False, encoding="ISO-8859-1")
    L_AIRPORT_SEQ_ID_DestAirportSeq = L_AIRPORT_SEQ_ID.copy(deep=True).rename(columns={"Description":"DestAirportSeq_Description"})
    L_AIRPORT_SEQ_ID_Div1AirportSeq = L_AIRPORT_SEQ_ID.copy(deep=True).rename(columns={"Description":"Div1AirportSeq_Description"})
    L_AIRPORT_SEQ_ID_Div2AirportSeq = L_AIRPORT_SEQ_ID.copy(deep=True).rename(columns={"Description":"Div2AirportSeq_Description"})
    L_AIRPORT_SEQ_ID_OriginAirportSeq = L_AIRPORT_SEQ_ID.copy(deep=True).rename(columns={"Description":"OriginAirportSeq_Description"})


    L_CANCELLATION = pd.read_csv(f"{path}/L_CANCELLATION.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"Cancellation_Description"})

    L_CITY_MARKET_ID = pd.read_csv(f"{path}/L_CITY_MARKET_ID.csv", low_memory=False, encoding="ISO-8859-1")
    L_CITY_MARKET_ID_DestCityMarket = L_CITY_MARKET_ID.copy(deep=True).rename(columns={"Description":"DestCityMarket_Description"})
    L_CITY_MARKET_ID_OriginCityMarket = L_CITY_MARKET_ID.copy(deep=True).rename(columns={"Description":"OriginCityMarket_Description"})


    L_DEPARRBLK = pd.read_csv(f"{path}/L_DEPARRBLK.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"DepTimeBlk_Description"})

    L_DISTANCE_GROUP_250 = pd.read_csv(f"{path}/L_DISTANCE_GROUP_250.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"DistanceGroup_Description"})

    L_DIVERSIONS = pd.read_csv(f"{path}/L_DIVERSIONS.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"DivAirportLandings_Description"})

    L_MONTHS = pd.read_csv(f"{path}/L_MONTHS.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"Month_Description"})

    L_ONTIME_DELAY_GROUPS = pd.read_csv(f"{path}/L_ONTIME_DELAY_GROUPS.csv", low_memory=False, encoding="ISO-8859-1")
    L_ONTIME_DELAY_GROUPS_ArrivalDelayGroups = L_ONTIME_DELAY_GROUPS.copy(deep=True).rename(columns={"Description":"ArrivalDelayGroups_Description"})
    L_ONTIME_DELAY_GROUPS_DepartureDelayGroups = L_ONTIME_DELAY_GROUPS.copy(deep=True).rename(columns={"Description":"DepartureDelayGroups_Description"})


    L_QUARTERS = pd.read_csv(f"{path}/L_QUARTERS.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"Quarter_Description"})

    L_STATE_ABR_AVIATION = pd.read_csv(f"{path}/L_STATE_ABR_AVIATION.csv", low_memory=False, encoding="ISO-8859-1")
    L_STATE_ABR_AVIATION_DestState = L_STATE_ABR_AVIATION.copy(deep=True).rename(columns={"Description":"DestState_Description"})
    L_STATE_ABR_AVIATION_OriginState = L_STATE_ABR_AVIATION.copy(deep=True).rename(columns={"Description":"OriginState_Description"})

    L_STATE_FIPS = pd.read_csv(f"{path}/L_STATE_FIPS.csv", low_memory=False, encoding="ISO-8859-1")
    L_STATE_FIPS_DestStateFips = L_STATE_FIPS.copy(deep=True).rename(columns={"Description":"DestStateFips_Description"})
    L_STATE_FIPS_OriginStateFips = L_STATE_FIPS.copy(deep=True).rename(columns={"Description":"OriginStateFips_Description"})

    L_UNIQUE_CARRIERS = pd.read_csv(f"{path}/L_UNIQUE_CARRIERS.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"UniqueCarrier_Description"})

    L_WEEKDAYS = pd.read_csv(f"{path}/L_WEEKDAYS.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"Description":"DayOfWeek_Description"})

    L_WORLD_AREA_CODES = pd.read_csv(f"{path}/L_WORLD_AREA_CODES.csv", low_memory=False, encoding="ISO-8859-1")
    L_WORLD_AREA_CODES_DestWac = L_WORLD_AREA_CODES.copy(deep=True).rename(columns={"Description":"DestWac_Description"})
    L_WORLD_AREA_CODES_OriginWac = L_WORLD_AREA_CODES.copy(deep=True).rename(columns={"Description":"OriginWac_Description"})

    L_YESNO_RESP = pd.read_csv(f"{path}/L_YESNO_RESP.csv", low_memory=False, encoding="ISO-8859-1")
    L_YESNO_RESP_ArrDel15 = L_YESNO_RESP.copy(deep=True).rename(columns={"Description":"ArrDel15_Description"})
    L_YESNO_RESP_Cancelled = L_YESNO_RESP.copy(deep=True).rename(columns={"Description":"Cancelled_Description"})
    L_YESNO_RESP_DepDel15 = L_YESNO_RESP.copy(deep=True).rename(columns={"Description":"DepDel15_Description"})
    L_YESNO_RESP_Diverted = L_YESNO_RESP.copy(deep=True).rename(columns={"Description":"Diverted_Description"})

    data = pd.read_csv(f"{path}/On_Time_On_Time_Performance_2016_1.csv", low_memory=False, encoding="ISO-8859-1")    

    data = pd.merge(data,L_AIRLINE_ID, how='left',left_on=["AirlineID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT, how='left',left_on=["Dest"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    
    data = pd.merge(data,L_AIRPORT_ID_DestAirport, how='left',left_on=["DestAirportID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT_ID_Div1Airport, how='left',left_on=["Div1AirportID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT_ID_Div2Airport, how='left',left_on=["Div2AirportID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT_ID_OriginAirport, how='left',left_on=["OriginAirportID"],right_on=['Code'])
    data = data.drop('Code', axis=1)

    data = pd.merge(data,L_AIRPORT_SEQ_ID_DestAirportSeq, how='left',left_on=["DestAirportSeqID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT_SEQ_ID_Div1AirportSeq, how='left',left_on=["Div1AirportSeqID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT_SEQ_ID_Div2AirportSeq, how='left',left_on=["Div2AirportSeqID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_AIRPORT_SEQ_ID_OriginAirportSeq, how='left',left_on=["OriginAirportSeqID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    
    data = pd.merge(data,L_CANCELLATION, how='left',left_on=["CancellationCode"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    
    data = pd.merge(data,L_CITY_MARKET_ID_DestCityMarket, how='left',left_on=["DestCityMarketID"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_CITY_MARKET_ID_OriginCityMarket, how='left',left_on=["OriginCityMarketID"],right_on=['Code'])
    data = data.drop('Code', axis=1)

    data = pd.merge(data,L_DEPARRBLK, how='left',left_on=["DepTimeBlk"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_DISTANCE_GROUP_250, how='left',left_on=["DistanceGroup"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_DIVERSIONS, how='left',left_on=["DivAirportLandings"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_MONTHS, how='left',left_on=["Month"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    
    data = pd.merge(data,L_ONTIME_DELAY_GROUPS_ArrivalDelayGroups, how='left',left_on=["ArrivalDelayGroups"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_ONTIME_DELAY_GROUPS_DepartureDelayGroups, how='left',left_on=["DepartureDelayGroups"],right_on=['Code'])
    data = data.drop('Code', axis=1)

    data = pd.merge(data,L_QUARTERS, how='left',left_on=["Quarter"],right_on=['Code'])
    data = data.drop('Code', axis=1)

    data = pd.merge(data,L_STATE_ABR_AVIATION_DestState, how='left',left_on=["DestState"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_STATE_ABR_AVIATION_OriginState, how='left',left_on=["OriginState"],right_on=['Code'])
    data = data.drop('Code', axis=1)


    data = pd.merge(data,L_STATE_FIPS_DestStateFips, how='left',left_on=["DestStateFips"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_STATE_FIPS_OriginStateFips, how='left',left_on=["OriginStateFips"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    
    data = pd.merge(data,L_UNIQUE_CARRIERS, how='left',left_on=["UniqueCarrier"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_WEEKDAYS, how='left',left_on=["DayOfWeek"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    
    data = pd.merge(data,L_WORLD_AREA_CODES_DestWac, how='left',left_on=["DestWac"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_WORLD_AREA_CODES_OriginWac, how='left',left_on=["OriginWac"],right_on=['Code'])
    data = data.drop('Code', axis=1)

    data = pd.merge(data,L_YESNO_RESP_ArrDel15, how='left',left_on=["ArrDel15"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_YESNO_RESP_Cancelled, how='left',left_on=["Cancelled"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_YESNO_RESP_DepDel15, how='left',left_on=["DepDel15"],right_on=['Code'])
    data = data.drop('Code', axis=1)
    data = pd.merge(data,L_YESNO_RESP_Diverted, how='left',left_on=["Diverted"],right_on=['Code'])
    data = data.drop('Code', axis=1)

    return data

def create_join_Financial(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    account = pd.read_csv(f"{path}/account.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"date":"account.date"})
    card = pd.read_csv(f"{path}/card.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"type":"card.type"})
    client = pd.read_csv(f"{path}/client.csv", low_memory=False, encoding="ISO-8859-1")
    disp = pd.read_csv(f"{path}/disp.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"type":"disp.type"})
    district = pd.read_csv(f"{path}/district.csv", low_memory=False, encoding="ISO-8859-1")
    loan = pd.read_csv(f"{path}/loan.csv", low_memory=False, encoding="ISO-8859-1")
    order = pd.read_csv(f"{path}/order.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"amount":"order.amount", "k_symbol":"order.k_symbol"})
    trans = pd.read_csv(f"{path}/trans.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"amount":"trans.amount", "date":"trans.date", "type":"trans.type","k_symbol":"trans.k_symbol"})
    
    data = pd.merge(loan,account, how='left',left_on=["account_id"],right_on=['account_id'])
    data = pd.merge(data,district, how='left',left_on=["district_id"],right_on=['district_id'])
    data = pd.merge(data, disp, how='left',left_on=["account_id"],right_on=['account_id'])
    data = pd.merge(data, card, how='left',left_on=["disp_id"],right_on=['disp_id'])
    
    client_district = pd.merge(client,district, how='inner',left_on=["district_id"],right_on=['district_id']).rename(columns={"A2":"client.A2","A3":"client.A3","A4":"client.A4","A5":"client.A5","A6":"client.A6","A7":"client.A7","A8":"client.A8","A9":"client.A9","A10":"client.A10","A11":"client.A11","A12":"client.A12","A13":"client.A13","A14":"client.A14","A15":"client.A15","A16":"client.A16"})
    client_district = client_district.drop('district_id', axis=1)    

    data = pd.merge(data, client_district, how='left',left_on=["client_id"],right_on=['client_id'])
    data = pd.merge(data, order, how='left',left_on=["account_id"],right_on=['account_id'])
    data = pd.merge(data, trans, how='left',left_on=["account_id"],right_on=['account_id'])
    return data

def create_join_IMDB(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    actors = pd.read_csv(f"{path}/actors.csv", low_memory=False, encoding="ISO-8859-1")
    business = pd.read_csv(f"{path}/business.csv", low_memory=False, encoding="ISO-8859-1")
    countries = pd.read_csv(f"{path}/countries.csv", low_memory=False, encoding="ISO-8859-1")
    directors = pd.read_csv(f"{path}/directors.csv", low_memory=False, encoding="ISO-8859-1")
    distributors = pd.read_csv(f"{path}/distributors.csv", low_memory=False, encoding="ISO-8859-1")
    editors = pd.read_csv(f"{path}/editors.csv", low_memory=False, encoding="ISO-8859-1")
    genres = pd.read_csv(f"{path}/genres.csv", low_memory=False, encoding="ISO-8859-1")
    language = pd.read_csv(f"{path}/language.csv", low_memory=False, encoding="ISO-8859-1")
    movies = pd.read_csv(f"{path}/movies.csv", low_memory=False, encoding="ISO-8859-1")
    movies2actors = pd.read_csv(f"{path}/movies2actors.csv", low_memory=False, encoding="ISO-8859-1")
    movies2directors = pd.read_csv(f"{path}/movies2directors.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"addition":"movies2directors.addition"})
    movies2editors = pd.read_csv(f"{path}/movies2editors.csv", low_memory=False, encoding="ISO-8859-1")
    movies2producers = pd.read_csv(f"{path}/movies2producers.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"addition":"movies2producers.addition"})
    movies2writers = pd.read_csv(f"{path}/movies2writers.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"addition":"movies2writers.addition"})
    prodcompanies = pd.read_csv(f"{path}/prodcompanies.csv", low_memory=False, encoding="ISO-8859-1")
    producers = pd.read_csv(f"{path}/producers.csv", low_memory=False, encoding="ISO-8859-1")
    ratings = pd.read_csv(f"{path}/ratings.csv", low_memory=False, encoding="ISO-8859-1")
    runningtimes = pd.read_csv(f"{path}/runningtimes.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"addition":"runningtimes.addition"})
    writers = pd.read_csv(f"{path}/writers.csv", low_memory=False, encoding="ISO-8859-1")

   
    movies_data = pd.merge(movies, movies2actors, how='inner',left_on=["movieid"],right_on=['movieid'])
    
    movies_data = pd.merge(movies_data, movies2directors, how='inner',left_on=["movieid"],right_on=['movieid'])
    movies_data = pd.merge(movies_data, movies2editors, how='inner',left_on=["movieid"],right_on=['movieid'])
    movies_data = pd.merge(movies_data, movies2producers, how='inner',left_on=["movieid"],right_on=['movieid'])
    movies_data = pd.merge(movies_data, movies2writers, how='inner',left_on=["movieid"],right_on=['movieid'])
    
    movies_data = pd.merge(movies_data, prodcompanies, how='inner',left_on=["movieid"],right_on=['movieid'])
    movies_data = pd.merge(movies_data, ratings, how='inner',left_on=["movieid"],right_on=['movieid'])
    movies_data = pd.merge(movies_data, genres, how='inner',left_on=["movieid"],right_on=['movieid'])
    movies_data = pd.merge(movies_data, language, how='inner',left_on=["movieid"],right_on=['movieid'])
    # movies_data = pd.merge(movies_data, distributors, how='inner',left_on=["movieid"],right_on=['movieid'])
    # movies_data = pd.merge(movies_data, business, how='inner',left_on=["movieid"],right_on=['movieid'])
    # movies_data = pd.merge(movies_data, runningtimes, how='inner',left_on=["movieid"],right_on=['movieid'])
    # movies_data = pd.merge(movies_data, countries, how='inner',left_on=["movieid"],right_on=['movieid'])

    # movies_data = pd.merge(movies_data, writers, how='left',left_on=["writerid"],right_on=['writerid'])
    # movies_data = pd.merge(movies_data, producers, how='left',left_on=["producerid"],right_on=['producerid'])
    # movies_data = pd.merge(movies_data, editors, how='left',left_on=["editorid"],right_on=['editorid'])
    # movies_data = pd.merge(movies_data, directors, how='left',left_on=["directorid"],right_on=['directorid'])

    data = pd.merge(actors, movies_data, how='inner',left_on=["actorid"],right_on=['actorid'])
    
    return data

def create_join_Lahman_2014(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    allstarfull = pd.read_csv(f"{path}/allstarfull.csv", low_memory=False, encoding="ISO-8859-1")
    appearances = pd.read_csv(f"{path}/appearances.csv", low_memory=False, encoding="ISO-8859-1")
    awardsmanagers = pd.read_csv(f"{path}/awardsmanagers.csv", low_memory=False, encoding="ISO-8859-1")
    awardsplayers = pd.read_csv(f"{path}/awardsplayers.csv", low_memory=False, encoding="ISO-8859-1")
    awardssharemanagers = pd.read_csv(f"{path}/awardssharemanagers.csv", low_memory=False, encoding="ISO-8859-1")
    awardsshareplayers = pd.read_csv(f"{path}/awardsshareplayers.csv", low_memory=False, encoding="ISO-8859-1")
    batting = pd.read_csv(f"{path}/batting.csv", low_memory=False, encoding="ISO-8859-1")
    battingpost = pd.read_csv(f"{path}/battingpost.csv", low_memory=False, encoding="ISO-8859-1")
    els_teamnames = pd.read_csv(f"{path}/els_teamnames.csv", low_memory=False, encoding="ISO-8859-1")
    fielding = pd.read_csv(f"{path}/fielding.csv", low_memory=False, encoding="ISO-8859-1")
    fieldingof = pd.read_csv(f"{path}/fieldingof.csv", low_memory=False, encoding="ISO-8859-1")
    fieldingpost = pd.read_csv(f"{path}/fieldingpost.csv", low_memory=False, encoding="ISO-8859-1")
    halloffame = pd.read_csv(f"{path}/halloffame.csv", low_memory=False, encoding="ISO-8859-1")
    managers = pd.read_csv(f"{path}/managers.csv", low_memory=False, encoding="ISO-8859-1")
    managershalf = pd.read_csv(f"{path}/managershalf.csv", low_memory=False, encoding="ISO-8859-1")
    pitching = pd.read_csv(f"{path}/pitching.csv", low_memory=False, encoding="ISO-8859-1")
    pitchingpost = pd.read_csv(f"{path}/pitchingpost.csv", low_memory=False, encoding="ISO-8859-1")
    players = pd.read_csv(f"{path}/players.csv", low_memory=False, encoding="ISO-8859-1")
    salaries = pd.read_csv(f"{path}/salaries.csv", low_memory=False, encoding="ISO-8859-1")
    schools = pd.read_csv(f"{path}/schools.csv", low_memory=False, encoding="ISO-8859-1")
    schoolsplayers = pd.read_csv(f"{path}/schoolsplayers.csv", low_memory=False, encoding="ISO-8859-1")
    seriespost= pd.read_csv(f"{path}/seriespost.csv", low_memory=False, encoding="ISO-8859-1")
    teams = pd.read_csv(f"{path}/teams.csv", low_memory=False, encoding="ISO-8859-1")
    teamsfranchises = pd.read_csv(f"{path}/teamsfranchises.csv", low_memory=False, encoding="ISO-8859-1")
    teamshalf = pd.read_csv(f"{path}/teamshalf.csv", low_memory=False, encoding="ISO-8859-1")

    return None

def create_join_Walmart(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    key = pd.read_csv(f"{path}/key.csv", low_memory=False, encoding="ISO-8859-1")
    station = pd.read_csv(f"{path}/station.csv", low_memory=False, encoding="ISO-8859-1")
    train = pd.read_csv(f"{path}/train.csv", low_memory=False, encoding="ISO-8859-1")
    weather = pd.read_csv(f"{path}/weather.csv", low_memory=False, encoding="ISO-8859-1")

    data = pd.merge(train, key, how='inner',left_on=["store_nbr"],right_on=['store_nbr'])
    data = pd.merge(data, station, how='inner',left_on=["station_nbr"],right_on=['station_nbr'])
    data = pd.merge(data, weather, how='inner',left_on=["station_nbr"],right_on=['station_nbr'])

    return data
   


def create_join_IMDB_IJS(root_path: str, dataset_name: str):
    path = f'{root_path}/{dataset_name}'
    actors = pd.read_csv(f"{path}/actors.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"id":"actor_id"})
    directors = pd.read_csv(f"{path}/directors.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"id":"director_id"})
    directors_genres = pd.read_csv(f"{path}/directors_genres.csv", low_memory=False, encoding="ISO-8859-1")
    movies = pd.read_csv(f"{path}/movies.csv", low_memory=False, encoding="ISO-8859-1").rename(columns={"id":"movie_id"})
    movies_directors = pd.read_csv(f"{path}/movies_directors.csv", low_memory=False, encoding="ISO-8859-1")
    movies_genres = pd.read_csv(f"{path}/movies_genres.csv", low_memory=False, encoding="ISO-8859-1")
    roles = pd.read_csv(f"{path}/roles.csv", low_memory=False, encoding="ISO-8859-1")

    data = pd.merge(actors, roles, how='inner',left_on=["actor_id"],right_on=['actor_id'])
    data = pd.merge(data, movies, how='inner',left_on=["movie_id"],right_on=['movie_id'])
    data = pd.merge(data, movies_genres, how='inner',left_on=["movie_id"],right_on=['movie_id'])
    data = pd.merge(data, movies_directors, how='inner',left_on=["movie_id"],right_on=['movie_id'])
    data = pd.merge(data, directors, how='inner',left_on=["director_id"],right_on=['director_id'])
    data = pd.merge(data, directors_genres, how='inner',left_on=["director_id"],right_on=['director_id'])

    return data


if __name__ == '__main__':
    args = parse_arguments()

    if args.target_table is None:
        args.target_table = args.dataset_name
    
    write_data = False
    # Read dataset
    if args.multi_table == 'True' and args.mtos == 'True':
        dependency_file = f"{args.catalog_root_path}/{args.dataset_name}/dependency.yaml"
        relations = load_dependency_info(dependency_file=dependency_file, datasource_name= args.dataset_name)
        data = create_join(root_path=args.dataset_root_path, dataset_name=args.dataset_name, main_table=args.target_table, relations=relations)
        args.multi_table = 'False'
        args.target_table = args.dataset_name
        write_data = True

    else:
        data = pd.read_csv(f"{args.dataset_root_path}/{args.dataset_name}/{args.target_table}.csv", low_memory=False, encoding='UTF-8')#encoding="ISO-8859-1"
        write_data = False

    # Split and save original dataset
    nrows, ncols, number_classes = get_metadata(data=data, target_attribute=args.target_attribute)
    split_data_save(data=data, ds_name=args.dataset_name,out_path= args.data_out_path, target_table=args.target_table, write_data=write_data)
    save_config(dataset_name=args.dataset_name, target=args.target_attribute, task_type=args.task_type, data_out_path=args.data_out_path, description=args.dataset_description, target_table=args.target_table, multi_table=args.multi_table)
