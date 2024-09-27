import json
import os


class ProfileInfo(object):
    def __init__(self,
                 column_id: str,
                 dataset_name: str,
                 dataset_id: str,
                 path: str,
                 table_name: str,
                 table_id: str,
                 column_name: str,
                 data_source: str,
                 data_type: str,
                 total_values_count: int,
                 distinct_values_count: int,
                 missing_values_count: int,
                 true_ratio: float,
                 min_value: float,
                 max_value: float,
                 mean: float,
                 median: float,
                 iqr: float,
                 embedding_scaling_factor: float,
                 embedding=None,
                 category_values=None,
                 category_values_ratio=None,
                 samples=None,
                 nrows: int = 0,
                 categorical_values_restricted_size=-1
                 ):
        self.distinct_values_count = distinct_values_count
        self.data_source = data_source
        self.embedding_scaling_factor = embedding_scaling_factor
        if embedding is None:
            embedding = []
        self.embedding = embedding
        self.iqr = iqr
        self.median = median
        self.mean = mean
        self.max_value = max_value
        self.min_value = min_value
        self.true_ratio = true_ratio
        self.missing_values_count = missing_values_count
        self.total_values_count = total_values_count
        self.data_type = data_type
        self.column_name = column_name
        self.table_id = table_id
        self.table_name = table_name
        self.path = path
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.column_id = column_id
        self.is_active = True
        if self.data_type == "boolean":
            self.short_data_type = "bool"
        elif self.data_type == "string" or self.data_type == "natural_language_text" or self.data_type == "named_entity":
            self.short_data_type = "str"
        else:
            self.short_data_type = self.data_type

        self.samples = samples
        if samples is None or len(samples) == 0:
            self.samples = None

        self.categorical_values = None
        self.categorical_values_ratio = None
        self.is_categorical = False

        if category_values is not None and categorical_values_restricted_size == -1:
            rsize = len(category_values)
        else:
            rsize = categorical_values_restricted_size
        if category_values is not None and 0 < len(category_values) <= rsize:
            self.is_categorical = True
            self.categorical_values = category_values
            self.categorical_values_ratio = category_values_ratio
        self.nrows = nrows

    def deactivate(self):
        self.is_active = False


class ProfileInfoUpdate(object):
    def __init__(self, column_name: str, column_values: []):
        self.column_name = column_name.replace("/","###")
        self.column_values = column_values

    def to_dict(self):
        profile_dict = {'column_name': self.column_name,
                        'column_values': [v for v in self.column_values]}
        return profile_dict

    def save_profile(self, profile_update_dir):
        os.makedirs(profile_update_dir, exist_ok=True)
        with open(os.path.join(profile_update_dir, f'{self.column_name}.json'), 'w') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)


def load_JSON_profile_info(file_name: str, categorical_values_restricted_size):
    with open(file_name, 'r') as file:
        raw_data = file.read().replace('\n', '')
        json_data = json.loads(raw_data)
        profile_info = ProfileInfo(**json_data, categorical_values_restricted_size=categorical_values_restricted_size)
        return profile_info


def load_JSON_profile_info_with_update(data_profile_update: str, file_name: str):
    with open(file_name, 'r') as file:
        raw_data = file.read().replace('\n', '')
        json_data = json.loads(raw_data)
        profile_info = ProfileInfo(**json_data, categorical_values_restricted_size=-1)
        # check the update
        update = load_JSON_profile_info_update(f"{data_profile_update}/{profile_info.column_name.replace('###','/')}.json")
        if update is not None:
            profile_info.is_categorical = True
            profile_info.categorical_values = update.column_values
            profile_info.samples = update.column_values
        return profile_info


def load_JSON_profile_info_update(file_name: str):
    if os.path.isfile(file_name):
        with open(file_name, 'r') as file:
            raw_data = file.read().replace('\n', '')
            json_data = json.loads(raw_data)
            profile_info = ProfileInfoUpdate(**json_data)
            return profile_info
    return None
