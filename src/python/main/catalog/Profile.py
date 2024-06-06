import json


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
        if category_values is not None and 0 < len(category_values) < 50:
            self.is_categorical = True
            self.categorical_values = category_values
            self.categorical_values_ratio = category_values_ratio

        self.nrows = nrows

    def deactivate(self):
        self.is_active = False


def load_JSON_profile_info(file_name: str):
    with open(file_name, 'r') as file:
        raw_data = file.read().replace('\n', '')
        json_data = json.loads(raw_data)
        profile_info = ProfileInfo(**json_data)
        return profile_info
