
class ProfileInfo(object):
    def __init__(self,                
                 data_type=None,
                 count=0,
                 nullable = False,
                 number_of_nulls=-1,
                 is_categorical=False,
                 category_values=None,
                 category_values_len=0,
                 min_value=None,
                 max_value=None,
                 mean_value=None,
                 std_value=None,
                 is_unique=False,
                 top_value=None,
                 freq_value=None):
        self.category_values_len = category_values_len
        if category_values is None:
            category_values = []
        self.freq_value = freq_value
        self.top_value = top_value
        self.is_unique = is_unique
        self.std_value = std_value
        self.mean_value = mean_value
        self.max_value = max_value
        self.min_value = min_value
        self.category_values = category_values
        self.is_categorical = is_categorical
        self.number_of_nulls = number_of_nulls
        self.nullable = nullable
        self.data_type = data_type
        self.count = count