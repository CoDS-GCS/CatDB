class InputArgs(object):
    def __init__(self, args):
        inputs = args[1:]
        key_values = dict()
        for item in inputs:
            kv = item.split('=')
            key_values[kv[0]] = kv[1]

        self.dataset = key_values['--dataset']
        self.target_attribute = key_values['--target_attribute']
        self.time_left = key_values['--time_left']
        self.per_run_time_limit = key_values['--per_run_time_limit']
        self.log_file_name = key_values['--log_file_name']

    def __dataset(self):
        return self.dataset

    def __target_attribute(self):
        return self.target_attribute

    def __time_left(self):
        return self.time_left

    def __per_run_time_limit(self):
        return self.per_run_time_limit

    def __log_file_name(self):
        return self.log_file_name