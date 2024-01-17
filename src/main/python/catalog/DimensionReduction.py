import multiprocessing
import threading
import math
import concurrent.futures

class SortItem(object):
    def __init__(self, i: int, j: int, value: float):
        self.i = i
        self.j = j
        self.value = value


class ReduceDimension(object):
    def __init__(self, profile_info: dict, reduction_method: str, reduce_size: int, target_attribute: str):
        self.profile_info = profile_info
        self.reduction_method = reduction_method
        self.reduce_size = reduce_size
        self.number_threads = multiprocessing.cpu_count()
        self.target_attribute = target_attribute
        self.ration_threshold = 0.1

    def get_new_profile_info(self):
        target = self.profile_info.pop(self.target_attribute)

        all_keys = [k for k in self.profile_info.keys()]
        bool_keys = []
        date_keys = []
        natural_language_text_keys = []
        named_entity_keys = []
        string_keys = []
        int_keys = []
        float_keys = []

        for k in all_keys:
            pi = self.profile_info[k]
            if pi.data_type == "boolean":
                bool_keys.append(k)

            elif pi.data_type == "date":
                date_keys.append(k)

            elif pi.data_type == "natural_language_text":
                natural_language_text_keys.append(k)

            elif pi.data_type == "string":
                string_keys.append(k)

            elif pi.data_type == "named_entity":
                named_entity_keys.append(k)

            elif pi.data_type == "int":
                int_keys.append(k)

            elif pi.data_type == "float":
                float_keys.append(k)

        self.profile_info[self.target_attribute] = target
        self.run_reduction_function(task=self.remove_constant_and_unique_columns, keys=all_keys)
        self.run_reduction_function(task=self.remove_low_ratio_columns, keys=all_keys)

        #self.run_similarity_reduction_function(task=self.calc_embedding_euclidean_distance_similarity, keys=int_keys)

        schema_info = dict()
        drop_schema_info = dict()
        for k in self.profile_info.keys():
            pi = self.profile_info[k]
            if pi.is_active:
                schema_info[k] = pi.data_type
            else:
                drop_schema_info[k] = pi.data_type
        return schema_info, drop_schema_info, self.profile_info

    def run_similarity_reduction_function(self, keys: [], task: str):

        active_keys = [k for k in keys if self.profile_info[k].is_active]
        len_keys = len(active_keys)
        ed_list = []
        for indexi in range(0, len_keys - self.number_threads):
            threads = list()
            blklen = int(math.ceil((len_keys - indexi - 1) / self.number_threads))
            for i in range(0, self.number_threads):
                if i * blklen < len_keys:
                    indexj = i * blklen + indexi + 1
                    indexj_end = min((i + 1) * blklen + indexi + 1, len_keys)
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        t = executor.submit(task, active_keys, indexi, indexj, indexj_end)
                        threads.append(t)
                else:
                    break

            for index, thread in enumerate(threads):
                ed = thread.result()
                ed_list.extend(ed)

        sort_list = sorted(ed_list, key=lambda x: x.value, reverse=False)
        # for i in range(0, min(20, len(sort_list))):
        #     print(f"{sort_list[i].i},{sort_list[i].j} >> {sort_list[i].value}")

    def run_reduction_function(self, keys: [], task: str):
        threads = list()
        len_keys = len(keys)
        if len_keys <= self.number_threads:
            task(keys)
        else:
            blklen = int(math.ceil(len_keys / self.number_threads))
            for i in range(0, self.number_threads):
                if i * blklen < len_keys:
                    begin_index = i * blklen
                    end_index = min((i + 1) * blklen, len_keys)
                    arg_val = keys[begin_index: end_index]
                    t = threading.Thread(target=task, args=(arg_val,))
                    threads.append(t)
                    t.start()
                else:
                    break

            for index, thread in enumerate(threads):
                thread.join()

    def remove_constant_and_unique_columns(self, keys: []):
        for k in keys:
            pi = self.profile_info[k]
            if pi.distinct_values_count == 1 or pi.distinct_values_count == pi.total_values_count:
                pi.deactivate()

    def remove_low_ratio_columns(self, keys: []):
        for k in keys:
            pi = self.profile_info[k]
            if pi.missing_values_count / pi.total_values_count > 1 - self.ration_threshold:
                pi.deactivate()

    def calc_embedding_euclidean_distance_similarity(self, keys: [], indexi: int, indexj: int, indexj_end: int):
        ed_list = []
        xi = self.profile_info[keys[indexi]].embedding
        for j in range(indexj, indexj_end):
            ed = self.calc_euclidean_distance(xi, self.profile_info[keys[j]].embedding)
            ed_list.append(SortItem(i=indexi, j=j, value=ed))

        #sort_list = sorted(ed_list, key=lambda x: x.value, reverse=False)
        return ed_list

    def calc_euclidean_distance(self, list1, list2):
        return sum((p - q) ** 2 for p, q in zip(list1, list2)) ** .5