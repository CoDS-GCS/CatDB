import unittest
from  main.python.catalog.Catalog import load_data_source_profile

class PipelineGen(unittest.TestCase):

    def test_catalog(self):
        profile_info_path = "../data/dataset_4/data_profile"
        target_attribute = "class"
        catalog = load_data_source_profile(data_source_path=profile_info_path,
                                       file_format="JSON",
                                       reduction_method="NA",
                                       reduce_size=0,
                                       target_attribute=target_attribute)

if __name__ == '__main__':
    unittest.main()
