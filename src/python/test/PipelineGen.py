import unittest
from catalog.Catalog import load_data_source_profile
from pipegen.GeneratePipeLine import GeneratePipeLine


class PipelineGen(unittest.TestCase):
    profile_info_path = "data/dataset_4/data_profile"
    target_attribute = "class"
    catalog = None

    def test_pipeline_gen(self):
        catalog = load_data_source_profile(data_source_path=self.profile_info_path,
                                           file_format="JSON",
                                           reduction_method="NA",
                                           reduce_size=0,
                                           target_attribute=self.target_attribute)

        gpl = GeneratePipeLine(catalog=catalog, target_attribute=self.target_attribute, train_fname="ds_train.csv",
                               test_fname="ds_test.csv")
        src = gpl.get_pipeline()
        print(src)


if __name__ == '__main__':
    unittest.main()
