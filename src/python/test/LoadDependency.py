import unittest
from catalog.Dependency import load_dependency_info
class PipelineGen(unittest.TestCase):
    dependency_info_path = "/home/saeed/Documents/Github/CatDB/Experiments/metadata/Accident/dependency.yaml"
    datasource_name = "Accident"

    def test_load_dependency(self):
        dep = load_dependency_info(dependency_file=self.dependency_info_path, datasource_name=self.datasource_name)


if __name__ == '__main__':
    unittest.main()
