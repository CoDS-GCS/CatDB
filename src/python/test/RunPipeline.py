import unittest
from main.util.FileHandler import read_text_file_line_by_line
from main.runcode.RunCode import RunCode


class PipelineGen(unittest.TestCase):
    # pipeline_paths = ["dataset_1_rnc-ALL-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-CATEGORICAL_VALUE-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-DISTINCT-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-DISTINCT_MISSING_VALUE-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-MISSING_VALUE-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-MISSING_VALUE_CATEGORICAL_VALUE-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-MISSING_VALUE_NUMERIC_STATISTIC-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-NUMERIC_STATISTIC-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-NUMERIC_STATISTIC_CATEGORICAL_VALUE-Random-0-SHOT-gpt-4.py",
    #                   "dataset_1_rnc-SCHEMA-Random-0-SHOT-gpt-4.py"]
    pipeline_paths = ["gpt-5-CatDB-Random-0-SHOT.py"]
    rc = RunCode()

    def test_execute_code(self):

        # for path in self.pipeline_paths:
            path = self.pipeline_paths[0]
            src = read_text_file_line_by_line(f"code/{path}")
            result = self.rc.execute_code(src=src)
            print(result.get_results())
            print(f"{path}  >> {result.parse_results()}")


if __name__ == '__main__':
    unittest.main()
