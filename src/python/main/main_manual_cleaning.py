from argparse import ArgumentParser
from util.DedubplicateData import DeduplicateData
import pandas as pd


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--target-attribute', type=str, default=None)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--data-featurized-path', type=str, default=None)
    parser.add_argument('--data-input-down-path', type=str, default=None)

    args = parser.parse_args()

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    args.data_source_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_name}.csv"
    return args


if __name__ == '__main__':
    args = parse_arguments()

    data_input_down = pd.read_csv(args.data_input_down_path)
    duplicate_column = data_input_down.columns.values.tolist()[0]

    args.target_attribute = "Position "
    dataDownstream = pd.read_csv(args.data_source_path)
    dataDownstream = dataDownstream.sample(frac=1, random_state=100)
    dataDownstream = dataDownstream[dataDownstream[args.target_attribute].notna()]
    dataDownstream = dataDownstream[dataDownstream[duplicate_column].notna()]
    dataDownstream = dataDownstream.fillna('0')
    dataDownstream = dataDownstream.reset_index(drop=True)
    # dataDownstream = dataDownstream.drop(args.target_attribute, axis=1)

    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    if args.data_source_path not in {'Midwest-Survey'}:
        labels_file = pd.read_csv(args.data_featurized_path)
        for index, row in labels_file.iterrows():
            attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicate_column:
            attribute_dic[x] = 7
        elif str(x).isdigit():
            attribute_dic[x] = 0

    # attribute_names = dataDownstream.columns.values.tolist()
    y_cur = []
    for x in attribute_names: y_cur.append(attribute_dic[x])

    clean_data = DeduplicateData(data_down_stream=dataDownstream, input_down_path=data_input_down,
                                     duplicate_column=duplicate_column)
