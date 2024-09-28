from argparse import ArgumentParser
from pipegen.DataCleaningPatch import DataCleaningPatch
from util.FileHandler import read_text_file_line_by_line


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--source-dataset-name', type=str, default=None)
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--patch-src', type=str, default=None)
    parser.add_argument('--split-dataset', type=str, default='True')

    args = parser.parse_args()

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    if args.split_dataset == 'True':
        args.split_dataset = True
    else:
        args.split_dataset = False

    return args


if __name__ == '__main__':
    args = parse_arguments()
    patch_src = read_text_file_line_by_line(args.patch_src)
    dcp = DataCleaningPatch(code=patch_src, root_data=args.root_data_path, dataset_name=args.dataset_name,
                            source_dataset_name=args.source_dataset_name, split_dataset=args.split_dataset)
    dcp.apply_patch()
