from argparse import ArgumentParser
from dataprofiling import build_catalog


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--data-profile-path', type=str, default=None)
    parser.add_argument('--source-dataset-path', type=str, default=None)

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    data = {
        "dataset_name": args.dataset_name,
        "metadata_path": args.metadata_path,
        "data_profile_path": args.data_profile_path,
        "source_dataset_path": args.source_dataset_path
    }

    catalog = build_catalog(data=data, categorical_ratio=0.05, n_workers=1, max_memory=10)
    print(catalog)

