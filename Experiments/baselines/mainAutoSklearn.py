from util.main import parse_arguments
from AutoSklearnAutoML.AutoSklearnAutoML import AutoSklearnAutoML
from AutoSklearnAutoML.AutoSklearnAutoMLACC import AutoSklearnAutoMLACC

if __name__ == '__main__':
    args = parse_arguments()
    for i in range(1, args.iteration + 1):
        try:
            args.config.max_runtime_seconds = args.max_runtime_seconds * i
            args.config.iteration = i
            if args.dataset_name != "EU-IT":
                ml = AutoSklearnAutoML(dataset=args.dataset, config=args.config)
            else:
                ml = AutoSklearnAutoMLACC(dataset=args.dataset, config=args.config)
            ml.run()
        except Exception as ex:
            continue