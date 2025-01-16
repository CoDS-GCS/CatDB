from util.main import parse_arguments
from AutogluonAutoML.AutogluonAutoML import AutogluonAutoML
from AutogluonAutoML.AutogluonAutoMLACC import AutogluonAutoMLACC

if __name__ == '__main__':
    args = parse_arguments()
    for i in range(1, args.iteration + 1):
        args.config.max_runtime_seconds = args.max_runtime_seconds * i
        args.config.iteration = i
        if args.dataset_name != "EU-IT":
            ml = AutogluonAutoML(dataset=args.dataset, config=args.config)
        else:
            ml = AutogluonAutoMLACC(dataset=args.dataset, config=args.config)
        ml.run()