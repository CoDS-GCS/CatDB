from util.main import parse_arguments
from H2OAutoML.H2O import H2O
from H2OAutoML.H2OACC import H2OACC

if __name__ == '__main__':
    args = parse_arguments()
    for i in range(1, args.iteration + 1):
        args.config.max_runtime_seconds = args.max_runtime_seconds * i
        args.config.iteration = i
        if args.dataset_name != "EU-IT":
            ml = H2O(dataset=args.dataset, config=args.config)
        else:
            ml = H2OACC(dataset=args.dataset, config=args.config)
        ml.run()