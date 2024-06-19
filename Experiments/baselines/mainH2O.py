from util.main import parse_arguments
from H2OAutoML.H2O import H2O

if __name__ == '__main__':
    args = parse_arguments()
    for i in range(1, args.iteration + 1):
        args.config.max_runtime_seconds = args.max_runtime_seconds * i
        args.config.iteration = i
        ml = H2O(dataset=args.dataset, config=args.config)
        ml.run()