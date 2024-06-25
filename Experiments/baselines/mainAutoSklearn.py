from util.main import parse_arguments
from AutoSklearnAutoML.AutoSklearnAutoML import AutoSklearnAutoML

if __name__ == '__main__':
    args = parse_arguments()
    for i in range(1, args.iteration + 1):
        try:
            args.config.max_runtime_seconds = args.max_runtime_seconds * i
            args.config.iteration = i
            ml = AutoSklearnAutoML(dataset=args.dataset, config=args.config)
            ml.run()
        except Exception as ex:
            continue