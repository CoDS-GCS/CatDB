from util.main import parse_arguments
from AutoSklearnAutoML.AutoSklearnAutoML import AutoSklearnAutoML

if __name__ == '__main__':
    args = parse_arguments()
    ml = AutoSklearnAutoML(dataset=args.dataset, config=args.config)
    ml.run()