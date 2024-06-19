from util.main import parse_arguments
from AutogluonAutoML.AutogluonAutoML import AutogluonAutoML

if __name__ == '__main__':
    args = parse_arguments()
    ml = AutogluonAutoML(dataset=args.dataset, config=args.config)
    ml.run()