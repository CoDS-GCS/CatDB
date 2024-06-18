from util.main import parse_arguments
from H2OAutoML.H2O import H2O

if __name__ == '__main__':
    args = parse_arguments()
    ml = H2O(dataset=args.dataset, config=args.config)
    ml.run()