from util.main import parse_arguments
from FlamlAutoML.FlamlAutoML import FlamlAutoML

if __name__ == '__main__':
    args = parse_arguments()
    ml = FlamlAutoML(dataset=args.dataset, config=args.config)
    ml.run()