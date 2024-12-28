from util.main import parse_arguments
from AutoGenAutoML.AutoGenAutoML import AutoGenAutoML

if __name__ == '__main__':
    args = parse_arguments(is_automl=False)
    for i in range(1, args.iteration + 1):
        ml = AutoGenAutoML(dataset=args.dataset, config=args.config, llm_model=args.llm_model, iteration=i)
        ml.run()