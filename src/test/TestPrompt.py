from src.main.python.catalog.Catalog import get_data_catalog
from src.main.python.prompt.PromptBuilder import prompt_factory


if __name__ == '__main__':
    a = 11
    cat = get_data_catalog(dataset_name='data/adult.dat', file_format='csv')
    prompt = prompt_factory(catalog=cat, repr_type='TEXT', k_shot=0)
    prompt.format(example=None)
    # print(prompt.template_info)
    # print(prompt.template_question)
    a = 100
