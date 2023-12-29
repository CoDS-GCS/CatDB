from src.main.python.catalog.Catalog import load_data_source_profile
from src.main.python.prompt.PromptBuilder import prompt_factory


if __name__ == '__main__':
    a = 11
    cat = load_data_source_profile(data_source_path="/home/saeed/projects/kglids/storage/profiles/tus_profiles_fine_grained/", file_format="JSON")
    prompt = prompt_factory(catalog=cat, repr_type='TEXT', k_shot=0, iterative=1, target_attribute='class')
    p = prompt.format(example=None)
    print(p)
    # print(prompt.template_info)
    # print(prompt.template_question)
    a = 100
