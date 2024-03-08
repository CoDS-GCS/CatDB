def read_text_file_line_by_line(fname:str):
    try:
        with open(fname) as f:
            lines = f.readlines()
            raw = "".join(lines)
            return raw
    except Exception as ex:
        raise Exception (f"Error in reading file:\n {ex}")


def save_text_file(fname: str, data):
    try:
        f = open(fname, 'w')
        f.write(data)
        f.close()
    except Exception as ex:
        raise Exception (f"Error in save file:\n {ex}")


def save_prompt(fname: str, prompt_rule: str, prompt_msg):
    prompt_items = ["SYSTEM MESSAGE:\n",
                    prompt_rule,
                    "\n---------------------------------------\n",
                    "PROMPT TEXT:\n",
                    prompt_msg]
    prompt_data = "".join(prompt_items)
    save_text_file(fname=fname, data=prompt_data)


def save_config(fname: str, dataset_name: str, target: str, task_type: str):
    config_strs = [f"- name: {dataset_name}",
                       "  dataset:",
                       f"    train: \'{{user}}/data/{dataset_name}/{dataset_name}_train.csv\'",
                       f"    test: \'{{user}}/data/{dataset_name}/{dataset_name}_test.csv\'",
                       f"    target: {target}",
                       f"    type: {task_type}",
                       "  folds: 1",
                       "\n"]
    config_str = "\n".join(config_strs)
    save_text_file(fname=fname, data=config_str)