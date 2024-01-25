from argparse import ArgumentParser
from llm.GenerateLLMCode import GenerateLLMCode


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--pipeline-in-path', type=str, default=None)
    parser.add_argument('--pipeline-out-path', type=str, default=None)
    parser.add_argument('--prompt-out-path', type=str, default=None)
    parser.add_argument('--error-message-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)

    args = parser.parse_args()

    if args.pipeline_in_path is None:
        raise Exception("--pipeline-in-path is a required parameter!")

    if args.pipeline_out_path is None:
        raise Exception("--pipeline-out-path is a required parameter!")

    if args.prompt_out_path is None:
        raise Exception("--prompt-out-path is a required parameter!")

    if args.error_message_path is None:
        raise Exception("--error-message-path is a required parameter!")

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    return args


def read_file(fname: str):
    with open(fname) as f:
        lines = f.readlines()
        raw = "\n".join(lines)
        return raw


if __name__ == '__main__':
    args = parse_arguments()
    pipeline_code = read_file(args.pipeline_in_path)
    pipeline_error = read_file(args.error_message_path)

    prompt_rules = ['You are expert in coding assistant. Your task is fix the error of this pipeline code.\n'
                    'The user will provide a pipeline code enclosed in "<CODE> pipline code will be here. </CODE>", '
                    'and an error message enclosed in "<ERROR> error message will be here. </ERROR>".',
                    'Fix the code error provided and return only the corrected pipeline without additional explanations'
                    ' regarding the resolved error.']
    prompt_contents = ["<CODE>",
                       pipeline_code,
                       "</CODE>",
                       "\n",
                       "<ERROR>",
                       pipeline_error,
                       "</ERROR>",
                       "Question: Fix the code error provided and return only the corrected pipeline without additional"
                       " explanations regarding the resolved error."]

    llm = GenerateLLMCode(model=args.llm_model)
    prompt_rule = "\n\n".join(prompt_rules)
    prompt_msg = "\n".join(prompt_contents)
    code = llm.generate_llm_code(prompt_rules=prompt_rule, prompt_message=prompt_msg)

    # Save prompt text

    f = open(args.prompt_out_path, 'w')
    f.write(f"SYSTEM MESSAGE: \n {prompt_rule} \n")
    f.write("----------------------------------------------------------------------------\n")
    f.write(f"PROMPT TEXT:\n{prompt_msg}\n")
    f.close()

    if len(code) > 100:
        f = open(args.pipeline_out_path, 'w')
        f.write(code)
        f.close()
