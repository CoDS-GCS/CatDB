from runcode.RunCode import RunCode
from util.LogResults import save_log
import time
from util.FileHandler import read_text_file_line_by_line


def run_local_pipeline(args, file_name, run_mode):
    time_execute = 0
    final_status = False

    code = read_text_file_line_by_line(file_name)
    code = code.replace("train.csv", args.data_source_train_path)
    code = code.replace("test.csv", args.data_source_test_path)

    iteration_error = 0
    results_verified = False
    results = None
    time_start_1 = time.time()
    result = RunCode.execute_code(src=code, parse=None, run_mode=run_mode)
    time_end_1 = time.time()
    if result.get_status():
        results_verified, results = result.parse_results()
        if results_verified:
            time_execute = time_end_1 - time_start_1
            final_status = True

    if final_status:
        args.dataset_name = f"{args.dataset_name}"
        save_log(args=args, sub_task='', iteration=1, iteration_error=iteration_error, time_catalog=0,
                 time_generate=0, time_total=time_execute, time_execute=time_execute,
                 prompt_token_count=0, all_token_count=0, operation_tag=f'Run-Local-Pipeline-{args.extra_name}',
                 run_mode=run_mode, results_verified=results_verified, results=results, final_status=final_status)

    return final_status