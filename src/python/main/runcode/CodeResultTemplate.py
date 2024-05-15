
class CodeResultTemplate(object):
    def __init__(self, status: bool = True, result=None, exception: Exception = None,
                 run_mode: str = 'generate-and-run', code: str = None):
        self.status = status
        self.result = result
        self.exception = exception
        self.run_mode = run_mode
        self.code = code

    def set_status(self, status: bool = True):
        self.status = status

    def set_result(self, result=None):
        self.result = result

    def set_exception(self, exception: Exception = None):
        self.exception = exception

    def get_status(self):
        return self.status

    def get_results(self):
        return self.result

    def get_exception(self):
        return self.exception

    def parse_results(self):
        from util.Config import __gen_run_mode
        if self.run_mode == __gen_run_mode:
            pipeline_evl = {"Train_Accuracy": -1,
                            "Train_F1_score": -1,
                            "Train_Log_loss": -1,
                            "Train_R_Squared": -1,
                            "Train_RMSE": -1,
                            "Test_Accuracy": -1,
                            "Test_F1_score": -1,
                            "Test_Log_loss": -1,
                            "Test_R_Squared": -1,
                            "Test_RMSE": -1}
            if self.status and self.result is not None:
                raw_results = self.result.splitlines()
                for rr in raw_results:
                    row = rr.replace("LogResults.py loss", "Log_loss").strip().split(":")
                    if row[0] in pipeline_evl.keys():
                        pipeline_evl[row[0]] = row[1].strip()

            return pipeline_evl
        else:
            return self.code
