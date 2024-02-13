class CodeResultTemplate(object):
    def __init__(self, status: bool = True, result=None, exception: Exception = None):
        self.status = status
        self.result = result
        self.exception = exception

    def set_status(self, status: bool = True):
        self.status = status

    def set_result(self, result: None):
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
        pipeline_evl = {"Accuracy": -1, "F1_score": -1, "Log_loss": -1, "R_Squared": -1, "RMSE": -1}
        if self.status and self.result is not None:
            raw_results = self.result.splitlines()
            for rr in raw_results:
                row = rr.replace("Log loss", "Log_loss").strip().split(":")
                if row[0] in pipeline_evl.keys():
                    pipeline_evl[row[0]] = row[1].strip()

        return pipeline_evl