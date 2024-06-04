class CodeResultTemplate(object):
    def __init__(self, status: bool = True, result=None, exception: Exception = None,
                 run_mode: str = 'generate-and-run', code: str = None,
                 error_class=None, error_type=None, error_value=None, error_detail=None, error_exception=None):
        self.status = status
        self.result = result
        self.exception = exception
        self.run_mode = run_mode
        self.code = code
        self.error_class = error_class
        self.error_type = error_type
        self.error_value = error_value
        self.error_detail = error_detail
        self.error_exception = error_exception

    def set_status(self, status: bool = True):
        self.status = status

    def set_result(self, result=None):
        self.result = result

    def set_exception(self, exception: Exception = None):
        self.exception = exception

    def set_error_class(self, error_class):
        self.error_class = error_class

    def set_error_type(self, error_type):
        self.error_type = error_type

    def set_error_value(self, error_value):
        self.error_type = error_value

    def set_error_detail(self, error_detail):
        self.error_detail = error_detail

    def set_error_exception(self, error_exception):
        self.error_exception = error_exception

    def get_status(self):
        return self.status

    def get_results(self):
        return self.result

    def get_exception(self):
        return self.exception

    def get_error_class(self):
        return self.error_class

    def get_error_type(self):
        return self.error_type

    def get_error_value(self):
        return self.error_type

    def get_error_detail(self):
        return self.error_detail

    def get_error_exception(self):
        return self.error_exception

    def parse_results(self):
        print(self.result)
        if self.run_mode == 'generate-and-run':  # __gen_run_mode:
            pipeline_evl = {"Train_AUC": -2,
                            "Train_AUC_OVO": -2,
                            "Train_AUC_OVR": -2,
                            "Train_Accuracy": -2,
                            "Train_F1_score": -2,
                            "Train_Log_loss": -2,
                            "Train_R_Squared": -2,
                            "Train_RMSE": -2,
                            "Test_AUC": -2,
                            "Test_AUC_OVO": -2,
                            "Test_AUC_OVR": -2,
                            "Test_Accuracy": -2,
                            "Test_F1_score": -2,
                            "Test_Log_loss": -2,
                            "Test_R_Squared": -2,
                            "Test_RMSE": -2}
            if self.status and self.result is not None:
                raw_results = self.result.splitlines()
                for rr in raw_results:
                    row = rr.replace("LogResults.py loss", "Log_loss").strip().split(":")
                    if row[0] in pipeline_evl.keys():
                        pipeline_evl[row[0]] = row[1].strip()
            verify = False
            # verify results
            if ((pipeline_evl["Train_AUC"] !=-2 and # binary classification
                pipeline_evl["Train_Accuracy"] != -2 and
                pipeline_evl["Train_F1_score"] != -2 and
                pipeline_evl["Test_AUC"] != -2 and
                pipeline_evl["Test_Accuracy"] != -2 and
                pipeline_evl["Test_F1_score"] != -2) or
                (pipeline_evl["Train_AUC_OVO"] !=-2 and # multiclass classification
                 pipeline_evl["Train_AUC_OVR"] !=-2 and
                 pipeline_evl["Train_Accuracy"] != -2 and
                 pipeline_evl["Train_Log_loss"] !=-2 and
                 pipeline_evl["Test_AUC_OVO"] != -2 and
                 pipeline_evl["Test_AUC_OVR"] != -2 and
                 pipeline_evl["Test_Accuracy"] != -2 and
                 pipeline_evl["Test_Log_loss"] != -2) or
                (pipeline_evl["Train_R_Squared"] != -2 and # regression
                 pipeline_evl["Train_RMSE"] != -2 and
                 pipeline_evl["Test_R_Squared"] != -2 and
                 pipeline_evl["Test_RMSE"] != -2 )):
                verify = True

            return verify, pipeline_evl
        else:
            return True, self.code
