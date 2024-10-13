import os
import time
import datetime
import yaml


class LLM_API_Key(object):
    def __init__(self, api_config_path: str):
        from .Config import _llm_platform
        self.platform_keys = {}
        aks = dict()
        with open(api_config_path, "r") as f:
            try:
                configs = yaml.load(f, Loader=yaml.FullLoader)
                for conf in configs:
                    plt = conf.get('llm_platform')
                    if plt != _llm_platform:
                        continue
                    try:
                        if conf.get('llm_platform') is not None:
                            for ki in range(1, 10):
                                ID = f"key_{ki}"
                                api_key = conf.get(ID)
                                if api_key is None:
                                    continue
                                aks[ID] = {"count": 0, "last_time": time.time(), "api_key": api_key}
                    except:
                        pass

            except yaml.YAMLError as ex:
                raise Exception(ex)

        self.platform_keys[_llm_platform] = aks
        self.begin = True

    def get_API_Key(self):
        return 0, "AIzaSyCWK5Bhagheijc_6kIcr9FkdISKbGTJhL4"
        # from .Config import _llm_platform, _delay, _last_API_Key
        # aks = self.platform_keys[_llm_platform]
        # sleep_time = _delay
        # selectedID = None
        # for ID in aks.keys():
        #     ak = aks[ID]
        #     if ak["api_key"] == _last_API_Key:
        #         continue
        #     diff_time = time.time() - ak["last_time"]
        #     if diff_time > _delay or self.begin:
        #         selectedID = ID
        #         break
        #     else:
        #         sleep_time = min(_delay - diff_time, sleep_time)
        #
        # if selectedID is not None:
        #     self.set_update(ID=selectedID, save_log=True)
        #     self.begin = False
        #     ak = aks[selectedID]
        #     return 0, ak["api_key"]
        # else:
        #     time.sleep(sleep_time)
        #     return self.get_API_Key()

    def set_update(self, ID, save_log: bool = False):
        from .Config import _llm_platform, _system_log_file
        self.platform_keys[_llm_platform][ID]["count"] += 1
        self.platform_keys[_llm_platform][ID]["last_time"] = time.time()

        if save_log:
            log = f'{_llm_platform},{ID},{self.platform_keys[_llm_platform][ID]["count"]},{datetime.datetime.utcnow().isoformat()}'
            with open(_system_log_file, "a") as log_file:
                log_file.write(log + "\n")
                log_file.close()
