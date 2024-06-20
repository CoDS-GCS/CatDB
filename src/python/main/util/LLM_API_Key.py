import os
import time
import datetime


class LLM_API_Key(object):
    def __init__(self):
        from .Config import _llm_platform, _OPENAI, _GOOGLE, _META, _delay
        key = ""
        if _llm_platform == _OPENAI:
            key = "OPENAI_API_KEY"
        elif _llm_platform == _GOOGLE:
            key = "GOOGLE_API_KEY"
        elif _llm_platform == _META:
            key = "GROQ_API_KEY"

        self.platform_keys = {}
        aks = dict()
        for ki in range(1, 10):
            try:
                ID = f"{key}_{ki}"
                api_key = os.environ.get(ID)
                if api_key is None:
                    continue
                aks[ID] = {"count": 0, "last_time": time.time(), "api_key": api_key}
            except Exception:
                continue
        self.platform_keys[_llm_platform] = aks
        self.begin = True

    def get_API_Key(self):
        from .Config import _llm_platform, _delay, _last_API_Key
        aks = self.platform_keys[_llm_platform]
        sleep_time = _delay
        selectedID = None
        for ID in aks.keys():
            ak = aks[ID]
            if ak["api_key"] == _last_API_Key:
                continue
            diff_time = time.time() - ak["last_time"]
            if diff_time > _delay or self.begin:
                selectedID = ID
                break
            else:
                sleep_time = min(_delay-diff_time, sleep_time)

        if selectedID is not None:
            self.set_update(ID=selectedID, save_log=True)
            self.begin = False
            ak = aks[selectedID]
            return 0, ak["api_key"]
        else:
            time.sleep(sleep_time)
            return self.get_API_Key()

    def set_update(self, ID, save_log: bool=False):
        from .Config import _llm_platform, _system_log_file
        self.platform_keys[_llm_platform][ID]["count"] += 1
        self.platform_keys[_llm_platform][ID]["last_time"] = time.time()

        if save_log:
            log = f'{_llm_platform},{ID},{self.platform_keys[_llm_platform][ID]["count"]},{datetime.datetime.utcnow().isoformat()}'
            with open(_system_log_file, "a") as log_file:
                log_file.write(log+"\n")