from util.Config import Config
from util.Data import Dataset
import os
import zipfile
import itertools
import shutil
from util.LogResults import LogResults


class AutoML(object):
    def __init__(self, dataset: Dataset, config: Config):
        self.dataset = dataset
        self.config = config
        self.automl_framework = None
        self.log_results = LogResults(dataset_name=dataset.dataset_name,
                                      task_type=dataset.task_type,
                                      time_total=config.max_runtime_seconds)

    def run(self):
        pass

    # def complex_function(x): 
    #     x = x.lower().strip()
    #     x = re.sub('[^A-Za-z0-9 ]+', ' ', x)
    #     x = re.sub("\s\s+" , " ", x)
    #     x = lemmatizer.lemmatize(x)    
    #     x = ' '.join([word for word in x.split(' ') if word not in stops])    
    #     return x
    
    def touch(self, path, as_dir=False):
        path = self.normalize_path(path)
        if not os.path.exists(path):
            dirname, basename = (path, '') if as_dir else os.path.split(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            if basename:
                open(path, 'a').close()
        os.utime(path, times=None)
        return path

    def normalize_path(self, path):
        return os.path.realpath(os.path.expanduser(path))

    def output_subdir(self, name):
        subdir = os.path.join(self.config.output_dir, name, self.config.name)
        self.touch(subdir, as_dir=True)
        return subdir

    def zip_path(self, path, dest_archive, compression=zipfile.ZIP_DEFLATED, arc_path_format='short', filter_=None):
        path = self.normalize_path(path)
        if not os.path.exists(path): return
        with zipfile.ZipFile(dest_archive, 'w', compression) as zf:
            if os.path.isfile(path):
                in_archive = os.path.basename(path)
                zf.write(path, in_archive)
            elif os.path.isdir(path):
                def add_to_archive(file, isdir):
                    if isdir: return
                    in_archive = (os.path.relpath(file, path) if arc_path_format == 'short'
                                  else os.path.relpath(file, os.path.dirname(path)) if arc_path_format == 'long'
                    else os.path.basename(file) is arc_path_format == 'flat'
                                  )
                    zf.write(file, in_archive)

                self.walk_apply(path, add_to_archive,
                           filter_=lambda p: (filter_ is None or filter_(p)) and not os.path.samefile(dest_archive, p))

    def walk_apply(self, dir_path, apply, topdown=True, max_depth=-1, filter_=None):
        dir_path = self.normalize_path(dir_path)
        for dir, subdirs, files in os.walk(dir_path, topdown=topdown):
            if max_depth >= 0:
                depth = 0 if dir == dir_path else len(str.split(os.path.relpath(dir, dir_path), os.sep))
                if depth > max_depth:
                    continue
            for p in itertools.chain(files, subdirs):
                path = os.path.join(dir, p)
                if filter_ is None or filter_(path):
                    apply(path, isdir=(p in subdirs))

    def clean_dir(self, dir_path, filter_=None):
        def delete(path, isdir):
            rm = filter_ is None or filter_(path)
            if not rm:
                return
            if isdir:
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)

        self.walk_apply(dir_path, delete, max_depth=0)



def result(output_file=None,
           predictions=None,
           truth=None,
           probabilities=None,
           probabilities_labels=None,
           error_message=None,
           models_count=None,
           **others):
    return locals()
