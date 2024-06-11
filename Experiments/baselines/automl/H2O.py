from AutoML import AutoML
from util.Config import _nthreads, _jvm_memory

import os
import h2o
from h2o.automl import H2OAutoML


class H2O(AutoML):
    def __init__(self, dataset, config, *args, **kwargs):
        AutoML.__init__(dataset, config, *args, **kwargs)

    def run(self):
        print(f"\n**** H2O AutoML [v{h2o.__version__}] ****\n")

        try:
            # nthreads = os.cpu_count()
            jvm_memory = str(round(_jvm_memory * 2 / 3)) + "M"  # leaving 1/3rd of available memory for XGBoost
            max_port_range = 49151
            min_port_range = 1024
            rnd_port = os.getpid() % (max_port_range - min_port_range) + min_port_range
            port = config.framework_params.get('_port', rnd_port)

            init_params = config.framework_params.get('_init', {})

            h2o.init(nthreads=_nthreads,
                     port=port,
                     min_mem_size=jvm_memory,
                     max_mem_size=jvm_memory,
                     **init_params)

            import_kwargs = dict(escapechar='\\')
            train = None
            if version.parse(h2o.__version__) >= version.parse(
                    "3.32.1"):  # previous versions may fail to parse correctly some rare arff files using single quotes as enum/string delimiters (pandas also fails on same datasets)
                import_kwargs['quotechar'] = '"'
                train = h2o.import_file(dataset.train.path, destination_frame=frame_name('train', config),
                                        **import_kwargs)
                if train.nlevels() != dataset.domains.cardinalities:
                    h2o.remove(train)
                    train = None
                    import_kwargs['quotechar'] = "'"

            if not train:
                train = h2o.import_file(dataset.train.path, destination_frame=frame_name('train', config),
                                        **import_kwargs)
                # train.impute(method='mean')
            log.debug("Loading test data from %s.", dataset.test.path)
            test = h2o.import_file(dataset.test.path, destination_frame=frame_name('test', config), **import_kwargs)
            # test.impute(method='mean')

            if config.type == 'classification' and dataset.format == 'csv':
                train[dataset.target.index] = train[dataset.target.index].asfactor()
                test[dataset.target.index] = test[dataset.target.index].asfactor()

            log.info("Running model on task %s, fold %s.", config.name, config.fold)
            log.debug("Running H2O AutoML with a maximum time of %ss on %s core(s), optimizing %s.",
                      config.max_runtime_seconds, config.cores, sort_metric)

            aml = H2OAutoML(max_runtime_secs=config.max_runtime_seconds,
                            sort_metric=sort_metric,
                            seed=config.seed,
                            **training_params)

            monitor = (BackendMemoryMonitoring(interval_seconds=config.ext.monitoring.interval_seconds,
                                               check_on_exit=True,
                                               verbosity=config.ext.monitoring.verbosity)
                       if config.framework_params.get('_monitor_backend', False)
                       else contextlib.nullcontext()  # Py 3.7+ only
                       # else contextlib.contextmanager(lambda: (_ for _ in (0,)))()
                       )
            with Timer() as training:
                with monitor:
                    aml.train(y=dataset.target.index, training_frame=train)
            log.info(f"Finished fit in {training.duration}s.")

            if not aml.leader:
                raise FrameworkError("H2O could not produce any model in the requested time.")

            def infer(path: str):
                filename = pathlib.Path(path).name
                # H2O can't do inference on single row arff, it needs columns explicitly:
                # https://github.com/h2oai/h2o-3/issues/15572
                batch = h2o.import_file(path, col_names=train.names, destination_frame=frame_name(filename, config),
                                        **import_kwargs)
                return aml.predict(batch)

            inference_times = {}
            if config.measure_inference_time:
                inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
                log.info(f"Finished inference time measurements.")

            with Timer() as predict:
                preds = aml.predict(test)
            log.info(f"Finished predict in {predict.duration}s.")

            preds = extract_preds(preds, test, dataset=dataset)
            save_artifacts(aml, dataset=dataset, config=config)

            return result(
                output_file=config.output_predictions_file,
                predictions=preds.predictions,
                truth=preds.truth,
                probabilities=preds.probabilities,
                probabilities_labels=preds.probabilities_labels,
                models_count=len(aml.leaderboard),
                training_duration=training.duration,
                predict_duration=predict.duration,
                inference_times=inference_times,
            )

        finally:
            con = h2o.connection()
            if con:
                # h2o.remove_all()
                con.close()
                if con.local_server:
                    con.local_server.shutdown()
            # if h2o.cluster():
            #     h2o.cluster().shutdown()
