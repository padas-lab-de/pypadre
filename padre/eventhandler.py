from pymitter import EventEmitter

eventemitter = EventEmitter()

logger_list = []


# Event handlers
def log_start_experiment(args):
    experiment = args.get('experiment', None)
    append_runs = args.get('append_runs', False)

    if experiment is not None:
        for logger in logger_list:
            logger.log_start_experiment(experiment=experiment, append_runs=append_runs)


def log_stop_experiment(args):
    experiment = args.get('experiment', None)

    if experiment is not None:
        for logger in logger_list:
            logger.log_stop_experiment(experiment=experiment)


def put_experiment_configuration(args):
    pass


def log_start_run(args):
    run = args.get('run', None)
    if run is not None:
        for logger in logger_list:
            logger.log_start_run(run=run)


def log_stop_run(args):
    run = args.get('run', None)
    if run is not None:
        for logger in logger_list:
            logger.log_stop_run(run=run)


def log_start_split(args):
    split = args.get('split', None)
    if split is not None:
        for logger in logger_list:
            logger.log_start_split(split=split)


def log_stop_split(args):
    split = args.get('split', None)
    if split is not None:
        for logger in logger_list:
            logger.log_start_split(split=split)


def log_score(args):
    pass


def log_results(args):
    pass


def log_event(args):
    pass


def log(args):
    pass


def warn(args):
    pass


def error(args):
    pass


# Binding events to event handlers
eventemitter.on("start_experiment", log_start_experiment)
eventemitter.on("stop_experiment", log_stop_experiment)

eventemitter.on("start_run", log_start_run)
eventemitter.on("stop_run", log_stop_run)

eventemitter.on("start_split", log_start_split)
eventemitter.on("stop_split", log_stop_split)

eventemitter.on("log_score", log_score)
eventemitter.on("log_results", log_results)

eventemitter.on("log_event", log_event)
eventemitter.on("log", log)
eventemitter.on("warn", warn)
eventemitter.on("error", error)

