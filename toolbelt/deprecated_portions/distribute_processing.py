# -*- coding: utf-8 -*-'

import multiprocessing
import os
# from .utils import validate_list

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


# @validate_list
def multiprocess_function(data_in: list, func: object, n_jobs: int = 4, shared_resources: object = None) -> object:
    """
    Theis function can take a list of data and multiprocess-apply it against a function
    :param data_in: input list
    :param func: the function that the data needs to be run through
    :param n_jobs: number of processes to spawn
    :param shared_resources: any reference file needed to be passed into the function to avoid disk/GIL issues
    :return: a 2D list of given & processed value
    """
    manager = multiprocessing.Manager()
    _process_results = manager.dict()

    # break user input data into chunks by number of planned processes
    _data_segments = []
    _last_break = 0
    for x in range(n_jobs - 1):
        _data_segments.append(data_in[_last_break:_last_break + len(data_in) // n_jobs])
        _last_break = _last_break + len(data_in) // n_jobs
    _data_segments.append(data_in[_last_break:])

    # build process list
    processes = []

    # run jobs in parallel based on the _data_segments(chunks)
    for segment in _data_segments:
        proc = multiprocessing.Process(target=func, args=(segment, _process_results, shared_resources))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    _values_list = []
    _keys_list = []
    for key, value in _process_results.items():
        _values_list.append(value)
        _keys_list.append(key)
    return [list(a) for a in zip(_keys_list, _values_list)]
